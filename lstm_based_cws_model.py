# -*- coding: utf-8 -*-

#Author: Jay Yip
#Date 04Mar2017


"""Chinese words segmentation model based on aclweb.org/anthology/D15-1141"""




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
from ops import input_ops


class LSTMCWS(object):
    """docstring for LSTMCWS"""
    def __init__(self, config, mode):
        """
        Init mode.

        Args:
            Config: configuration object
            mode: 'train', 'eval' or 'inference'
        """

        self.config = config
        self.mode = mode

        #Set up reader
        self.reader = tf.TFRecordReader()

        #Set up initializer
        self.initializer = tf.contrib.layers.xavier_initializer(uniform=True, 
            seed=None, dtype=tf.float32)

        #Set up sequence embeddings with the shape of [batch_size, padded_length, embedding_size]
        self.seq_embedding = None

        #Set up batch losses for tracking performance with the length of batch_size * padded_length
        self.batch_losses = None

        #Set up global step tensor
        self.global_step = None

        #Set up embedding table
        self.embedding_tensor = None

    def is_training(self):
        return self.mode == 'train'

    def build_inputs(self):
        """
        Input prefetching, preprocessing and batching for trianing

        For inference mode, input seqs and input mask needs to be provided.

        Returns:
            self.input_seqs: A tensor of Input sequence to seq_lstm with the shape of [batch_size, padding_size]
            self.tag_seqs: A tensor of output sequence to seq_lstm with the shape of [batch_size, padding_size]
            self.tag_input_seq: A tensor of input sequence to tag inference model with the shape of [batch_size, padding_size -1]
            self.tag_output_seq: A tensor of input sequence to tag inference model with the shape of [batch_size, padding_size -1]
        """
        #Get all TFRecord path into a list
        data_files = []
        file_pattern = '*.TFRecord'
        data_files.extend(tf.gfile.Glob(file_pattern))
        if not data_files:
          tf.logging.fatal("Found no input files matching %s", file_pattern)
        else:
          tf.logging.info("Prefetching values from %d files matching %s",
                          len(data_files), file_pattern)


        #Create file queue
        if self.is_training():
            filename_queue = tf.train.string_input_producer(
                data_files, shuffle = True, capacity = 16, name = 'filename_input_queue')

            #Create example queue
            example_queue = input_ops.example_queue_shuffle(self.reader, filename_queue, 
                self.is_training(), capacity = 50000, num_reader_threads = self.config.num_preprocess_thread)

            #Parse one example
            input_seq_queue, tag_seq_queue = input_ops.parse_example_queue(example_queue, 
                self.config.context_feature_name, self.config.tag_feature_name)

            
            seq_length = tf.expand_dims(tf.subtract(tf.shape(tag_seq_queue)[0], 1),0)
            indicator = tf.ones(seq_length, dtype=tf.int32)

            input_seqs, tag_seqs, input_mask = tf.train.batch(
                [input_seq_queue, tag_seq_queue, indicator], 
                batch_size=self.config.batch_size,
                capacity=50000,
                dynamic_pad=True,
                name="batch_and_pad")

        else:
            #Inference
            input_seq_feed = tf.get_default_graph().get_tensor_by_name("input_seq_feed:0")
            input_seqs = tf.expand_dims(input_seq_feed, 0)
            input_mask = tf.ones(tf.shape(input_seqs), dtype=tf.int32)

            tag_seqs = None

        
        self.input_seqs = input_seqs
        self.tag_seqs = tag_seqs
        self.input_mask = input_mask


    def build_chr_embedding(self):
        """
        Build Chinese character embedding

        Returns:
            self.seq_embedding: A tensor with the shape of [batch_size, padding_size, embedding_size]
            self.tag_embedding: A tensor with the shape of [batch_size, padding_size, num_tag]
        """
        with tf.variable_scope('seq_embedding', reuse = True) as seq_embedding_scope:
            #chr_embedding = tf.Variable(self.embedding_tensor, name="chr_embedding")
            chr_embedding = tf.get_variable(name="chr_embedding", validate_shape = False)

            seq_embedding = tf.nn.embedding_lookup(chr_embedding, self.input_seqs)
            if self.is_training():
                tag_embedding = tf.one_hot(self.tag_seqs, self.config.num_tag)
            else:
                tag_embedding = None


        self.seq_embedding = seq_embedding
        self.tag_embedding = tag_embedding



    def build_lstm_model(self):
        """
        Build model.

        Returns:
            self.logit: A tensor containing the probability of prediction with the shape of [batch_size, padding_size, num_tag]
        """

        #Setup LSTM Cell
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units = self.config.num_lstm_units, state_is_tuple = True)

        #Dropout when training
        if self.is_training():
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell,
                input_keep_prob = self.config.lstm_dropout_keep_prob,
                output_keep_prob = self.config.lstm_dropout_keep_prob)

        self.seq_embedding.set_shape([None, None, self.config.embedding_size])

        with tf.variable_scope('seq_lstm') as lstm_scope:
            #Init lstm
            #Get the initial state for dynamic_rnn
            if self.is_training():
                init_state = lstm_cell.zero_state(batch_size = self.config.batch_size, dtype = tf.float32)
            else:
                init_state = lstm_cell.zero_state(batch_size = 1, dtype = tf.float32)
                self.seq_embedding = tf.cast(self.seq_embedding, tf.float32)


            #Run LSTM with sequence_length timesteps
            sequence_length = tf.add(tf.reduce_sum(self.input_mask, 1), 1)
            lstm_output, _ = tf.nn.dynamic_rnn(cell = lstm_cell,
                inputs = self.seq_embedding,
                sequence_length = sequence_length,
                initial_state = init_state,
                dtype = tf.float32,
                scope = lstm_scope)


        self.lstm_output = lstm_output

    def build_sentence_score_loss(self):
        """
        Use CRF log likelihood to get sentence score and loss
        """
        #Fully connected layer to get logit
        with tf.variable_scope('logit') as logit_scope:
            logit = tf.contrib.layers.fully_connected(inputs = self.lstm_output,
                num_outputs = self.config.num_tag,
                activation_fn = None,
                weights_initializer = self.initializer,
                scope = logit_scope)

        if not self.is_training():
            # In inference, logit will be returned since the decode fn only takes numpy array.
            logit = tf.squeeze(logit)

            self.logit = logit

        else:
            with tf.variable_scope('tag_inf') as tag_scope:
                sequence_length = tf.reduce_sum(self.input_mask, 1)
                sentence_likelihood, transition_param = tf.contrib.crf.crf_log_likelihood(inputs = logit,
                    tag_indices = tf.to_int32(self.tag_seqs),
                    sequence_lengths = sequence_length)

                share_transition_param = tf.get_variable(name = 'transition_param', 
                    shape = [self.config.num_tag,self.config.num_tag],
                    initializer = self.initializer)
                ass_op = share_transition_param.assign(transition_param)

            batch_loss = tf.reduce_sum(-sentence_likelihood)

            #Add to total loss
            tf.losses.add_loss(batch_loss)

            #Get total loss
            total_loss = tf.losses.get_total_loss()

            tf.summary.scalar('losses/batch_loss', batch_loss)
            tf.summary.scalar('losses/total_loss', total_loss)

            #For test only
            self.logit = logit
            self.sentence_likelihood = sentence_likelihood

            #Output loss
            self.batch_loss = batch_loss
            self.total_loss = total_loss


    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step

    def build(self):
        """Create all ops for model"""
        self.build_inputs()
        self.build_chr_embedding()
        self.build_lstm_model()
        self.build_sentence_score_loss()
        self.setup_global_step()
