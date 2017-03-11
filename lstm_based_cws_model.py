# -*- coding: utf-8 -*-

#Author: Jay Yip
#Date 04Mar2017

#TODO:
#1, Inference and Evaluation in build_input

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
        self.embedding_placeholder = tf.placeholder(tf.float32, [self.config.vocab_size, self.config.embedding_size])

    def is_training(self):
        return self.mode == 'train'

    def build_inputs(self):
        """
        Input prefetching, preprocessing and batching

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

        else:
            #TODO
            pass

        #Create example queue
        example_queue = input_ops.example_queue_shuffle(reader, filename_queue, 
            self.is_training(), capacity = 50000, num_reader_threads = self.config.num_preprocess_thread)

        #Parse one example
        input_seq_queue, tag_seq_queue = input_ops.parse_example_queue(example_queue, 
            self.config.context_feature_name, self.config.tag_feature_name)

        #Right shift the tag seq as target seq
        seq_length = tf.expand_dims(tf.subtract(tf.shape(tag_seq_queue)[0], 1),0)

        # The input seq is from 0 to t-1
        # The output seq is from 1 to t
        indicator = tf.ones(seq_length, dtype=tf.int32)
        tag_input_seq_queue = tf.slice(tag_seq_queue, [0], seq_length)
        tag_output_seq_queue = tf.slice(tag_seq_queue, [1], seq_length)

        #Use shuffle batch to create shuffle queue and get batch examples
        input_seqs, tag_seqs, tag_input_seq, tag_output_seq, input_mask = tf.train.batch(
            [input_seq_queue, tag_seq_queue, tag_input_seq_queue, tag_output_seq_queue, indicator], 
            batch_size=config.batch_size,
            capacity=50000,
            dynamic_pad=True,
            name="batch_and_pad")
        
        self.input_seqs = input_seqs
        self.tag_seqs = tag_seqs
        self.tag_input_seq = tag_input_seq
        self.tag_output_seq = tag_output_seq

    def build_chr_embedding(self):
        """
        Build Chinese character embedding

        Returns:
            self.seq_embedding: A tensor with the shape of [batch_size, padding_size, embedding_size]
            self.tag_embedding: A tensor with the shape of [batch_size, padding_size, num_tag]
        """
        with tf.variable_scope('seq_embedding') as seq_embedding_scope:
            chr_embedding = tf.Variable(tf.constant(0.0, shape=[self.config.embedding_size, self.config.embedding_size]),
                validate_shape=False, trainable=False, name="chr_embedding")

            chr_assign_op = chr_embedding.assign(self.embedding_placeholder)
            seq_embedding = tf.nn.embedding_lookup(chr_embedding, self.input_seqs)

            tag_embedding = tf.one_hot(self.tag_seqs, self.config.num_tag)
            tag_input_embedding = tf.one_hot(self.tag_input_seq, self.config.num_tag)
            tag_output_embedding = tf.one_hot(self.tag_output_seq, self.config.num_tag)


        self.seq_embedding = seq_embedding
        self.tag_embedding = tag_embedding
        self.tag_input_embedding = tag_input_embedding
        self.tag_output_embedding = tag_output_embedding


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

        self.seq_embedding.set_shape([None, None, config.embedding_size])

        with tf.variable_scope('seq_lstm') as lstm_scope:
            #Init lstm
            #Get the initial state for dynamic_rnn
            init_state = lstm_cell.zero_state(batch_size = self.config.batch_size, dtype = tf.float32)

            if self.mode != 'inference':

                #Run LSTM with sequence_length timesteps
                sequence_length = tf.add(tf.reduce_sum(input_mask, 1), 1)
                lstm_output, _ = tf.nn.dynamic_rnn(cell = lstm_cell,
                    inputs = self.seq_embedding,
                    sequence_length = sequence_length,
                    initial_state = init_state,
                    dtype = tf.float32,
                    scope = lstm_scope)

            else:

                #TODO
                #Reference.

        # Stack batches vertically.
        #lstm_output = tf.reshape(lstm_output, [-1, lstm_cell.output_size])


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
                weights_initializer = initializer,
                scope = logit_scope)

        if self.mode == 'inference':
            #Get maximum sentence score
            pass
        else:
            with tf.variable_scope('tag_inf') as tag_scope:

                sequence_length = tf.reduce_sum(input_mask, 1)
                sentence_likelihood, transition_param = tf.contrib.crf.crf_log_likelihood(inputs = logit,
                    tag_indices = tf.to_int32(self.tag_seqs),
                    sequence_lengths = sequence_length)

            #Create weights: 0 weight for padding
            weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

            batch_loss = tf.div(tf.reduce_sum(tf.multiply(-sentence_likelihood, weights)),
                              tf.reduce_sum(weights),
                              name="batch_loss")

            #Add to total loss
            tf.losses.add_loss(batch_loss)

            #Get total loss
            total_loss = tf.losses.get_total_loss()

            tf.summary.scalar('losses/batch_loss', batch_loss)
            tf.summary.scalar('losses/total_loss', total_loss)



    def build():
        """Create all ops for model"""
        self.build_inputs()
        self.build_chr_embedding()
        self.build_lstm_model()
        self.build_sentence_score_loss()
