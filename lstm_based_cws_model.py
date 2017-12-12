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
            mode: 'train', 'test' or 'inference'
        """

        self.config = config
        self.mode = mode

        #Set up initializer
        self.initializer = tf.contrib.layers.xavier_initializer(
            uniform=True, seed=None, dtype=tf.float32)

        #Set up sequence embeddings with the shape of [batch_size, padded_length, embedding_size]
        self.seq_embedding = None

        #Set up batch losses for tracking performance with the length of batch_size * padded_length
        self.batch_losses = None

        #Set up global step tensor
        self.global_step = None

    def is_training(self):
        return self.mode == 'train'

    def is_test(self):
        return self.mode == 'test'

    def is_inf(self):
        return self.mode == 'inference'

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

        if not self.is_inf():

            with tf.variable_scope('train_eval_input'):
                #Get all TFRecord path into a list
                data_files = []
                file_pattern = os.path.join(self.config.input_file_dir, '*.TFRecord')
                data_files.extend(tf.gfile.Glob(file_pattern))


                data_files = [
                    x for x in data_files
                    if os.path.split(x)[-1].startswith(self.mode)
                ]

                if not data_files:
                    tf.logging.fatal("Found no input files matching %s",
                                     file_pattern)
                else:
                    tf.logging.info(
                        "Prefetching values from %d files matching %s",
                        len(data_files), file_pattern)

                def _parse_wrapper(l):
                    return input_ops.parse_example_queue(l, self.config)

                dataset = tf.data.TFRecordDataset(data_files).map(
                    _parse_wrapper)
                if self.is_training():
                    dataset = dataset.shuffle(
                        buffer_size=256).repeat(10000).shuffle(buffer_size=256)

                dataset = dataset.padded_batch(batch_size=self.config.batch_size,
                                            padded_shapes = (tf.TensorShape([self.config.seq_max_len]),
                                                                tf.TensorShape([self.config.seq_max_len]),
                                                                tf.TensorShape([]))).filter(
                                                                    lambda x, y, z: tf.equal(tf.shape(x)[0], self.config.batch_size) )

                iterator = dataset.make_one_shot_iterator()

                input_seqs, tag_seqs, sequence_length = iterator.get_next()

        else:
            with tf.variable_scope('inf_input'):
                #Inference
                input_seq_feed = tf.get_default_graph().get_tensor_by_name(
                    "input_seq_feed:0")
                sequence_length = tf.get_default_graph().get_tensor_by_name(
                    "seq_length:0")
                input_seqs = tf.expand_dims(input_seq_feed, 0)
                sequence_length = tf.expand_dims(sequence_length, 0)

                tag_seqs = None

        self.input_seqs = input_seqs
        self.tag_seqs = tag_seqs
        self.sequence_length = sequence_length

    def build_chr_embedding(self):
        """
        Build Chinese character embedding

        Returns:
            self.seq_embedding: A tensor with the shape of [batch_size, padding_size, embedding_size]
            self.tag_embedding: A tensor with the shape of [batch_size, padding_size, num_tag]
        """
        with tf.variable_scope(
                'seq_embedding', reuse=True) as seq_embedding_scope:
            #chr_embedding = tf.Variable(self.embedding_tensor, name="chr_embedding")
            if self.is_training():
                chr_embedding = tf.get_variable(
                    name="chr_embedding", validate_shape=False, trainable=False)
            else:
                chr_embedding = tf.Variable(
                    tf.zeros([10]), validate_shape=False, name="chr_embedding")

            seq_embedding = tf.nn.embedding_lookup(chr_embedding,
                                                   self.input_seqs)
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
        fw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=self.config.num_lstm_units, state_is_tuple=True)
        bw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=self.config.num_lstm_units, state_is_tuple=True)

        #Dropout when training
        if self.is_training():
            fw_lstm_cell = tf.contrib.rnn.DropoutWrapper(
                fw_lstm_cell,
                input_keep_prob=self.config.lstm_dropout_keep_prob,
                output_keep_prob=self.config.lstm_dropout_keep_prob)
            bw_lstm_cell = tf.contrib.rnn.DropoutWrapper(
                bw_lstm_cell,
                input_keep_prob=self.config.lstm_dropout_keep_prob,
                output_keep_prob=self.config.lstm_dropout_keep_prob)

        self.seq_embedding.set_shape([None, None, self.config.embedding_size])

        with tf.variable_scope('seq_lstm') as lstm_scope:

            #Run LSTM with sequence_length timesteps
            bi_output, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_lstm_cell,
                cell_bw=bw_lstm_cell,
                inputs=self.seq_embedding,
                sequence_length=self.sequence_length,
                dtype=tf.float32,
                scope=lstm_scope)
            fw_out, bw_out = bi_output
            lstm_output = tf.concat([fw_out, bw_out], 2)

        self.lstm_output = lstm_output

    def build_sentence_score_loss(self):
        """
        Use CRF log likelihood to get sentence score and loss
        """
        #Fully connected layer to get logit
        with tf.variable_scope('logit') as logit_scope:
            logit = tf.contrib.layers.fully_connected(
                inputs=self.lstm_output,
                num_outputs=self.config.num_tag,
                activation_fn=None,
                weights_initializer=self.initializer,
                scope=logit_scope)
            self.logit = logit

        if self.is_inf():
            with tf.variable_scope('tag_inf') as tag_scope:
                transition_param = tf.get_variable(
                    'transitions',
                    shape=[self.config.num_tag, self.config.num_tag])

            self.predict_tag, _ = tf.contrib.crf.crf_decode(
                logit, transition_param, self.sequence_length)

        else:
            with tf.variable_scope('tag_inf') as tag_scope:
                sentence_likelihood, transition_param = tf.contrib.crf.crf_log_likelihood(
                    inputs=logit,
                    tag_indices=tf.to_int32(self.tag_seqs),
                    sequence_lengths=self.sequence_length)

            self.predict_tag, _ = tf.contrib.crf.crf_decode(
                logit, transition_param, self.sequence_length)

            with tf.variable_scope('loss'):
                batch_loss = tf.reduce_mean(-sentence_likelihood)

                # if self.is_inf():
                #     prob = tf.nn.softmax(logit)
                #     self.predict_tag = tf.squeeze(tf.nn.top_k(prob, k=1)[1])

                # else:
                #     with tf.variable_scope('loss'):
                #         prob = tf.nn.softmax(logit)
                #         self.predict_tag = tf.squeeze(tf.nn.top_k(prob, k=1)[1])

                #         batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=self.tag_seqs)
                #         batch_loss = tf.reduce_mean(batch_loss)

                #Add to total loss
                tf.losses.add_loss(batch_loss)

                #Get total loss
                total_loss = tf.losses.get_total_loss()

                tf.summary.scalar('batch_loss', batch_loss)
                tf.summary.scalar('total_loss', total_loss)

            with tf.variable_scope('accuracy'):

                seq_len = tf.cast(
                    tf.reduce_sum(self.sequence_length), tf.float32)
                padded_len = tf.cast(
                    tf.reduce_sum(
                        self.config.batch_size * self.config.seq_max_len),
                    tf.float32)

                # Calculate acc
                correct = tf.cast(
                    tf.equal(self.predict_tag, tf.cast(self.tag_seqs,
                                                       tf.int32)), tf.float32)
                correct = tf.reduce_sum(correct) - padded_len + seq_len

                self.accuracy = correct / seq_len

                if self.is_test():

                    tf.summary.scalar('eval_accuracy', self.accuracy)
                else:
                    tf.summary.scalar('average_len',
                                      tf.reduce_mean(self.sequence_length))
                    tf.summary.scalar('train_accuracy', self.accuracy)

            #Output loss
            self.batch_loss = batch_loss
            self.total_loss = total_loss

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[
                tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES
            ])

        self.global_step = global_step

    def build(self):
        """Create all ops for model"""
        self.build_inputs()
        self.build_chr_embedding()
        self.build_lstm_model()
        self.build_sentence_score_loss()
        self.setup_global_step()
