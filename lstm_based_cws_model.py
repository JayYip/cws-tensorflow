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
            self.input_seqs
            self.tag_seqs
        """
        #Get all TFRecord path into a list
        data_files = []
        data_files.extend(tf.gfile.Glob('*.TFRecord'))
        if not data_files:
          tf.logging.fatal("Found no input files matching %s", file_pattern)
        else:
          tf.logging.info("Prefetching values from %d files matching %s",
                          len(data_files), file_pattern)


        #Create file queue
        if self.is_training():
            filename_queue = tf.train.string_input_producer(
                data_files, shuffle = True, capacity = 16, name = filename_input_queue)

        else:
            #TODO
            pass

        #Create example queue
        example_queue = input_ops.example_queue_shuffle(reader, filename_queue, 
            self.is_training(), capacity = 50000, num_reader_threads = self.config.num_preprocess_thread)

        #Parse one example
        input_seq_queue, tag_seq_queue = input_ops.parse_example_queue(example_queue, 
            self.config.context_feature_name, self.config.tag_feature_name)

        #Use shuffle batch to create shuffle queue and get batch examples
        input_seqs, tag_seqs = tf.train.batch(
            [input_seq_queue, tag_seq_queue], 
            num_threads = self.config.num_preprocess_thread,
            batch_size=self.config.batch_size,
            capacity=queue_capacity,
            dynamic_pad=True,
            name="batch_and_pad")
        
        self.input_seqs = input_seqs
        self.tag_seqs = tag_seqs

    def build_chr_embedding(self):
        """
        Build Chinese character embedding

        Returns:
            self.seq_embedding: A tensor with the shape of [batch_size, padding_size, embedding_size]
            self.tag_embedding: A tensor with the shape of [batch_size, padding_size, num_tag]
        """
        with tf.variable_scope('seq_embedding') as seq_embedding_scope:
            chr_embedding = tf.Variable(tf.constant(0.0, shape=[self.config.vocab_size, self.config.embedding_size]),
                trainable=False, name="chr_embedding")

            chr_assign_op = chr_embedding.assign(self.embedding_placeholder)
            seq_embedding = tf.nn.embedding_lookup(chr_embedding, self.input_seqs)

            tag_embedding = tf.one_hot(self.tag_seqs, self.config.num_tag)


        self.seq_embedding = seq_embedding
        self.tag_embedding = tag_embedding

    def build_lstm_model():
        """
        Build model.

        Returns:
            PENDING
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

        with tf.variable_scope('lstm') as lstm_scope:
            #Init lstm
            #Get the initial state for dynamic_rnn
            init_state = lstm_cell.zero_state(batch_size = self.config.batch_size, dtype = tf.float32)

            #Allow variable reuse
            lstm_scope.reuse_variable()

            if self.mode != 'inference':

                #Run LSTM with sequence_length timesteps
                sequence_length = tf.reduce_sum(self.input_seqs, 1)
                lstm_output, _ = tf.nn.dynamic_rnn(cell = lstm_cell,
                    inputs = self.input_seqs,
                    sequence_length = sequence_length,
                    initial_state = init_state,
                    dtype = tf.float32,
                    scope = lstm_scope)

            else:

                #TODO
                #Reference.

        # Stack batches vertically.
        lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

        #Final fully connected layer to create output
        with tf.variable_scope('logit') as logit_scope:
            logit = tf.contrib.fully_connected(input = self.lstm_output,
                num_outputs = self.config.num_tag,
                activation_fn = None,
                weight_initializer = self.initializer,
                score = logit_scope)

        self.logit = logit

    def build_tag_inference():
        """
        Create tag inference for sentence score.

        """




    def build():
        """Create all ops for model"""
        self.build_inputs()
        self.build_chr_embedding()
        self.build_lstm_model()
        self.build_tag_inference()
