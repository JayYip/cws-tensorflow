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

    def is_training(self):
        return self.mode == 'train'

    def build_inputs(self):
        """
        Input prefetching, preprocessing and batching

        Outputs:
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
        example_queue = example_queue_shuffle(reader, filename_queue, 
            self.is_training(), capacity = 50000, num_reader_threads = 1)

        #Parse simple example
        input_seq_queue, tag_seq_queue = input_ops.parse_example_queue(example_queue, 
            self.config.context_feature_name, self.config.tag_feature_name)

        #Use shuffle batch to create shuffle queue and get batch examples
        input_seqs, tag_seqs, input_mask = input_ops.batch_with_dynamic_pad(input_seq_queue, 
            tag_seq_queue, self.config.batch_size)
        
        self.input_seqs = input_seqs
        self.tag_seqs = tag_seqs
        self.input_mask = input_mask

    def build_chr_embedding(self):
        """
        Build Chinese character embedding

        Output:
            
        """

