# -*- coding: utf-8 -*-

#Author: Jay Yip
#Date 04Mar2017

"""Set the configuration of model and training parameters"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ModelConfig(object):
    """docstring for ModelConfig"""
    def __init__(self):
        
        #Set the feature name of context and tags
        self.context_feature_name = 'text/content_id'
        self.tag_feature_name = 'text/tag_id'

        #Number of thread for prefetching SequenceExample
        #self.num_input_reader_thread = 2
        #Number of preprocessing threads
        self.num_preprocess_thread = 2

        #Batch size
        self.batch_size = 512

        #LSTM input and output dimensions
        self.embedding_size = 64
        self.num_lstm_units = 64

        #Fully connected layer output dimensions
        self.num_tag = 4

        #Dropout
        self.lstm_dropout_keep_prob = 0.7
        #Margin loss discount
        self.margin_loss_discount = 0.2
        #Regularization
        self.regularization = 0.0001
        

class TrainingConfig(object):
    """docstring for TrainingConfig"""
    def __init__(self):
        
        self.num_examples_per_epoch = 5120

        #Optimizer for training
        self.optimizer = 'Adam'

        #Learning rate
        self.initial_learning_rate = 2.0
        #If decay factor <= 0 then not decay
        self.learning_rate_decay_factor = 0.5
        self.num_epochs_per_decay = 5.0

        #Gradient clipping
        self.clip_gradients = 5.0

        #Max checkpoints to keep
        self.max_checkpoints_to_keep = 5

        #Set training step
        self.training_step = 100000


