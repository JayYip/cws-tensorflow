# -*- coding: utf-8 -*-

#Author: Jay Yip
#Date 04Mar2017

"""Train the model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

import configuration
from lstm_based_cws_model import LSTMCWS

import pickle


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_dir", "data\output_dir",
                       "Path of TFRecord input files.")
tf.flags.DEFINE_string("train_dir", "save_model",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_integer("number_of_steps", 1000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 10,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_string("tag_inf_ckpt", "tag_inf.ckpt",
                        "If the tag inference model checkpoint cannot be found, tag inference will be trained.")

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):

    sess = tf.InteractiveSession()

    assert FLAGS.input_file_dir, "--input_file_dir is required"
    assert FLAGS.train_dir, "--train_dir is required"

    #Load configuration
    model_config = configuration.ModelConfig()
    train_config = configuration.TrainingConfig()
    model_config.train_dir = FLAGS.train_dir

    #Create train dir
    train_dir = FLAGS.train_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info('Create Training dir as %s', train_dir)
        tf.gfile.MakeDirs(train_dir)

    #Load chr emdedding table
    chr_embedding = pickle.load(open('chr_embedding.pkl', 'wb'))

    #Build graph
    g = tf.Graph()
    with g.as_default():
        #Build model
        model = LSTMCWS(ModelConfig, 'train')

        #Set up learning rate and learning rate decay function

        #Set up training op


        #Set up saver


    #Set up trainer with TFLearn


if __name__ == '__main__':
    tf.app.run()