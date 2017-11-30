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
from ops.vocab import Vocabulary

import pickle


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_dir", "data\output_dir",
                       "Path of TFRecord input files.")
tf.flags.DEFINE_string("train_dir", "save_model",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_integer("log_every_n_steps", 100,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_string("log_dir", "log", "Path of summary")

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):


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
    if train_config.embedding_random:
        shape = [len(pickle.load(open('data/vocab.pkl', 'rb'))._vocab), model_config.embedding_size]
    else:
        chr_embedding = pickle.load(open('chr_embedding.pkl', 'rb'))
        shape = chr_embedding.shape

    #Build graph
    g = tf.Graph()
    with g.as_default():
        #Set embedding table
        with tf.variable_scope('seq_embedding') as seq_embedding_scope:
            chr_embedding_var = tf.get_variable(name = 'chr_embedding', 
                shape = (shape[0], shape[1]), trainable=True, initializer=tf.initializers.orthogonal(-0.1, 0.1))
            if not train_config.embedding_random:
                embedding = tf.convert_to_tensor(chr_embedding, dtype = tf.float32)
                embedding_assign_op = chr_embedding_var.assign(chr_embedding)

        #Build model
        model = LSTMCWS(model_config, 'train')
        print('Building model...')
        model.build()

        # merged = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter(FLAGS.logdir + '/train',
        #                                     g)

        #Set up learning rate and learning rate decay function
        learning_rate_decay_fn = None
        learning_rate = tf.constant(train_config.initial_learning_rate)
        if train_config.learning_rate_decay_factor > 0:
            num_batches_per_epoch = (train_config.num_examples_per_epoch /
                                     model_config.batch_size)
            decay_steps = int(num_batches_per_epoch *
                              train_config.num_epochs_per_decay)

            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps=decay_steps,
                    decay_rate=train_config.learning_rate_decay_factor,
                    staircase=True)

            learning_rate_decay_fn = _learning_rate_decay_fn

        print('Setting up training ops...')
        #Set up training op
        train_op = tf.contrib.layers.optimize_loss(
            loss = model.batch_loss,
            global_step = model.global_step,
            learning_rate = learning_rate,
            optimizer = train_config.optimizer,
            clip_gradients = train_config.clip_gradients,
            learning_rate_decay_fn = learning_rate_decay_fn,
            name = 'train_op')

        #Set up saver
        saver = tf.train.Saver(max_to_keep = train_config.max_checkpoints_to_keep)

    print('Start Training...')
    # Run training.
    tf.contrib.slim.learning.train(
        train_op,
        train_dir,
        log_every_n_steps=FLAGS.log_every_n_steps,
        graph=g,
        global_step=model.global_step,
        number_of_steps=train_config.training_step,
        saver=saver,
        save_summaries_secs=5)


if __name__ == '__main__':
    tf.app.run()