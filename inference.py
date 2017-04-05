# -*- coding: utf-8 -*-

#Author: Jay Yip
#Date 22Mar2017


"""Inference"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pickle

import os
from ops import input_ops
from ops.vocab import Vocabulary
import configuration
from lstm_based_cws_model import LSTMCWS

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_dir", "data\\output_dir\\icwb2-data\\testing\\",
                       "Path of input files.")
tf.flags.DEFINE_string("vocab_dir", "data/vocab.pkl",
                       "Path of vocabulary file.")
tf.flags.DEFINE_string("train_dir", "save_model",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_string("out_dir", 'output',
                        "Frequency at which loss and global step are logged.")

def _create_restore_fn(checkpoint_path, saver):
    """Creates a function that restores a model from checkpoint.

    Args:
      checkpoint_path: Checkpoint file or a directory containing a checkpoint
        file.
      saver: Saver for restoring variables from the checkpoint file.

    Returns:
      restore_fn: A function such that restore_fn(sess) loads model variables
        from the checkpoint file.
 
    Raises:
      ValueError: If checkpoint_path does not refer to a checkpoint file or a
        directory containing a checkpoint file.
    """
    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        if not checkpoint_path:
            raise ValueError("No checkpoint file found in: %s" % checkpoint_path)

    def _restore_fn(sess):
        tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)
        saver.restore(sess, checkpoint_path)
        tf.logging.info("Successfully loaded checkpoint: %s",
                        os.path.basename(checkpoint_path))

    return _restore_fn

def insert_space(char, tag):
    if tag == 0 or tag == 3:
        return char + ' '
    else:
        return char

def get_final_output(line, predict_tag):
    return ''.join([insert_space(char, tag) for char, tag in zip(line, predict_tag)])

def append_to_file(output_buffer, filename):
    filename = os.path.join(FLAGS.out_dir, 'out_' + os.path.split(filename)[-1])

    if os.path.exists(filename):
        append_write = 'ab' # append if already exists
    else:
        append_write = 'wb' # make a new file if not

    with open(filename, append_write) as file:
        for item in output_buffer:
            file.write(item.encode('utf8'))




def main(unused_argv):

    #Preprocess before building graph
    #Read vocab file
    with open(FLAGS.vocab_dir, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()

    if not tf.gfile.IsDirectory(FLAGS.out_dir):
        tf.logging.info('Create Output dir as %s', FLAGS.out_dir)
        tf.gfile.MakeDirs(FLAGS.out_dir)

    filename_list = []
    for dirpath, dirnames, filenames in os.walk(FLAGS.input_file_dir):
        for filename in filenames:
            fullpath = os.path.join(dirpath, filename)
            if fullpath.split('.')[-1] in ['utf8', 'txt', 'csv']:
                filename_list.append(fullpath)

    #filename_list = ['data\\output_dir\\icwb2-data\\testing\\as_test.utf8']
    checkpoint_path = FLAGS.train_dir

    model_config = configuration.ModelConfig()
    inference_config = configuration.InferenceConfig()



    #Build graph for inference
    g = tf.Graph()
    with g.as_default():

        input_seq_feed = tf.placeholder(name = 'input_seq_feed', dtype = tf.int64)

        #Add transition var to graph
        with tf.variable_scope('tag_inf') as scope:
            transition_param = tf.get_variable(name = 'transition_param', 
                shape = [model_config.num_tag,model_config.num_tag])

        #Build model
        model = LSTMCWS(model_config, 'inference')
        print('Building model...')
        model.build()



    with tf.Session(graph=g) as sess:

        #Restore ckpt
        saver = tf.train.Saver()
        restore_fn = _create_restore_fn(checkpoint_path, saver)
        restore_fn(sess)


        for filename in filename_list:
            output_buffer = []
            with tf.gfile.GFile(filename) as f:
                for line in f:
                    input_seqs_list = [p.word_to_id(x) for x in line]

                    if len(input_seqs_list) == 1:
                        predict_tag = [0]
                        output_buffer.append(get_final_output(line, predict_tag))

                    else:
                        logit, transition_param_p = sess.run([model.logit, transition_param], 
                            feed_dict = {input_seq_feed:input_seqs_list})
                        predict_tag = tf.contrib.crf.viterbi_decode(logit, transition_param_p)[0]
                        output_buffer.append(get_final_output(line, predict_tag))

                    if len(output_buffer) >= 1000:
                        append_to_file(output_buffer, filename)
                        output_buffer = []

                if output_buffer:
                    append_to_file(output_buffer, filename)


if __name__ == '__main__':
    tf.app.run()
