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
from os.vocab import Vocabulary
import configuration
from lstm_based_cws_model import LSTMCWS


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

def get_final_output(line, predict_tag):
    return ''.join([insert_space(char, tag) for char, tag in zip(line, predict_tag)])

def append_to_file(output_buffer, filename):
    filename = os.path.join('output', 'out' + filename)

    if os.path.exists(filename):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not

    with open(filename, append_write) as file:
        for item in output_buffer:
            file.write("%s\n" % item)




def main(unused_argv)

    #Read vocab file
    with open('data/vocab.pkl', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()

    #Prepare for hash table
    hash_keys = []
    hash_values = []
    for k in p._vocab:
        hash_keys.append(k)
        hash_values.append(p._vocab[k])

    filename_list = ['data\\output_dir\\icwb2-data\\testing\\as_test.utf8']
    checkpoint_path = 'saved_model'

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
                    logit, transition_param_array = sess.run([model.logit, transition_param], 
                        feed_dict = {input_seq_feed:input_seqs_list})
                    predict_tag = tf.contrib.crf.viterbi_decode(logit, transition_param_array)[0]
                    output_buffer.append(get_final_output(line, predict_tag))

                    if len(output_buffer) >= 1000:
                        append_to_file(output_buffer, filename)
                        output_buffer = []

                if output_buffer:
                    append_to_file(output_buffer, filename)


if __name__ == '__main__':
    tf.app.run()
