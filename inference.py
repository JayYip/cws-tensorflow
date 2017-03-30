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
import configuration
from lstm_based_cws_model import LSTMCWS



class Vocabulary(object):
  """Simple vocabulary wrapper."""

  def __init__(self, vocab, unk_id, unk_word = '<UNK>'):
    """Initializes the vocabulary.

    Args:
      vocab: A dictionary of word to word_id.
      unk_id: Id of the special 'unknown' word.
    """
    self._vocab = vocab
    self._unk_id = unk_id
    self._vocab[unk_word] = 0

  def word_to_id(self, word):
    """Returns the integer id of a word string."""
    if word in self._vocab:
      return self._vocab[word]
    else:
      return self._unk_id

  def id_to_word(self, word_id):
    """Returns the word string of an integer word id."""
    if word_id >= len(self._vocab):
      return self._vocab[self.unk_id]
    else:
      return self._vocab[word_id]

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

    chr_embedding = pickle.load(open('chr_embedding.pkl', 'rb'))

    #Build graph for inference
    g = tf.Graph()
    with g.as_default():

        input_seq_feed = tf.placeholder(name = 'input_seq_feed', dtype = tf.int64)
        embedding = tf.convert_to_tensor(chr_embedding, dtype = tf.float32)

        #Add transition var to graph
        with tf.variable_scope('tag_inf') as scope:
            transition_param = tf.get_variable(name = 'transition_param', 
                shape = [model_config.num_tag,model_config.num_tag])

        #Build model
        model = LSTMCWS(model_config, 'inference')
        model.embedding_tensor = embedding
        print('Building model...')
        model.build()



    with tf.Session(graph=g) as sess:

        #Restore ckpt
        saver = tf.train.Saver()
        restore_fn = _create_restore_fn(checkpoint_path, saver)
        restore_fn(sess)


        for filename in filename_list:

            with tf.gfile.GFile(filename) as f:
                for line in f:
                    input_seqs_list = [p.word_to_id(x) for x in line]
                    logit, transition_param = sess.run([model.logit, transition_param], 
                        feed_dict = {input_seq_feed:input_seqs_list})
                    print(tf.contrib.crf.viterbi_decode(logit, transition_param)[0])
                    break


            #Make hash table to convert words to id
            ts_hash_keys = tf.convert_to_tensor(hash_keys)
            ts_hash_values = tf.convert_to_tensor(hash_values)
            vocab = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(ts_hash_keys, ts_hash_values), -1)

            raw_text = tf.expand_dims(raw_text, 0)

            split_text = tf.string_split(raw_text, delimiter='').values

            #Convert to id
            input_seq_queue = vocab.lookup(split_text)

            seq_length = tf.expand_dims(tf.subtract(tf.shape(tag_seq_queue)[0], 1),0)
            indicator = tf.ones(seq_length, dtype=tf.int32)

            input_seqs, input_mask = tf.train.batch(
                [input_seq_queue, indicator], 
                batch_size=inference_config.batch_size,
                capacity=50000,
                dynamic_pad=True,
                name="batch_and_pad")

            model.input_seqs = input_seqs
            model.input_mask = input_mask




if __name__ == '__main__':
    tf.app.run()
