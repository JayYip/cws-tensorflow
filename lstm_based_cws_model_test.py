# -*- coding: utf-8 -*-

#Author: Jay Yip
#Date 04Mar2017
"""Chinese words segmentation model Test"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import pickle

import configuration
import lstm_based_cws_model
from ops.vocab import Vocabulary


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


class LSTMCWS(lstm_based_cws_model.LSTMCWS):
    """Build input for test case"""

    def build_input(self):
        if self.mode == "inference":
            self.input_seqs = tf.random_uniform(
                [1, 15],
                minval=0,
                maxval=file_len(os.path.join('data', 'word_count')),
                dtype=tf.int64,
                name='input_seq_feed:0')
            self.input_mask = tf.ones_like(self.input_seqs)
        else:
            self.input_seqs = tf.random_uniform(
                [self.config.batch_size, 15],
                minval=0,
                maxval=file_len(os.path.join('data', 'word_count')),
                dtype=tf.int64)
            self.tag_seqs = tf.random_uniform(
                [self.config.batch_size, 15],
                minval=0,
                maxval=self.config.num_tag,
                dtype=tf.int64)
            self.input_mask = tf.ones_like(self.input_seqs)


class LSTMCWSTest(tf.test.TestCase):

    def setUp(self):
        super(LSTMCWSTest, self).setUp()
        self._model_config = configuration.ModelConfig()

    def _checkOutputs(self, expected_shapes, feed_dict=None):
        """Verifies that the model produces expected outputs.
        
        Args:
          expected_shapes: A dict mapping Tensor or Tensor name to expected output
            shape.
          feed_dict: Values of Tensors to feed into Session.run().
        """
        fetches = list(expected_shapes.keys())

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs = sess.run(fetches, feed_dict)

        for index, output in enumerate(outputs):
            tensor = fetches[index]
            expected = expected_shapes[tensor]
            actual = output.shape
            if expected != actual:
                self.fail("Tensor %s has shape %s (expected %s)." %
                          (tensor, actual, expected))

    def testBuildforTraining(self):
        #Load chr emdedding table
        chr_embedding = pickle.load(open('chr_embedding.pkl', 'rb'))
        shape = chr_embedding.shape

        #Set embedding table
        embedding = tf.convert_to_tensor(chr_embedding, dtype=tf.float32)
        with tf.variable_scope('seq_embedding') as seq_embedding_scope:
            chr_embedding_var = tf.get_variable(
                name='chr_embedding', shape=(shape[0], shape[1]))
            embedding_assign_op = chr_embedding_var.assign(embedding)

        #Load chr emdedding table
        model = LSTMCWS(self._model_config, mode="train")
        model.build()

        expected_shapes = {
            # [batch_size, sequence_length]
            model.input_seqs: (512, 15),
            # [batch_size, sequence_length]
            model.tag_seqs: (512, 15),
            # [batch_size, sequence_length]
            model.input_mask: (512, 15),
            # [batch_size, sequence_length, embedding_size]
            model.seq_embedding: (512, 15, 64),
            # [batch_size, sequence_length, num_tag]
            model.tag_embedding: (512, 15, 4),
            # [batch_size, sequence_length, num_tag]
            model.logit: (512, 15, 4),
            # [batch_size,]
            model.sentence_likelihood: (512,),
            # Scalar
            model.total_loss: ()
        }
        self._checkOutputs(expected_shapes)

    #def testBuildforInf(self):
    #    #Load chr emdedding table
    #    model = LSTMCWS(self._model_config, mode="inference")
    #    model.build()
    #    expected_shapes = {
    #        # [batch_size, sequence_length]
    #        model.input_seqs: (1, 15),
    #        # [batch_size, sequence_length]
    #        model.input_mask: (1, 15),
    #        # [batch_size, sequence_length, embedding_size]
    #        model.seq_embedding: (1, 15, 64),
    #        # [batch_size, sequence_length, num_tag]
    #        model.tag_embedding: (1, 15, 4),
    #        # [batch_size, sequence_length, num_tag]
    #        model.logit: (1, 15, 4)
    #    }
    #    self._checkOutputs(expected_shapes)


if __name__ == "__main__":
    tf.test.main()
