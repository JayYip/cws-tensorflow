# -*- coding: utf-8 -*-

#Author: Jay Yip
#Date 04Mar2017
"""Vocab class"""


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, vocab, id_vocab, unk_id, unk_word='<UNK>'):
        """Initializes the vocabulary.

    Args:
      vocab: A dictionary of word to word_id.
      unk_id: Id of the special 'unknown' word.
    """
        self._vocab = vocab
        self._id_vocab = id_vocab
        self._unk_id = unk_id
        self._vocab[unk_word] = len(self._vocab)
        self._id_vocab[len(self._vocab)] = unk_word

    def word_to_id(self, word):
        """Returns the integer id of a word string."""
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._unk_id

    def id_to_word(self, word_id):
        """Returns the word string of an integer word id."""
        if word_id >= len(self._vocab):
            return self._id_vocab[self.unk_id]
        else:
            return self._id_vocab[word_id]
