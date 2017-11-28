# -*- coding: utf-8 -*-

#Author: Jay Yip
#Date 20Feb2017

"""
Download the PKU-MSR datasets or Chinese Wiki dataset and convert
it to TFRecords.

PKU-MSR Download Address: http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip
Chinese Wiki Address: PENDING

Each file is a TFRecord

Output:
download_dir/train-00000-of-00xxx
...
download_dir/train-00127-of-00xxx

Processing Description:


Files Description:


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import urllib.request
import zipfile
import zlib
import os
from collections import Counter
import numpy as np
import threading
from datetime import datetime
import sys
import pickle
from hanziconv.hanziconv import HanziConv
from multiprocessing import Process

import tensorflow as tf


tf.flags.DEFINE_string("data_source", "pku-msr",
                       "Specify the data source: pku-msr or wiki-chn")
tf.flags.DEFINE_string("download_dir", "download_dir", "Output data directory.")
tf.flags.DEFINE_string("word_counts_output_file", "word_count", "Word Count output dir")
tf.flags.DEFINE_integer("train_shards", 128,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("num_threads", 4,
                        "Number of threads to preprocess the images.")
tf.flags.DEFINE_integer("window_size", 5,
                        "The window size of skip-gram model")
FLAGS = tf.flags.FLAGS


class Vocabulary(object):
  """Simple vocabulary wrapper."""

  def __init__(self, vocab, id_vocab, unk_id, unk_word = '<UNK>'):
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

def tag_to_id(t):
    if t == 's':
        return 0

    elif t == 'b':
        return 1

    elif t == 'm':
        return 2

    elif t == 'e':
        return 3

#Line processing functions

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

def process_line_msr_pku(l):
    decoded_line = l.decode('utf8').strip().split('  ')
    return [w.strip('\r\n') for w in decoded_line]

def process_line_as_training(l):
    decoded_line = HanziConv.toSimplified(l.decode('utf8')).strip().split('\u3000')
    return [w.strip('\r\n') for w in decoded_line]

def process_line_cityu(l):
    decoded_line = HanziConv.toSimplified(l.decode('utf8')).strip().split(' ')
    return [w.strip('\r\n') for w in decoded_line]

def get_process_fn(filename):

    if 'msr' in filename or 'pk' in filename:
        return process_line_msr_pku

    elif 'as' in filename:
        return process_line_as_training

    elif 'cityu' in filename:
        return process_line_cityu

def _is_valid_data_source(data_source):
    return data_source in ['pku-msr', 'wiki-chn']


# Convert feature functions
def _int64_feature(value):
  """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf8')]))

def _int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def download_extract(data_source, download = 'Y'):
    """
    Download files from web and extract
    """
    if data_source == 'pku-msr':

        if download == 'Y':
            file_name = 'icwb2-data.zip'
            urllib.request.urlretrieve('http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip', 
                os.path.join(FLAGS.download_dir, file_name))
            
            zip_ref = zipfile.ZipFile(os.path.join(FLAGS.download_dir, file_name), 'r')
            zip_ref.extractall(FLAGS.download_dir)
            zip_ref.close()


    elif data_source == 'wiki-chn':

        #Implement in the future...
        #If there's future...
        pass

    else:
        assert _is_valid_num_shards(FLAGS.data_source), (
        "Please make sure the data source is either 'pku-msr' or 'wiki-chn'")



def _create_vocab(path_list):
    """
    Create vocab objects
    """

    counter = Counter()
    row_count = 0

    for file_path in path_list:
        print("Processing"+file_path)
        with open(file_path, 'rb') as f:
            for l in f:
                counter.update(HanziConv.toSimplified(l.decode('utf8')))
                row_count = row_count + 1

    print("Total char:", len(counter))

    # Filter uncommon words and sort by descending count.
    word_counts = [x for x in counter.items() if x is not ' ']
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Words in vocabulary:", len(word_counts))

    # Write out the word counts file.
    with open(FLAGS.word_counts_output_file, "wb") as f:

      #line = str("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
      line = ["%s %d" % (w, c) for w, c in word_counts]
      line = "\n".join(w for w in line).encode('utf8')

      f.write(line)
    print("Wrote vocabulary file:", FLAGS.word_counts_output_file)

    # Create the vocabulary dictionary.
    reverse_vocab = [x[0] for x in word_counts]
    unk_id = len(reverse_vocab)
    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
    id_vocab_dict = dict([(y, x) for (y, x) in enumerate(reverse_vocab)])
    vocab = Vocabulary(vocab_dict, id_vocab_dict, unk_id)

    return vocab



def _to_sequence_example(decoded_str, pos_tag_str, vocab):

    #Transfor word to word_id
    content_id = [vocab.word_to_id(c) for c in decoded_str]
    tag_id = [tag_to_id(t) for t in pos_tag_str]


    feature_lists = tf.train.FeatureLists(feature_list={
        "text/content_id": _int64_feature_list(content_id),
        #"text/left": _bytes_feature_list(left),
        "text/tag_id": _int64_feature_list(tag_id),
        #"text/right": _bytes_feature_list(right)
        })

    sequence_example = tf.train.SequenceExample(feature_lists=feature_lists)

    return sequence_example




def _process_text_files(thread_index, name, path_list, vocab, num_shards):


    #Create possible tags for fast lookup
    possible_tags = []
    for i in range(1, 30):
        if i == 1:
            possible_tags.append('s')
        else:
            possible_tags.append('b' + 'm' * (i - 2) + 'e')


    for s in range(len(path_list)):
        filename = path_list[s]
        #Create file names for shards
        output_filename = "%s-%s" % (name, filename.split('\\')[-1].split('.')[0])
        output_file = os.path.join(output_filename + '.TFRecord')

        #Init writer
        writer = tf.python_io.TFRecordWriter(output_file)

        #Get the input file name
        

        counter = 0


        #Init left and right queue
        
        sequence_example = None
        with open(filename, 'rb') as f:

            process_fn = get_process_fn(os.path.split(filename)[-1])

            for l in f:
                pos_tag = []
                final_line = []

                decoded_line = process_fn(l)

                for w in decoded_line:
                    if w and len(w) <= 29:
                        final_line.append(w)
                        pos_tag.append(possible_tags[len(w)-1])

                decode_str = ''.join(final_line)

                pos_tag_str = ''.join(pos_tag)

                if len(pos_tag_str) != len(decode_str):
                    continue
                    print('Skip one row. ' + pos_tag_str + ';' + decode_str)

                if len(decode_str) > 0: 
                    sequence_example = _to_sequence_example(decode_str, pos_tag_str, vocab)
                    writer.write(sequence_example.SerializeToString())
                    counter += 1

                if not counter % 5000:
                    print("%s [thread %d]: Processed %d in thread batch." %
                          (datetime.now(), thread_index, counter))
                    sys.stdout.flush()



        writer.close()
        print("%s [thread %d]: Finished writing to %s" %
              (datetime.now(), thread_index, output_file))
        sys.stdout.flush()
        counter = 0





def _process_dataset(name, path_list, vocab):
    """
    """

    #Set number of threads
    num_threads = FLAGS.num_threads
    num_shards = len(path_list)



    #Decide 
    spacing = np.linspace(0, len(path_list), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    #Assign path_list based on thread to avoid error
    path_list_list = split_list(path_list, wanted_parts = num_threads)
    print(path_list_list)

    #Launch thread for batch processing
    print("Launching %d threads" % (num_threads))
    for thread_index in range(num_threads):
        args = (thread_index, name, path_list_list[thread_index], vocab, num_shards)
        t = Process(target=_process_text_files, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d text files in data set '%s'." %
        (datetime.now(), len(path_list), name))


def get_path(data_dir = '.', suffix = 'utf8', mode = 'train'):

    path_list = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for filename in filenames:
            fullpath = os.path.join(dirpath, filename)
            if fullpath.endswith(suffix) and mode in fullpath:
                path_list.append(fullpath)

    return path_list
    


def main(unused_argv):

    try:
        os.makedirs(FLAGS.download_dir)
    except (OSError, IOError) as err:
        # Windows may complain if the folders already exist
        pass

    download_extract(FLAGS.data_source, 'N')

    path_list = get_path(data_dir=os.path.join(FLAGS.download_dir, 'icwb2-data', 'training'))

    trimmed_path_list = []
    for filename in path_list:
        output_filename = "%s-%s" % ('train', filename.split('\\')[-1].split('.')[0])
        output_file = os.path.join(output_filename + '.TFRecord')
        if os.path.isfile(output_file):
            pass
        else:
            trimmed_path_list.append(filename)

    path_list = trimmed_path_list

    _process_dataset('train', path_list, vocab)

    trimmed_path_list = []
    for filename in path_list:
        output_filename = "%s-%s" % ('test', filename.split('\\')[-1].split('.')[0])
        output_file = os.path.join(output_filename + '.TFRecord')
        if os.path.isfile(output_file):
            pass
        else:
            trimmed_path_list.append(filename)

    path_list = trimmed_path_list

    _process_dataset('test', path_list, vocab)

if __name__ == '__main__':
    tf.app.run()