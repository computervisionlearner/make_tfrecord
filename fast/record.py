from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import config
import glob
import random
IMAGE_HEIGHT = config.IMAGE_HEIGHT  #256
IMAGE_WIDTH = config.IMAGE_WIDTH   #256
CLASSES_NUM = config.CLASSES_NUM   #30


RECORD_DIR = config.RECORD_DIR
TRAIN_FILE = config.TRAIN_FILE
VALID_FILE = config.VALID_FILE

FLAGS = None

def label_to_one_hot(label):
  one_hot_label = np.zeros([CLASSES_NUM])
  one_hot_label[label] = 1.0
  return one_hot_label.astype(np.uint8)  #(4,10)

def get_all_paths(input_dir, shuffle=True):
  """Read images from input_dir then shuffle them
  Args:
    input_dir: string, path of input dir, e.g., /path/to/dir
  Returns:
    file_paths: list of strings
  """
  file_paths = glob.glob(os.path.join(input_dir,'*.jpg'))
  if shuffle:
    random.shuffle(file_paths)
  return file_paths


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(label, image):
  """Build an Example proto for an example.
  Args:
    file_path: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
  Returns:
    Example proto
  """

  example = tf.train.Example(features=tf.train.Features(feature={
      'label_raw': _bytes_feature(label),
      'image_raw': _bytes_feature(image)}))
  return example

def conver_to_tfrecords(input_dir, output_file):
  """Write data to tfrecords
  """
  file_paths = get_all_paths(input_dir)
  output_dir = os.path.join(RECORD_DIR, output_file)
  images_num = len(file_paths)
  # dump to tfrecords file
  writer = tf.python_io.TFRecordWriter(output_dir)

  for i in range(len(file_paths)):
    file_path = file_paths[i]
    label = int(os.path.basename(file_path).split('_')[0]) - 1
    label_raw = label_to_one_hot(label).tostring()

    with tf.gfile.FastGFile(file_path, 'rb') as f:
      image_raw = f.read()

    example = _convert_to_example(label_raw, image_raw)
    writer.write(example.SerializeToString())
    if i % 500 == 0:
      print("Processed {}/{}.".format(i, images_num))
  print("Done.")
  writer.close()



def main(_):
#  conver_to_tfrecords(FLAGS.train_dir, TRAIN_FILE)
  conver_to_tfrecords(FLAGS.valid_dir, VALID_FILE)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_dir',
      type=str,
      default='/home/no1/Documents/darknet-pig/frame',
      help='Directory training to get captcha data files and write the converted result.'
  )
  parser.add_argument(
      '--valid_dir',
      type=str,
      default='./pig_face/face_valid',
      help='Directory validation to get captcha data files and write the converted result.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
