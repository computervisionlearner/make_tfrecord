from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf
import scipy.misc as misc
import config
import numpy as np
RECORD_DIR = config.RECORD_DIR
TRAIN_FILE = config.TRAIN_FILE
VALID_FILE = config.VALID_FILE

IMAGE_WIDTH = config.IMAGE_WIDTH
IMAGE_HEIGHT = config.IMAGE_HEIGHT
CLASSES_NUM = config.CLASSES_NUM

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'label_raw': tf.FixedLenFeature([], tf.string),
          'image_raw': tf.FixedLenFeature([], tf.string),
      })
  image = tf.image.decode_jpeg(features['image_raw'], channels=3)
  resize_image = tf.image.resize_images(image, [IMAGE_WIDTH, IMAGE_HEIGHT])
  resize_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
  resize_image = tf.cast(resize_image, tf.float32) * (1. / 127.5) - 1    #(-1,1)
  label = tf.decode_raw(features['label_raw'], tf.uint8)
  reshape_label = tf.reshape(label, [CLASSES_NUM])#(30,)
  return tf.cast(resize_image, tf.float32), tf.cast(reshape_label, tf.float32)


def inputs(train, batch_size):
  filename = os.path.join(RECORD_DIR,
                          TRAIN_FILE if train else VALID_FILE)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer([filename])
    tf.train.string_input_producer
    image, label = read_and_decode(filename_queue)
    if train:
        images, sparse_labels = tf.train.shuffle_batch([image, label],
                                                       batch_size=batch_size,
                                                       num_threads=6,
                                                       capacity=2000 + 3 * batch_size,
                                                       min_after_dequeue=2000)
    else:
        images, sparse_labels = tf.train.batch([image, label],
                                               batch_size=batch_size,
                                               num_threads=6,
                                               capacity=2000 + 3 * batch_size)

    return images, sparse_labels

def main():
  images, labels = inputs(train = False, batch_size = 13)
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  imgs,labs_sp = sess.run([images,labels])
  labs = np.argmax(labs_sp,1)
  for i in range(len(imgs)):
    name = labs[i]
    misc.imsave('record_test/{}_{}.jpg'.format(name, i),imgs[i])

  coord.request_stop()
  coord.join(threads)
  sess.close()

if __name__ == '__main__':
  main()






