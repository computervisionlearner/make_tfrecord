#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:13:21 2017

@author: no1
如果标签全是数字，则可以直接将字符类型转化成int型，如果不全是数字，本版本任可使用
"""

import tensorflow as tf
import numpy as np
import glob
import os
from matplotlib import pyplot as plt

img_dir = '/home/no1/Documents/pig_v4/test_B/*.JPG'

def read_picture_and_label(filename_queue):
  file_contents=tf.read_file(filename_queue[0])
  img = tf.image.decode_image(file_contents,channels=3)
  label = filename_queue[1]
  return img, label

def distorted_inputs(img_dir):
  paths = glob.glob(img_dir)
  filenames = [os.path.basename(path) for path in paths]
  filename_queue = tf.train.slice_input_producer([paths,filenames], shuffle=False)
  img, label = read_picture_and_label(filename_queue)
  img = tf.image.convert_image_dtype(img, tf.float32)
  croped_image = tf.random_crop(img, [300, 300,3])

  return croped_image, label

image, label =distorted_inputs(img_dir)
batch_images, batch_labels = tf.train.batch([image, label],
                        batch_size=16, capacity=50, num_threads=2)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    photos, label_bytes = sess.run([batch_images, batch_labels])
    labels = [ bytes.decode(label) for label in label_bytes]
    coord.request_stop()
    coord.join(threads)

fig, axes = plt.subplots(figsize=(12,12), nrows=4, ncols=4)
for ax, img, img_name in zip(axes.flatten(), photos, labels):
  ax.imshow(img)
  ax.set_title(img_name)
  ax.axis('off')

plt.show()



