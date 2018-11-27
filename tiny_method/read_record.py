#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 23:02:02 2018

@author: fangshuai
使用tf.data读取tfrecord文件
"""
import tensorflow as tf
import random
import scipy.misc as misc
import numpy as np
class Reader():
  def __init__(self, tfrecords_file, height=64, width=64, batch_size=128, training = True, name = ''):
    """
    Args:
      tfrecords_file: string, tfrecords file path
      min_queue_examples: integer, minimum number of samples to retain in the queue that provides of batches of examples
      batch_size: integer, number of images per batch
      num_threads: integer, number of preprocess threads
    """
    self.height = height
    self.width =width
    self.batch_size = batch_size
    self.dataset = tf.data.TFRecordDataset([tfrecords_file])
    self.name = name
    self.training = training
  def parse_tfrecord(self, serialized_example):
    """
    Returns:
      images: 4D tensor [batch_size, image_width, image_height, image_depth]
    """
    with tf.name_scope(self.name):
  
      features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string)
            })
  
      image = tf.decode_raw(features['image_raw'], tf.uint8)
      image.set_shape([self.height * self.width *2])
      image = tf.cast(image, tf.float32) 
      label = tf.decode_raw(features['label_raw'], tf.int32)
      label = tf.reshape(tf.cast(label, tf.float32),shape=(1,))
      reshape_image = tf.reshape(image, [self.height, self.width, 2])   
      if self.training:
        reshape_image = tf.image.random_flip_left_right(reshape_image)
        reshape_image = tf.image.random_flip_up_down(reshape_image)
        left,right = tf.split(reshape_image,2,axis=2)
      
        distorted = tf.concat((left,right),axis=1)  #[64,128,1]
        distorted = tf.image.random_flip_left_right(distorted)
      
        result1 = tf.cond(tf.equal(label[0],1),lambda:tf.image.random_brightness(distorted,max_delta=32./255),lambda:distorted)#
        result4 = tf.cond(tf.equal(label[0],1),lambda:tf.image.random_contrast(result1,lower=0.5,upper=1.5),lambda:distorted)
        result4 = result4/tf.reduce_max(result4)
      
        left, right = tf.split(result4, 2, axis = 1)
      else:
        reshape_image = reshape_image/tf.reduce_max(reshape_image)
        left, right = tf.split(reshape_image,2,axis=2)

      return left, right, label
  def feed(self):
    if self.training:
      new_dataset = self.dataset.map(self.parse_tfrecord).repeat().batch(self.batch_size).shuffle(buffer_size = 5000)
    else:
      new_dataset = self.dataset.map(self.parse_tfrecord).repeat().batch(self.batch_size)
    iterator = new_dataset.make_one_shot_iterator()  
    lefts, rights, labels = iterator.get_next()
  
    labels = tf.squeeze(labels)
    return lefts, rights, labels

if __name__ == '__main__':
  train_reader = Reader('/mnt/lustre/mayukun/fangshuai/NIR-Dataset/tfrecords/country.tfrecord',name='test_data')
  lefts_op, rights_op, labels_op = train_reader.feed()
  
  
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)
  lefts, rights, labels = sess.run([lefts_op, rights_op, labels_op])
  imgs = np.concatenate((lefts, rights), axis=1)
  print(np.max(imgs))
  for i in range(len(lefts)):
    misc.imsave('record_test/{}_{}.jpg'.format(i,labels[i]),np.squeeze(imgs[i]))
    
  sess.close()  
