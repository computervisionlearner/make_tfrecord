import tensorflow as tf
from scipy import misc
from matplotlib import pyplot as plt

import numpy as np


def read_cifar10(filename_queue):

  reader = tf.WholeFileReader()
  key, value = reader.read(filename_queue)
  record_bytes = tf.image.decode_image(value,channels=3)
  return record_bytes
def distorted_inputs():

  filenames = ['apple.jpg','apple1.jpg','image.jpg']
  filename_queue = tf.train.string_input_producer(filenames)
  read_input = read_cifar10(filename_queue) 

  return read_input
image=distorted_inputs()
#images=tf.expand_dims(image)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)
all_image=[]
for i in range(3):
  image_value=sess.run(image)
  all_image.append(image_value)
  
fig, axes = plt.subplots(figsize=(12,12), nrows=1, ncols=3)
for ax, img in zip(axes.flatten(), all_image): 
  ax.imshow(np.squeeze(img))
plt.show()
