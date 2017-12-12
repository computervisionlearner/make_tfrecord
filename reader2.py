import tensorflow as tf

import glob
from matplotlib import pyplot as plt

image_dir = '/home/no1/Documents/pig_v4/test_B/*.JPG'

def read(filename_queue):
  reader = tf.WholeFileReader()
  key, value = reader.read(filename_queue)
  record_bytes = tf.image.decode_image(value,channels=3)
  return record_bytes

def distorted_inputs(image_dir):
  filenames = glob.glob(image_dir)
  filename_queue = tf.train.string_input_producer(filenames,shuffle=False)
  read_input = read(filename_queue)
  reshaped_image = tf.cast(read_input, tf.float32)
  height = 250
  width = 250
  distorted_image = tf.random_crop(reshaped_image, [height, width,3])
  #   add image preprocess

  return distorted_image

image=distorted_inputs(image_dir)
a_batch = tf.train.shuffle_batch([image],
                        batch_size=16, capacity=200, min_after_dequeue=100, num_threads=6)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)
image_value=sess.run(a_batch)
fig, axes = plt.subplots(figsize=(12,12), nrows=4, ncols=4)

for ax, img in zip(axes.flatten(), image_value):
  ax.imshow(img)

plt.show()
