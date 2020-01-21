# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 16:49:35 2019

@author: feodorb
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 64
learning_rate = 0.1
noise_prob = 0.3

ae_weights = {"conv1": tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1)),
              "b_conv1": tf.Variable(tf.truncated_normal([4], stddev=0.1)),
              "conv2": tf.Variable(tf.truncated_normal([5, 5, 4, 16], stddev=0.1)),
              "b_hidden": tf.Variable(tf.truncated_normal([16], stddev=0.1)),
              "deconv1": tf.Variable(tf.truncated_normal([5, 5, 4, 16], stddev=0.1)),
              "b_deconv1": tf.Variable(tf.truncated_normal([4], stddev=0.1)),
              "deconv2": tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1)),
              "b_visible": tf.Variable(tf.truncated_normal([1], stddev=0.1))
                           }
h1_shape = [batch_size, 14, 14, 4]
input_shape = [batch_size, 28, 28, 1]

ae_input = tf.placeholder(tf.float32, [batch_size, 784])
noisy_input = tf.placeholder(tf.float32, [batch_size, 784])

images = tf.reshape(ae_input, [-1, 28, 28, 1])
conv_h1_logits = tf.nn.conv2d(images, ae_weights["conv1"], strides=[1, 2, 2, 1], padding="SAME") + ae_weights["b_conv1"]
conv1 = tf.nn.relu(conv_h1_logits)

hidden_logits = tf.nn.conv2d(conv_h1_logits, ae_weights["conv2"], strides=[1, 2, 2, 1], padding="SAME") + ae_weights["b_hidden"]
hidden = tf.nn.relu(hidden_logits)

deconv_h1_logits = tf.nn.conv2d_transpose(hidden, ae_weights["deconv1"], h1_shape, strides=[1, 2, 2, 1], padding="SAME") + ae_weights["b_deconv1"]
deconv_h1 = tf.nn.relu(deconv_h1_logits)

visible_logits = tf.nn.conv2d_transpose(deconv_h1, ae_weights["deconv2"], input_shape, strides=[1, 2, 2, 1], padding="SAME") + ae_weights["b_visible"]
visible = tf.nn.sigmoid(visible_logits)

noised_hidden = tf.nn.relu(hidden - 0.1) + 0.1
noised_deconv_h1_logits = tf.nn.conv2d_transpose(noised_hidden, ae_weights["deconv1"], h1_shape, strides=[1, 2, 2, 1], padding="SAME") + ae_weights["b_deconv1"]
noised_deconv_h1 = tf.nn.relu(noised_deconv_h1_logits)

noised_visible_logits = tf.nn.conv2d_transpose(noised_deconv_h1, ae_weights["deconv2"], input_shape, strides=[1, 2, 2, 1], padding="SAME") + ae_weights["b_visible"]
noised_visible = tf.nn.sigmoid(noised_visible_logits)

optimizer = tf.train.AdagradOptimizer(learning_rate)
conv_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = visible_logits, labels = images))

conv_op = optimizer.minimize(conv_cost)

sess = tf.Session()
sess.run(tf.initializers.global_variables())

for i in range(1000):
    x_batch, _ = mnist.train.next_batch(batch_size)
    noise_mask = np.random.uniform(0., 1., [batch_size, 784]) < noise_prob
    noisy_batch = x_batch.copy()
    noisy_batch[noise_mask] = 0.0
    sess.run(conv_op, feed_dict = {ae_input: x_batch, noisy_input: noisy_batch})

x_batch, _ = mnist.train.next_batch(batch_size)
noise_mask = np.random.uniform(0., 1., [batch_size, 784]) < noise_prob
noisy_batch = x_batch.copy()
noisy_batch[noise_mask] = 0.0
final_x, final_noised_x = sess.run([visible, noised_visible], feed_dict = {ae_input: x_batch, noisy_input: noisy_batch})

plt.figure(num=None, figsize=(16, 24), dpi=80, facecolor='w', edgecolor='k')
for i in range(10):
    plt.subplot(10, 4, 4 * i + 1)
    im = plt.imshow(np.reshape(final_x, [-1, 28, 28])[i,:, :], cmap=plt.cm.Greys, extent=(-3, 3, 3, -3))  
    plt.subplot(10, 4, 4 * i + 2)
    im = plt.imshow(np.reshape(final_noised_x, [-1, 28, 28])[i,:, :], cmap=plt.cm.Greys, extent=(-3, 3, 3, -3))  
    plt.subplot(10, 4, 4 * i + 3)
    im = plt.imshow(np.reshape(x_batch, [-1, 28, 28])[i,:, :], cmap=plt.cm.Greys, extent=(-3, 3, 3, -3))  
    plt.subplot(10, 4, 4 * i + 4)
    im = plt.imshow(np.reshape(noisy_batch, [-1, 28, 28])[i,:, :], cmap=plt.cm.Greys, extent=(-3, 3, 3, -3))  


