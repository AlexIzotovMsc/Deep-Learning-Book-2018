# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 16:18:32 2019

@author: feodorb
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 17:31:27 2019

@author: feodorb
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 64
latent_space = 128
learning_rate = 0.1

rho = 0.05
beta = 1.0
noise_prob = 0.3

ae_weights = {"encoder_w": tf.Variable(tf.truncated_normal([784, latent_space], stddev=0.1)),
              "encoder_b": tf.Variable(tf.truncated_normal([latent_space], stddev=0.1)),
              "decoder_w": tf.Variable(tf.truncated_normal([latent_space, 784], stddev=0.1)),
              "decoder_b": tf.Variable(tf.truncated_normal([784], stddev=0.1))
                           }

ae_input = tf.placeholder(tf.float32, [batch_size, 784])
noisy_input = tf.placeholder(tf.float32, [batch_size, 784])

hidden = tf.nn.sigmoid(tf.matmul(noisy_input, ae_weights["encoder_w"]) + ae_weights["encoder_b"])
visible_logits = tf.matmul(hidden, ae_weights["decoder_w"]) + ae_weights["decoder_b"]
visible = tf.nn.sigmoid(visible_logits)

noised_hidden = tf.nn.relu(hidden - 0.1) + 0.1
noised_visible = tf.nn.sigmoid(tf.matmul(noised_hidden, ae_weights["decoder_w"]) + ae_weights["decoder_b"])

data_rho = tf.reduce_mean(hidden, 0)
reg_cost = -tf.reduce_mean(tf.log(data_rho/rho) * rho + tf.log((1-data_rho)/(1-rho)) * (1-rho))

ae_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = visible_logits, labels = ae_input))

total_cost = ae_cost + beta * reg_cost

optimizer = tf.train.AdagradOptimizer(learning_rate)
ae_op = optimizer.minimize(total_cost)

sess = tf.Session()
sess.run(tf.initializers.global_variables())

for i in range(100000):
    x_batch, _ = mnist.train.next_batch(batch_size)
    noise_mask = np.random.uniform(0., 1., [batch_size, 784]) < noise_prob
    noisy_batch = x_batch.copy()
    noisy_batch[noise_mask] = 0.0
    sess.run(ae_op, feed_dict = {ae_input: x_batch, noisy_input: noisy_batch})
   
x_batch, _ = mnist.train.next_batch(batch_size)
noise_mask = np.random.uniform(0., 1., [batch_size, 784]) < noise_prob
noisy_batch = x_batch.copy()
noisy_batch[noise_mask] = 0.0

final_x, final_noised_x = sess.run([visible, noised_visible], feed_dict = {ae_input: x_batch, noisy_input: noisy_batch})

#plt.imshow(np.reshape(final_x, [-1, 28, 28])[0,:, :], cmap=plt.cm.Greys, extent=(-3, 3, 3, -3))  

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

   