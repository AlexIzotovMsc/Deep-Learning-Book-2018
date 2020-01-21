# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 17:32:15 2019

@author: feodorb
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
n_samples = mnist.train.num_examples

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

w, n_input, n_z = {}, 784, 20

#erase me
n_z = 2

n_hidden_recog_1, n_hidden_recog_2 = 500, 500
n_hidden_gener_1, n_hidden_gener_2 = 500, 500

w['w_recog'] = {
        'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
        'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
        'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
        'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z)) }

w['b_recog'] = {
        'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
        'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
        'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
        'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32)) }

w['w_gener'] = {
        'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
        'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
        'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
        'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input)) }

w['b_gener'] = {
        'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
        'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
        'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
        'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32)) }
        
l_rate=0.001
batch_size=100
x=tf.placeholder(tf.float32,[None, n_input])

enc_layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, w["w_recog"]['h1']), w["b_recog"]['b1']))
enc_layer_2 = tf.nn.softplus(tf.add(tf.matmul(enc_layer_1, w["w_recog"]['h2']), w["b_recog"]['b2']))
z_mean = tf.add(tf.matmul(enc_layer_2, w["w_recog"]['out_mean']), w["b_recog"]['out_mean'])
z_log_sigma_sq = tf.add(tf.matmul(enc_layer_2, w["w_recog"]['out_log_sigma']), w["b_recog"]['out_log_sigma'])

eps = tf.random_normal((batch_size, n_z), 0, 1, dtype=tf.float32)
z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))

z_ = tf.placeholder(tf.float32,[None, n_z])

def def_x_reconstr_mean(z_param):
    dec_layer_1 = tf.nn.softplus(tf.add(tf.matmul(z_param, w["w_gener"]['h1']), w["b_gener"]['b1']))
    dec_layer_2 = tf.nn.softplus(tf.add(tf.matmul(dec_layer_1, w["w_gener"]['h2']), w["b_gener"]['b2']))

    return tf.nn.sigmoid(tf.add(tf.matmul(dec_layer_2, w["w_gener"]['out_mean']), w["b_gener"]['out_mean']))

x_reconstr_mean = def_x_reconstr_mean(z)
x_reconstr_mean_with_z_param = def_x_reconstr_mean(z_)
    
reconstr_loss = -tf.reduce_sum(x * tf.log(1e-10 + x_reconstr_mean) + (1-x) * tf.log(1e-10 + 1 - x_reconstr_mean), 1)

latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)

cost = tf.reduce_mean(reconstr_loss + latent_loss)

optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(cost)

def train(sess, batch_size=100, training_epochs=10, display_step=5):
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        
        for i in range(total_batch):
            xs, _ = mnist.train.next_batch(batch_size)
            
            _,c = sess.run((optimizer, cost), feed_dict={x: xs})
            avg_cost += c / n_samples * batch_size
        if epoch % display_step == 0:
            print("Epoch: %04d\tcost: %.9f" % (epoch +1, avg_cost))
            
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

train(sess, training_epochs=10, batch_size=batch_size)

nx = ny = 20
x_values = np.linspace(-3, 3, nx)
y_values = np.linspace(-3, 3, ny)

canvas = np.empty((28*nx, 28*ny))

for i, yi in enumerate(x_values):
    for j, xi in enumerate(y_values):
        z_mu= np.array([[xi, yi]])
        x_mean = sess.run(x_reconstr_mean_with_z_param, feed_dict={z_: z_mu})
        canvas[(nx-i-1)*28 : (nx-i) * 28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)
        
plt.figure(figsize=(8,10))
Xi, Yi = np.meshgrid(x_values, y_values)
plt.imshow(canvas, origin="upper", cmap=plt.cm.Greys)
plt.tight_layout()

#x_sample = mnist.test.next_batch(100)[0]
#x_logits = sess.run(x_reconstr_mean_logits, feed_dict={x: x_sample, eps: np.random.normal(loc = 0., scale=1., size=(batch_size, n_z))})

#gen_logits = sess.run(x_reconstr_mean_logits, feed_dict={z_mean: np.random.normal(loc = 0., scale=1., size=(batch_size, n_z))})


