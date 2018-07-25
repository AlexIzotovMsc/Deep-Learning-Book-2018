### LINEAR REGRESSION TENSORFLOW

import tensorflow as tf
import keras
import numpy as np

n_samples, batch_size, num_steps = 1000, 100, 20000
X_data = np.random.uniform(0, 1, (n_samples, 1))
y_data = 2 * X_data + 1 + np.random.normal(0, 0.2, (n_samples, 1))
 
# initialize placeholders and dimensionality
X = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))

with tf.variable_scope('linear-regression'):
    k = tf.Variable(tf.random_normal((1, 1), stddev=0.01), name = 'slope')
    b = tf.Variable(tf.zeros(1,), name = 'bias')
    
y_pred = tf.matmul(X, k) + b
loss = tf.reduce_sum(np.power(y - y_pred, 2)) 
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

# running the session
display_step = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_steps):
        # берём случайное подмножество из batch_size  индексов данных
        indicies = np.random.choice(n_samples, batch_size)
        # берём набор данных по выбранным индексам
        X_batch, y_batch = X_data[indicies], y_data[indicies]
        # подаём в функцию sess.run  список переменных, которые нужно подсчитать
        _, loss_val, k_val, b_val = sess.run([optimizer, loss, k, b], 
                                             feed_dict ={X : X_batch, y : y_batch})
        # выводим результат
        if (i+1) % display_step == 0:
            print('epoch %d: %.8f, k=%.4f, b=%.4f' %(i+1, loss_val, k_val, b_val))




