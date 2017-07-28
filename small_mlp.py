"""
Train a small MLP on a simple function.
"""

import tensorflow as tf
import numpy as np
from math import sqrt

def main():
    """
    Create, train, and validate the MLP.
    """
    inputs = tf.placeholder(tf.float32, shape=[3, None])
    targets = tf.placeholder(tf.float32, shape=[1, None])
    outputs = apply_network(inputs)
    loss = tf.reduce_mean(tf.square(outputs - targets))

    opt = tf.train.AdamOptimizer()
    minimize = opt.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_x, train_y = make_data(5000)
        test_x, test_y = make_data(2000)
        train_dict = {inputs: train_x, targets: train_y}
        test_dict = {inputs: test_x, targets: test_y}
        for i in range(0, 5000):
            if i % 100 == 0:
                print('epoch %d: cost=%f val_cost=%f' %
                      (i, sess.run(loss, feed_dict=train_dict),
                       sess.run(loss, feed_dict=test_dict)))
            sess.run(minimize, feed_dict=train_dict)

def apply_network(inputs):
    """
    Apply a small MLP to the input batch.
    """
    return apply_layer(tf.sigmoid(apply_layer(inputs, 64)), 1)

def apply_layer(inputs, out_size):
    """
    Apply a layer to the input batch.
    """
    in_size = inputs.get_shape()[0].value
    weights = tf.Variable(tf.random_normal([out_size, in_size],
                                           stddev=1/sqrt(in_size)))
    biases = tf.Variable(tf.zeros([64, 1]))
    return tf.matmul(weights, inputs) + biases

def make_data(num):
    """
    Make data allocates num samples with input dimension
    3 and output dimension of 1.
    """
    inputs = np.random.normal(size=[3, num])
    targets = np.cbrt(np.square(2.5*inputs[0:1, :]) -
                      inputs[1:2, :] * inputs[2:3, :])
    return inputs, targets

main()
