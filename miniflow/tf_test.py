#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019/12/1 下午4:29
#@Author  :hongyue pei 
#@FileName: tf_test.py
#@Software: PyCharm

import  tensorflow as tf
import numpy as np

x_input = np.array([[1,1,1,0,0], [0,1,1,1,0], [0,0,1,1,1], [0,0,1,1,0],[0,1,1,0,0]], dtype=np.float32)
x_kernel_1 = np.array([[1,0,1], [0,1,0], [1,0,1]], dtype=np.float32)
#
tf_x_input = tf.constant(np.reshape(x_input, newshape=[1, 5, 5, 1]), dtype=tf.float32)
tf_x_kernel_1 = tf.constant(np.reshape(x_kernel_1, newshape=[3, 3, 1, 1]), dtype=tf.float32)
#
y1 = tf.nn.conv2d(tf_x_input, tf_x_kernel_1, strides=[1,1], padding='VALID')
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     [y1_cov] = sess.run([y1])
#
# print(y1_cov[0, :, :, 0])


x_kernel_3 = np.array([[1,1,1], [1,1,1], [1,1,1]], dtype=np.float32)
tf_x_kernel_3 = tf.constant(np.reshape(x_kernel_3, newshape=[3,3,1,1]))
y1_trans = tf.nn.conv2d_transpose(y1, tf_x_kernel_3, output_shape=[1,5,5,1], strides=[1,2,2,1], padding='SAME')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    [y1_conv_trans] = sess.run([y1_trans])
    [y1_cov] = sess.run([y1])


print(y1_cov[0, :, :, 0])
print(y1_conv_trans[0, :, :, 0])


