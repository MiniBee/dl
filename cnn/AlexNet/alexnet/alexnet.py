#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019/10/19 下午1:40
#@Author  :hongyue pei 
#@FileName: alexnet.py
#@Software: PyCharm

import tensorflow as tf
import os
import numpy as np

def max_pool_layer(x, k_height, k_width, x_stride, y_stride, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, k_height, k_width, 1], strides=[1, x_stride, y_stride, 1], padding=padding, name=name)

def dropout(x, keep_prob, name=None):
    return tf.nn.dropout(x, keep_prob=keep_prob, name=name)

def LRN(x, R, alpha, beta, name=None, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=R, alpha=alpha, beta=beta, bias=bias, name=name)

def fc_layer(x, input_dim, output_dim, relu_flag, name):
    with tf.variable_scope(name) as scope:
        w = tf.Variable(tf.truncated_normal([input_dim, output_dim], dtype=tf.float32, stddev=1e-1), name='weights')
        b = tf.Variable(tf.truncated_normal([output_dim], dtype=tf.float32, stddev=1e-1), name='bias')
        out = tf.nn.xw_plus_b(x, w, b)
        if relu_flag:
            return tf.nn.relu(out)
        else:
            return out

def conv_layer(x, k_height, k_width, x_stride, y_stride, feature_num, name, padding='SAME', groups=1):
    channel = int(x.get_shape(x)[-1])
    conv = lambda a, b: tf.layers.conv2d(a, b, strides=[1, x_stride, y_stride, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        w = tf.Variable(tf.truncated_normal([k_height, k_width, channel/groups, feature_num], dtype=tf.float32, stddev=1e-1), name='weights')
        b = tf.Variable(tf.truncated_normal([feature_num], dtype=tf.float32, stddev=1e-1), name='bias')

        x_new = tf.split(value=x, num_or_size_splits=groups, axis=3)
        w_new = tf.split(value=w, num_or_size_splits=groups, axis=3)

        feature_map = [conv(t1, t2) for t1, t2 in zip(x_new, w_new)]
        merge_feature_map = tf.concat(values=feature_map, axis=3)

        out = tf.nn.bias_add(merge_feature_map, b)
        return tf.nn.relu(tf.reshape(out, merge_feature_map.get_shape().as_list()), name=scope.naem)


class AlexNet(object):
    def __init__(self, x, keep_prob, class_num, skip, model_path = './model/'):
        self.X = x
        self.keep_prob = keep_prob
        self.class_num = class_num
        self.skip = skip
        self.model_path = model_path
        self.build_cnn()

    def build_cnn(self):
        conv1 = conv_layer(self.X, 11, 11, 4, 4, 96, 'conv_1', 'VALID')
        lrn1 = LRN(conv1, 2, 2e-05, 0.75, 'norm1')
        pool1 = max_pool_layer(lrn1, 3, 3, 2, 2, 'pool1', 'VALID')

        conv2 = conv_layer(pool1, 5, 5, 1, 1, 256, 'conv2', groups=2)
        lrn2 = LRN(conv2, 2, 2e-05, 0.75, 'norm2')
        pool2 = max_pool_layer(lrn2, 3, 3, 2, 2, 'pool2', 'VALID')

        conv3 = conv_layer(pool2, 3, 3, 1, 1, 384, 'conv3')

        conv4 = conv_layer(conv3, 3, 3, 1, 1, 384, 'conv4', groups=2)

        conv5 = conv_layer(conv4, 3, 3, 1, 1, 256, 'conv5', groups=2)
        pool5 = max_pool_layer(conv5, 3, 3, 2, 2, 'pool5', 'VALID')

        fc_in = tf.reshape(pool5, shape=[-1, 256*6*6])
        fc1 = fc_layer(fc_in, [256*6*6, 4096, True, 'fc6'])
        dropout1 = dropout(fc1, self.keep_prob)

        fc2 = fc_layer(dropout1, 4096, 4096, True, 'fc7')
        dropout2 = dropout(fc2, self.keep_prob)

        self.fc3 = fc_layer(dropout2, 4096, self.class_num, True, 'fc8')

    def save_model(self, sess, saver):
        if not os.path.exists(self.model_path):
            os.mkdir(os.path)
        saver.save(sess, self.model_path)

    def restore_model(self, sess, saver):
        pass

    def load_model(self, sess):
        w_dict = np.load(self.model_path, encoding='bytes').item()



