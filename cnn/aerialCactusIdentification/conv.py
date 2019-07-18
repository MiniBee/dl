#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: conv.py
# @time: 2019/7/16 下午4:58
# @desc:


import tensorflow as tf

class Model():

    def __init__(self, pic_h, pic_w, y_dim, is_traning=True):
        self.is_training = is_traning
        self.pic_h = pic_h
        self.pic_w = pic_w
        self.y_dim = y_dim
        self.build_model()
        self.init_sess()

    def conv(self, net, n_conv, n_chl, convID, pool_size=None, pool_strides=None, kernel_size=(3, 3), batch_norm=False):
        with tf.variable_scope('blockID%d' % convID):
            for itr in range(n_conv):
                net = tf.layers.conv2d(net, n_chl, kernel_size, activation=tf.nn.relu, padding="same")
            if pool_size and pool_strides:
                net = tf.layers.max_pooling2d(net, pool_size, pool_strides)
            if batch_norm:
                net = tf.layers.batch_normalization(net)
            return net

    def build_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.pic_h, self.pic_w, 3], name='inputs')
            self.labels = tf.placeholder(dtype=tf.float32, shape=[None, self.y_dim], name='labels')
            net = self.conv(self.inputs, 2, 64, 1, 3, 3, (3, 3), batch_norm=True)
            net = self.conv(net, 2, 126, 2, 3, 3, (3, 3), batch_norm=True)
            net = self.conv(net, 2, 256, 3, 3, 3, (3, 3), batch_norm=True)
            net = tf.layers.flatten(net)
            net = tf.layers.batch_normalization(net)
            net = tf.layers.dense(net, 512, activation=tf.nn.relu)
            net = tf.layers.dense(net, 512, activation=tf.nn.relu)
            net = tf.layers.dropout(net, 0.5)
            self.logits = tf.layers.dense(net, self.y_dim, activation=tf.nn.softmax)

            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits)
            self.loss = tf.reduce_mean(self.loss)

            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            self.step = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
            self.all_var = tf.global_variables()
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def init_sess(self, restore=None):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        if restore != None:
            self.saver.restore(self.sess, restore)




