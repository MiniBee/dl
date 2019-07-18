#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019/7/18 下午9:51
#@Author  :hongyue pei 
#@FileName: fc_net.py
#@Software: PyCharm

import tensorflow as tf

class Model():
    def __init__(self, x_dim, learning_rate, is_training=True):
        self.is_training = is_training
        self.x_dim = x_dim
        self.learning_rate = learning_rate
        self.build_model()
        self.init_sess()

    def build_model(self):
        self.graph = tf.Graph()
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.x_dim], name="inputs")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y")
        net = self.block(self.inputs, 2, 64, 1)
        net = self.block(net, 2, 128, 2)
        net = self.block(net, 1, 256, 3)
        net = self.block(net, 1, 64, 4)
        self.y_pred = self.block(net, 1, 1, 5)
        self.loss = tf.losses.mean_squared_error(self.y, self.y_pred)
        self.step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.all_var = tf.global_variables()
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def block(self, net, n_fc, n_chl, blockID):
        '''
        :param net: 输入
        :param n_fc: 层数
        :param n_chl: 输出chl
        :param blockID:
        :return: net
        '''
        with tf.variable_scope("block%d" % blockID):
            for itr in range(n_fc):
                net = tf.layers.dense(net, n_chl, activation=tf.nn.relu)
            # net = tf.layers.max_pooling2d(net, 2, 2)
            net = tf.layers.batch_normalization(net)
            net = tf.layers.dropout(net, 0.1)
        return net

    def init_sess(self, restore=None):
        self.sess = tf.Session()
        self.sess.run(self.init)
        if restore != None:
            self.saver.restore(self.sess, restore)

