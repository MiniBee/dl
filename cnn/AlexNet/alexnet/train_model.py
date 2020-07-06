#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019/10/20 上午11:12
#@Author  :hongyue pei 
#@FileName: train_model.py
#@Software: PyCharm

import tensorflow as tf
import alexnet
import image_data
import os

def train(model_path, epochs, reuse_model=False):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if reuse_model and os.path.exists(model_path):
            pass
        else:
            sess.run(init)

        best_loss = float('inf')
        cur_epoch = 0
        while cur_epoch < epochs:

            cur_epoch += 1





