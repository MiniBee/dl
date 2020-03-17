#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/3/16 下午10:41
#@Author  :hongyue pei 
#@FileName: train.py
#@Software: PyCharm

import transformer
import os
import tensorflow as tf
import time
import numpy as np

D_MODEL = 1024
D_POINT_WISE_FF = 2048
ENCODER_COUNT = 1
EPOCHES = 40
ATTENTION_HEAD_COUNT = 1
DROPOUT_PROB = 0.1
BATCH_SIZE = 10
BPE_VOCAB_SIZE = 7494


def load_data(path):
    x_train = []
    y_train = []
    with open(path) as f:
        for line in f:
            i = line.split(',')
            i = [float(a.strip()) for a in i]
            if len(i) < 5000:
                i += [0.0] * (5000 - len(i) + 1)
            x_train.append(i[1:])
            y_train.append(i[0])
    return np.array(x_train), np.array(y_train)


if __name__ == '__main__':
    x_train, y_train = load_data(os.path.join(os.getcwd() + '/../../data/bc/trainVec'))
    x_test, y_test = load_data(os.path.join(os.getcwd() + '/../../data/bc/testVec'))
    # input_vocat_size, encoder_count, attention_head_count, d_model, d_point_wise_ff, dropout_prob
    # train_loss = tf.keras.metrics.BinaryCrossentropy('train_loss', dtype=tf.float32)
    # train_accuracy = tf.keras.metrics.BinaryAccuracy('train_accuracy')
    #
    # validation_loss = tf.keras.metrics.BinaryCrossentropy('validation_loss', dtype=tf.float32)
    # validation_accuracy = tf.keras.metrics.BinaryAccuracy('validation_accuracy')
    #
    # cur_path = os.getcwd()
    # for epoch in range(EPOCHES):
    #     pass
    model = tf.keras.Sequential([
        transformer.Transformer(BPE_VOCAB_SIZE, ENCODER_COUNT, ATTENTION_HEAD_COUNT, D_MODEL, D_POINT_WISE_FF, DROPOUT_PROB)
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=EPOCHES, validation_data=(x_test, y_test))
    model.summary()







