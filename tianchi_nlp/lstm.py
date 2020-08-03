#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: lstm.py
# @time: 2020/7/24 下午6:43
# @desc:

import tensorflow as tf
from sklearn.model_selection import train_test_split
import time

import get_config

try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

class_label = {'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}


# BaseLine
class BaseLine(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, max_len):
        super(BaseLine, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(units)
        self.bilstm = tf.keras.layers.Bidirectional(self.lstm)
        self.dense = tf.keras.layers.Dense(14, activation='softmax')

    def call(self, inputs):
        x_tensor = self.embedding(inputs)
        x_tensor = self.bilstm(x_tensor)
        outputs = self.dense(x_tensor)
        return outputs


class TextCNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, kernel_size, class_num):
        super(TextCNN, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.convs = []
        self.max_pooling = []
        for k in kernel_size:
            self.convs.append(tf.keras.layers.Conv1D(2, kernel_size=k))
            self.max_pooling.append(tf.keras.layers.GlobalMaxPool1D())
        self.dense = tf.keras.layers.Dense(class_num, activation='softmax')

    def call(self, inputs):
        inputs = self.embedding(inputs)
        tensor = []
        for i, conv in enumerate(self.convs):
            c = conv(inputs)
            c = self.max_pooling[i](c)
            tensor.append(c)
        inputs = tf.keras.layers.Concatenate()(tensor)
        outputs = self.dense(inputs)
        return outputs


class Han(tf.keras.Model):
    pass


if __name__ == '__main__':
    pass












