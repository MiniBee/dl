#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: lstm.py
# @time: 2020/7/24 下午6:43
# @desc:

import tensorflow as tf
from sklearn.model_selection import train_test_split

import get_config

try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

class_label = {'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}


# BaseLine
class BaseLine():
    def __init__(self, vocab_size, embedding_dim, units, max_len):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.units = units
        self.max_len = max_len

    def build_model(self):
        inputs = tf.keras.layers.Input(shape=(self.max_len, ))
        x_tensor = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)(inputs)
        x_tensor = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.units, return_sequences=True))(x_tensor)
        outputs = tf.keras.layers.Dense(14, activation='softmax')(x_tensor)

        model = tf.keras.Model(inputs, outputs)
        model.summary()
        return model


if __name__ == '__main__':
    BaseLine(2000, 1024, 256, 128).build_model()










