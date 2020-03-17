#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: lstm.py
# @time: 2020/3/17 下午3:56
# @desc:

import tensorflow as tf
import numpy as np
import os

try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


BPE_VOCAB_SIZE = 5000

def load_data(path):
    x_train = []
    y_train = []
    with open(path) as f:
        for line in f:
            i = line.split(',')
            i = [float(a.strip()) for a in i]
            if len(i) < BPE_VOCAB_SIZE:
                i += [0.0] * (BPE_VOCAB_SIZE - len(i) + 1)
            x_train.append(i[1:])
            y_train.append(i[0])
    return np.array(x_train), np.array(y_train)

if __name__ == '__main__':
    x_train, y_train = load_data(os.path.join(os.getcwd() + '/../../data/bc/trainVec'))
    x_test, y_test = load_data(os.path.join(os.getcwd() + '/../../data/bc/testVec'))


    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(7494, 512),
        tf.keras.layers.LSTM(256, return_sequences=True),
        tf.keras.layers.LSTM(256, return_sequences=True),
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.build(input_shape=(167, 5000))
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))




