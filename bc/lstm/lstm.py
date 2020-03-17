#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: lstm.py
# @time: 2020/3/17 下午3:56
# @desc:

import tensorflow as tf
import numpy as np
import os


BPE_VOCAB_SIZE = 3000

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
    return np.array(x_train[:20]), np.array(y_train[:20])

if __name__ == '__main__':
    x_train, y_train = load_data(os.path.join(os.getcwd() + '/../../data/bc/trainVec'))
    x_test, y_test = load_data(os.path.join(os.getcwd() + '/../../data/bc/testVec'))
    print(np.array(x_train).shape)
    print(y_train)
    # out = tf.keras.layers.Embedding(7494, 512)(x_train)
    # print(out)
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(7494, 512),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=5, epochs=10, validation_data=(x_test, y_test))




