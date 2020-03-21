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

D_MODEL = 256
D_POINT_WISE_FF = 512
ENCODER_COUNT = 1
EPOCHS = 20
ATTENTION_HEAD_COUNT = 8
DROPOUT_PROB = 0.1
BATCH_SIZE = 2
BPE_VOCAB_SIZE = 7494


class Mask():
    def __init__(self):
        pass

    def create_padding_mask(self, sequences):
        sequences = tf.cast(tf.math.equal(sequences, 0), dtype=tf.float32)
        return sequences[:, tf.newaxis, tf.newaxis, :]

    def create_mask(self, inputs):
        encoder_padding_mask = self.create_padding_mask(inputs)
        return encoder_padding_mask


def load_data(path):
    x_train = []
    y_train = []
    with open(path) as f:
        for line in f:
            i = line.split(',')
            i = [float(a.strip()) for a in i]
            # i = [len(i)] + i
            if len(i) < 4001:
                i += [0.] * (4001 - len(i) + 1)
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
    print('-' * 50, 'start learing ... ')
    model = tf.keras.Sequential([
        transformer.Transformer(BPE_VOCAB_SIZE, ENCODER_COUNT, ATTENTION_HEAD_COUNT, D_MODEL, D_POINT_WISE_FF, DROPOUT_PROB, BATCH_SIZE)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test))
    model.summary()










