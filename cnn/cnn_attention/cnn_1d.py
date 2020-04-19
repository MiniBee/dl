#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/4/19 上午10:13
#@Author  :hongyue pei 
#@FileName: cnn_1d.py
#@Software: PyCharm


import sys
import tensorflow as tf
sys.path.append('/home/peihongyue/project/python/dl/')
import numpy as np

from nlp_util import data_flow
from nlp_util import pre_process

try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


class CnnBlock(tf.keras.layers.Layer):
    def __init__(self, filter_size, kernel_size, padding='valid'):

        super(CnnBlock, self).__init__()
        self.cnn_layer = tf.keras.layers.Conv2D(filters=filter_size, kernel_size=kernel_size, padding=padding)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activacation = tf.keras.layers.ReLU()

    def call(self, input):
        input = self.cnn_layer(input)
        input = self.batch_norm(input)
        input = self.activacation(input)
        return input


class TextCnn(tf.keras.Model):
    def __init__(self, vocab_size, d_model, cnn_filters):
        super(TextCnn, self).__init__()
        self.cnn_count = len(cnn_filters)
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.cnn_layer = [CnnBlock(1, (i, d_model)) for i in cnn_filters]
        self.global_average_pooling = [tf.keras.layers.GlobalAveragePooling2D() for _ in cnn_filters]
        self.global_max_pooling = [tf.keras.layers.GlobalMaxPooling2D() for _ in cnn_filters]
        self.drop_out = tf.keras.layers.Dropout(0.5)
        self.dense1 = tf.keras.layers.Dense(128)
        self.dense2 = tf.keras.layers.Dense(1)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, input):
        input = self.embedding(input)
        input = tf.expand_dims(input, -1)
        input2 = None
        for i in range(self.cnn_count):
            a = self.cnn_layer[i](input)
            b = self.global_average_pooling[i](a)
            # a = tf.reshape(a, shape=(-1, a.shape[1]*a.shape[3]))
            # print('aaa222', a)
            # print('bbb', b)
            if i == 0:
                input2 = b
            else:
                input2 = tf.concat([b, input2], axis=1)
        input = self.drop_out(input2)
        input = self.dense1(input)
        input = self.dense2(input)
        return self.sigmoid(input)


if __name__ == '__main__':
    train_patn = '/home/peihongyue/project/python/dl/data/has_ae_train.csv'
    test_path = '/home/peihongyue/project/python/dl/data/has_ae_test.csv'
    x_train, y_train = data_flow.get_data(train_patn)
    x_test, y_test = data_flow.get_data(test_path)

    x_train, x_test = pre_process.padding(x_train, x_test)

    vocab_size = 6349
    d_model = 1024
    cnn_filters = [2, 3, 4]
    model = tf.keras.Sequential([
        TextCnn(vocab_size, d_model, cnn_filters)
    ])

    adam = tf.keras.optimizers.Adam(0.001)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

    model.fit(x_train, y_train, batch_size=64, epochs=80, validation_data=(x_test, y_test))
    model.summary()




