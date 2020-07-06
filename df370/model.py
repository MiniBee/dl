#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: model.py
# @time: 2020/6/2 下午3:35
# @desc:

import tensorflow as tf
import sys
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append('/home/peihongyue/project/python/dl/')
sys.path.append('/Users/peihongyue/phy/project/dl')

from nlp_util import word2idx


class Model():
    def __init__(self, hidden_num, loss, batch_size=2, epochs=2, opt='adam', activation='sigmoid'):
        self.hidden_num = hidden_num
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.opt = opt
        self.activation = activation


    def build_model(self):
        inputs = tf.keras.layers.Input(shape=(20))
        outputs = tf.keras.layers.Embedding(11475, 512)(inputs)
        outputs = tf.keras.layers.GRU(self.hidden_num)(outputs)
        outputs = tf.keras.layers.LayerNormalization()(outputs)
        outputs = tf.keras.layers.Dense(1, activation=self.activation)(outputs)
        model = tf.keras.Model(inputs, outputs)
        model.compile(loss=self.loss, optimizer=self.opt)
        model.summary()
        return model

    def train(self, x_train, y_train, x_test, y_test, save_path, logpath):
        callbacks = [
            tf.keras.callbacks.TensorBoard(logpath),
            tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True)
        ]
        self.model = self.build_model()
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(x_test, y_test), callbacks=callbacks)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)


if __name__ == '__main__':
    comment_list, label_list = word2idx.get_date('/data/phy/datas/df370/widx.csv', vocab_size=20)
    x_train, x_test, y_train, y_test = train_test_split(comment_list, label_list)

    model = Model(32, 'binary_crossentropy')
    model.build_model()
    model.train(x_train, y_train, x_test, y_test, '/data/phy/datas/df370/model.h5', '/data/phy/datas/df370/log')
    












