#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: autoencoder.py
# @time: 2020/9/23 上午11:15
# @desc:

import tensorflow as tf
import os
import joblib
import pandas as pd

curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(curPath)[0])[0]
import sys
sys.path.append(root_path)
from src.utils.tools import format_data


class AutoEncoder(object):
    def __init__(self, max_len, vocab_size, embedding_dim):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.init_model()

    def init_model(self):
        input = tf.keras.layers.Input(shape=(self.max_len, ))
        encoder = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)(input)
        encoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(75, return_sequences=True))(encoder)
        encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(25, return_sequences=True, activity_regularizer=tf.keras.regularizers.l1(1e-4)))(encoder)

        encoder_output = tf.keras.layers.Dense(self.embedding_dim)(encoder)
        # encoder = tf.keras.layers.Dense(50, activation='relu')(encoder)
        # encoder = tf.keras.layers.Dense(self.embedding_dim)(encoder)

        decoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(75, return_sequences=True))(encoder_output)
        decoder = tf.keras.layers.GlobalMaxPool1D()(decoder)
        decoder = tf.keras.layers.Dense(50, activation='relu')(decoder)
        decoder = tf.keras.layers.Dense(self.max_len)(decoder)

        self.model = tf.keras.Model(inputs=input, outputs=decoder)
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
        self.encoder = tf.keras.Model(inputs=input, outputs=encoder_output)
        self.encoder.summary()

    def train(self, data, epochs=10):
        self.x, self.tokenizer = format_data(data, self.max_features, self.max_len, shuffle=True)
        self.model.fit(self.x, self.x, epochs=epochs, batch_size=32)

    def save(self):
        # joblib.dump 模型保存
        joblib.dump(self.tokenizer, root_path + '/model/embedding/tokenizer')
        self.model.save_weights(root_path + '/model/embedding/autoencoder')
        self.encoder.save_weights(root_path + '/model/embedding/autoencoder_encoder')

    def load(self):
        # joblib.load 模型加载
        self.tokenizer = joblib.load(root_path + '/model/embedding/tokenizer')
        self.model.load_weights(root_path + '/model/embedding/autoencoder')
        self.encoder.load_weights(root_path + '/model/embedding/autoencoder_encoder')


if __name__ == '__main__':
    autoEncoder = AutoEncoder(100, 10000, 1024)
    data = pd.read_csv('')
    autoEncoder.train(data, 10)
    autoEncoder.save()







