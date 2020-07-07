#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: gru.py
# @time: 2020/7/7 下午3:39
# @desc:

import tensorflow as tf
import get_config


class BaseModel(tf.keras.Model):
    def __init__(self, units, vocab_size, embedding_dim):
        super(BaseModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.model = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)
        self.dropout = tf.keras.layers.Dropout(0.4)
        self.fc = tf.keras.layers.Dense(7, activation='softmax')

    def call(self, x):
        x = self.embedding(x)
        x, _ = self.model(x)
        x = self.dropout(x)
        x = self.fc(x)
        print(x)
        return x


gConfig = get_config.get_config()
vocab_size = gConfig['vocab_size']
embedding_dim = gConfig['embedding_dim']
model_data = gConfig['model_data']
units = gConfig['units']


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()


def loss_function(real, pred):
    loss = 0
    for i in range(real.shape[0]):
        loss += loss_object(real[i], pred[i])
    return loss


base_model = BaseModel(units, vocab_size, embedding_dim)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, base_model=base_model)


@tf.function
def train_step(inp, tar):
    with tf.GradientTape() as tape:
        predictions = base_model(inp)
        loss = loss_function(tar, predictions)
    variables = base_model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss








