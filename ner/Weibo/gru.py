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
        super(BaseModel, self).__ini__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units, return_sequences=True, return_state=True))
        self.dropout = tf.keras.layers.Dropout()
        self.fc = tf.keras.layers.Dense(7, activation='sigmoid')

    def call(self, x):
        x = self.embedding(x)
        x, _ = self.model(x)
        x = self.dropout(x)
        return self.fc(x)


gConfig = get_config.get_config()
print(gConfig)
vocab_size = gConfig['vocab_size']
embedding_dim = gConfig['embedding_dim']
model_data = gConfig['model_data']
units = gConfig['units']


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def loss_function(real, pred):
    loss = loss_object(real, pred)
    return tf.math.reduce_mean(loss)


base_model = BaseModel(units, vocab_size, embedding_dim)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, base_model=base_model)


@tf.function
def train_step(inp, tar):
    loss = 0
    with tf.GradientTape() as tape:
        predictions = base_model(inp)
        loss = loss_function(tar, predictions)

    variables = base_model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss








