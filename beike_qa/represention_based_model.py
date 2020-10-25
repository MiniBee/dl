#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   represention_based_model.py
@Time    :   2020/10/19 11:22:06
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

import tensorflow as tf 
from tensorflow.keras import backend as K

class Attention_layer(tf.keras.layers.Layer):
    def __init__(self, regularizer=None):
        super(Attention_layer, self).__init__()
        self.regularizer = regularizer

    def build(self, input_shape):
        self.weight = self.add_weight(shape=(input_shape[-1], 1), 
                            regularizer=self.regularizer,
                            trainable=True)
        super(Attention_layer, self).build(input_shape)
    
    def call(self, inputs):
        attention_in = K.exp(K.squeeze(K.dot(inputs, self.weight), axis=-1))
        attention = attention_in/K.expand_dims(K.sum(attention_in, axis=-1), -1)
        weighted_sum = K.batch_dot(K.permute_dimensions(inputs, [0, 2, 1]), attention)
        return weighted_sum

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        base_config = super(HAttention, self).get_config()
        return dict(list(base_config.items()))



class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))
        self.lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))
        self.dense1 = tf.keras.layers.Dense(1024)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x):
        x1 = x[:, :7]
        x2 = x[:, 7:]
        x1 = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)(x1)
        x2 = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)(x2)
        _, x1 = self.lstm1(x1)
        _, x2 = self.lstm2(x2)
        x = tf.concat([x1, x2], axis=0)
        x = self.dense1(x)
        x = self.dropout(x)
        # attention layer
        l2_reg = tf.keras.regularizers.l2(1e-8)
        x = Attention_layer(l2_reg)(x)
        x = Atten
        x = self.dense2(x)
        print('-' * 50, tf.keras.backend.sigmoid(x))
        return tf.keras.backend.sigmoid(x)


def train_step(model, data, labels, optimizer):
    with tf.GradientTape() as tape:
        pred = model(data)
        loss = tf.keras.losses.BinaryCrossentropy(pred, labels)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


if __name__ == "__main__":
    model = Model(10, 5)
    model(tf.random.uniform(shape=(2, 18)))
    





