#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: test_cnn.py
# @time: 2020/7/9 上午10:19
# @desc:

import tensorflow as tf
import getConfig


class TextCNN(tf.keras.Model):
    def __init__(self, maxlen, vocab_size, embedding_dim, kernel_sizes=[3,4,5], class_num=1, last_activation='sigmoid'):
        super(TextCNN, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=maxlen)
        self.convs = []
        self.max_pooling = []
        for kernel in kernel_sizes:
            self.convs.append(tf.keras.layers.Conv1D(128, kernel, activation='relu'))
            self.max_pooling.append(tf.keras.layers.GlobalMaxPool1D())
        self.dense = tf.keras.layers.Dense(class_num, activation=last_activation)

    def call(self, sentence):
        x = self.embedding(sentence)
        convs = []
        for i, conv in enumerate(self.convs):
            c = conv(x)
            c = self.max_pooling[i](c)
            convs.append(c)
        x = tf.keras.layers.Concatenate(x)
        x = self.dense(x)
        return x


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

gConfig = getConfig.get_config()
maxlen = gConfig['maxlen']
vocab_size = gConfig['maxlen']
embedding_dim = gConfig['embedding_dim']
class_num = gConfig['class_num']
last_activation = gConfig['last_activation']

textcnn = TextCNN(maxlen, vocab_size, embedding_dim, [3,4,5], class_num, last_activation)
checkpoint = tf.train.Checkpoint(textcnn=textcnn, optimizer=optimizer)
manager = tf.train.CheckpointManager(checkpoint, directory='', max_to_keep=5)

def loss_function(real, pred):
    loss = 0
    for i in range(real.shape[0]):
        loss += loss_object(real, pred)
    return tf.reduce_mean(loss)


@tf.function
def train_step(input, target):
    loss = 0
    with tf.GradientTape() as tape:
        pred = textcnn(input)
        loss += loss_function(target, pred)
    variables = textcnn.trainable_variables
    gradients = tape.gradients(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss


if __name__ == '__main__':
    pass

