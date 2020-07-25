#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: execute.py
# @time: 2020/7/24 下午7:05
# @desc:


import tensorflow as tf
import os
import time
import sys

import get_config
import lstm
import utils


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(3000, 1024),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
        tf.keras.layers.Dense(14, activation='softmax')
    ])
    return model


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()


def loss_function(real, pred):
    loss = 0
    for i in range(real.shape[0]):
        loss += loss_object(real[i], pred[i])
    return tf.reduce_mean(loss)


@tf.function
def train_step(base_model, inp, tar):
    with tf.GradientTape() as tape:
        predictions = base_model(inp)
        loss = loss_function(tar, predictions)
    variables = base_model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss

gconf = get_config.get_config()
def train(x_array, y_array, batch_size, epochs, checkpoint, checkpoint_dir, model):
    print('Training data in %s' % gconf['train_data'])
    steps_per_epoch = len(x_array) // gconf['batch_size']
    ckpt = tf.io.gfile.exists(checkpoint_dir)
    if ckpt:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    BUFFER_SIZE = len(x_array)
    dataset = tf.data.Dataset.from_tensor_slices((x_array, y_array)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    start_time = time.time()
    for i in range(epochs):
        print('---Training epoch ' + str(i) + '---')
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(model, inp, targ)
            total_loss += batch_loss

        step_time_epoch = (time.time() - start_time) / steps_per_epoch
        current_steps = +steps_per_epoch
        step_time_total = (time.time() - start_time) / current_steps
        print('当前epoch: {}'.format(str(i + 1)))
        print('训练总步数: {} 每步耗时: {}  最新每步耗时: {} 最新每步loss {:.4f}'.format(current_steps, step_time_total, step_time_epoch, total_loss.numpy()))
        print('=' * 100)
        checkpoint.save(file_prefix=checkpoint_prefix)
        sys.stdout.flush()


def predict(inputs, model):
    print(inputs.shape)
    predictions = model(inputs)
    return predictions


def train_lstm(x_array, y_array, vocab_size, embedding_dim, units, batch_size, epochs):
    lstm_model = lstm.BaseLine(vocab_size, embedding_dim, units, 128).build_model()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, lstm_model=lstm_model)
    train(x_array, y_array, batch_size, epochs, checkpoint, '', lstm_model)


if __name__ == '__main__':
    gconf = get_config.get_config()
    vocab_size = gconf.get('vocab_size')
    embedding_dim = gconf.get('embedding_dim')
    batch_size = gconf.get('batch_size')
    epochs = gconf.get('epochs')
    units = gconf.get('units')
    train_file = gconf['train_data']
    x_array, y_array = utils.load_data(train_file)
    x_array = tf.keras.preprocessing.sequence.pad_sequences(x_array, maxlen=gconf['max_inp'], padding='post')
    print(y_array)
    train_lstm(x_array, y_array, vocab_size, embedding_dim, units, batch_size, epochs)

