#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: execute.py
# @time: 2020/7/24 下午7:05
# @desc:


import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import time
import sys
import shutil

import get_config
import lstm
import utils


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()


def loss_function(real, pred):
    loss = loss_object(real, pred)
    return loss


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
        loss = total_loss / batch_size
        print('当前epoch: {}'.format(str(i + 1)))
        print('训练总步数: {} 每步耗时: {}  最新每步耗时: {} 最新每步loss {:.4f}'.format(current_steps, step_time_total, step_time_epoch,
                                                                      loss.numpy()))
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
    train(x_array, y_array, batch_size, epochs, checkpoint, get_config.get_config()['model_data'], lstm_model)


def loss_function(real, pred):
    loss = loss_object(real, pred)
    return loss


def val_loss(x_test, y_test, model):
    y_pred = model(x_test)
    return loss_function(y_test, y_pred)


if __name__ == '__main__':
    gconf = get_config.get_config()
    vocab_size = gconf.get('vocab_size')
    embedding_dim = gconf.get('embedding_dim')
    batch_size = gconf.get('batch_size')
    epochs = gconf.get('epochs')
    units = gconf.get('units')
    train_file = gconf['train_data']
    x_array, y_array = utils.load_data(train_file)
    print(x_array.shape)
    print(y_array.shape)
    # train_lstm(x_array, y_array, vocab_size, embedding_dim, units, batch_size, epochs)
    # --------------------------------------------------
    x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0.3)
    # baseline = lstm.BaseLine(8000, 1024, 256, 128)
    # vocab_size, embedding_dim, kernel_size, class_num
    baseline = lstm.TextCNN(vocab_size, embedding_dim, [2,3,5], 14)
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    checkpoint = tf.train.Checkpoint(baseline=baseline, optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint, directory='', max_to_keep=5)

    @tf.function
    def train_step(input, target):
        loss = 0
        with tf.GradientTape() as tape:
            pred = baseline(input)
            loss += loss_function(target, pred)
        variables = baseline.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss / input.shape[0]


    checkpoint_dir = gconf.get('model_data')
    if gconf.get('retrain') == 1:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    ckpt = tf.io.gfile.listdir(checkpoint_dir)
    if ckpt:
        print('reload pretrained model ... ')
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    BUFFER_SIZE = len(x_array)
    BATCH_SIZE=9
    print('-' * 100)
    steps_per_epoch = int(BUFFER_SIZE / BATCH_SIZE)
    dataset = tf.data.Dataset.from_tensor_slices((x_array, y_array)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    start_time = time.time()
    for i in range(epochs):
        start_time_epoch = time.time()
        total_loss = 0
        for (batch, (x, y)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(x, y)
            total_loss += batch_loss
            print('Epochs: {} 当前batch{}步loss {:.4f}'.format(str(i + 1), str(batch_size+1), batch_loss.numpy()))
        step_loss = total_loss / BATCH_SIZE
        checkpoint.save(file_prefix=checkpoint_prefix)
        loss_val = val_loss(x_test, y_test, baseline)
        print('Epochs: {} 训练集最新每步loss {:.4f}'.format(str(i + 1), step_loss.numpy()))
        print('Epochs: {} 测试集最新每步loss {:.4f}'.format(str(i + 1), loss_val.numpy()))
    baseline.summary()













