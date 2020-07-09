#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: execute.py
# @time: 2020/7/7 下午4:15
# @desc:

import tensorflow as tf
import time
import os
import sys
import numpy as np

import get_config
import data_util
import gru

gConfig = get_config.get_config()
train_data = gConfig['train_data']
test_data = gConfig['test_data']
epochs = gConfig['epochs']
batch_size = gConfig['batch_size']

x_array, y_array = data_util.create_data(train_data)
a_array, b_array = data_util.create_data(test_data)
x_array, lang_tokenizer = data_util.tokenizer(x_array, 'UNK', 0)
y_array = data_util.padding_target(y_array, gConfig['max_inp'])
y_array = np.expand_dims(y_array, 2)
print(x_array.shape)
print(y_array.shape)


def train():
    print('Training data in %s' % gConfig['train_data'])
    checkpoint_dir = gConfig['model_data']
    steps_per_epoch = len(x_array) // gConfig['batch_size']
    ckpt = tf.io.gfile.exists(checkpoint_dir)
    if ckpt:
        gru.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    BUFFER_SIZE = len(x_array)
    dataset = tf.data.Dataset.from_tensor_slices((x_array, y_array)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    start_time = time.time()
    for i in range(epochs):
        print('---Training epoch ' + str(i) + '---')
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = gru.train_step(inp, targ)
            total_loss += batch_loss

        step_time_epoch = (time.time() - start_time) / steps_per_epoch
        step_loss = total_loss / steps_per_epoch
        current_steps = +steps_per_epoch
        step_time_total = (time.time() - start_time) / current_steps
        print('当前epoch: {}'.format(str(i + 1)))
        print('训练总步数: {} 每步耗时: {}  最新每步耗时: {} 最新每步loss {:.4f}'.format(current_steps, step_time_total, step_time_epoch, total_loss.numpy()))
        print('=' * 100)
        gru.checkpoint.save(file_prefix=checkpoint_prefix)
        sys.stdout.flush()
    


def predict(sentence):
    checkpoint_dir = gConfig['model_data']
    gru.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    inputs = [lang_tokenizer.word_index.get(i, 3) for i in sentence]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=gConfig['max_inp'], padding='post')
    inputs = tf.convert_to_tensor(inputs)
    print(inputs.shape)
    predictions = gru.base_model(inputs)
    return predictions


if __name__ == '__main__':
    if gConfig['mode'] == 'train':
        train()
    elif gConfig['mode'] == 'test':
        print(predict('小明住在北京'))
    else:
        print('train or test ... ')








