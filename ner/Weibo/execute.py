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

import get_config
import data_util
import gru

gConfig = get_config.get_config()
train_data = gConfig['train_data']
test_data = gConfig['test_data']
epochs = gConfig['epochs']

x_array, y_array = data_util.create_data(train_data)
a_array, b_array = data_util.create_data(test_data)
x_array, lang_tokenizer = data_util.tokenizer(x_array)


def train():
    print('Training data in %s' % gConfig['train_data'])
    checkpoint_dir = gConfig['model_data']
    steps_per_epoch = len(x_array) // gConfig['batch_size']
    ckpt = tf.io.gfile.exists(checkpoint_dir)
    if ckpt:
        gru.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    BUFFER_SIZE = len(x_array)
    dataset = tf.data.Dataset.from_tensor_slices((x_array, y_array)).shuffle(BUFFER_SIZE)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    start_time = time.time()
    for i in range(epochs):
        print('---Training epoch ' + str(i) + '---')
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = gru.train_step(inp, targ)
            total_loss += batch_loss

        step_time_epoch = (time.time() - start_time_epoch) / steps_per_epoch
        step_loss = total_loss / steps_per_epoch
        current_steps = +steps_per_epoch
        step_time_total = (time.time() - start_time) / current_steps
        print('当前epoch: {}'.format(str(i + 1)))
        print('训练总步数: {} 每步耗时: {}  最新每步耗时: {} 最新每步loss {:.4f}'.format(current_steps, step_time_total, step_time_epoch, step_loss.numpy()))
        print('-' * 100)
        gru.checkpoint.save(file_prefix=checkpoint_prefix)
        sys.stdout.flush()


def predict(sentence):
    checkpoint_dir = gConfig['model_data']
    gru.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    inputs = [lang_tokenizer.word_index.get(i, 3) for i in sentence]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=gConfig['max_inp'], padding='post')
    inputs = tf.convert_to_tensor(inputs)
    predictions = gru.base_model(inputs)
    return predictions


if __name__ == '__main__':
    if gConfig['mode'] == 'train':
        train()
    elif gConfig['mode'] == 'test':
        predict('小明很喜欢吃北京烤鸭')
        predict('小明很喜欢吃北京的烤鸭')
    else:
        print('train or test ... ')








