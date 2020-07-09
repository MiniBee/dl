#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: data_util.py
# @time: 2020/7/9 上午11:09
# @desc:

import tensorflow as tf
import numpy as np
import jieba

import getConfig

gConfig = getConfig.get_config()
vocab_size = gConfig['vocab_size']
train_data = gConfig['train_data']


def create_date(file_name):
    x_array = []
    y_array = []
    with open(file_name) as f:
        for line in f:
            line = line.split('_!_')
            sentence = line[3]
            target = line[1]
            x_array.append(list(jieba.cut(sentence)))
            y_array.append(target)
    return x_array, y_array


def tokenizer(sentence):
    lang_tonken = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='UNK')
    lang_tonken.fit_on_texts(sentence)
    tensor = lang_tonken.texts_to_sequences(sentence)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=gConfig['max_len'], padding='post')
    return tensor, lang_tonken


if __name__ == '__main__':
    x_array, y_array = create_date(train_data)
    x_array, lang_token = tokenizer(x_array)
    y_array = [float(i) - 101 for i in y_array]
    print(x_array[:10])
    print(y_array[:10])


