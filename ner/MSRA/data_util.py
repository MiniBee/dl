#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: data_util.py
# @time: 2020/7/7 下午2:13
# @desc:

import get_config
import tensorflow as tf
import numpy as np

gConfig = get_config.get_config()
train_data = gConfig['train_data']
test_data = gConfig['test_data']

def create_data(file_name):
    x_array = []
    y_array = []
    with open(file_name) as f:
        x, y = [], []
        for line in f:
            if line.strip() == '':
                x_array.append(x)
                y_array.append(y)
                x, y = [], []
            line = line.strip().split('\t')
            if len(line) < 2:
                continue
            a = line[0]
            b = line[1]
            x.append(a)
            y.append(b)
    return x_array, y_array


def tokenizer(lang, oov_token, value=0.0,):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=20000, oov_token=oov_token)
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=gConfig['max_inp'], padding='post', value=value)
    return tensor, lang_tokenizer


def padding_target(y_array, max_inp):
    res_array = []
    for l_array in y_array:
        if len(l_array) < max_inp:
            res_array.append([gConfig[i.lower()] for i in l_array + ['O'] * (max_inp - len(l_array))])
        if len(l_array) >= max_inp:
            res_array.append([gConfig[i.lower()] for i in l_array[:max_inp]])
    return np.array(res_array)


if __name__ == '__main__':
    max_inp = gConfig['max_inp']
    x_array, y_array = create_data(train_data)
    y_array = padding_target(y_array, max_inp)
    print(y_array[0])
    print(y_array[1])
    print(y_array[99])
    # a_array, b_array = create_data(test_data)
    # a, b = tokenizer(x_array, 'UNK')
    # c, d = tokenizer(y_array, 'o')
    # print(d.word_index)
    # print(d.index_word)









