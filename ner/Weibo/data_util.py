#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: data_util.py
# @time: 2020/7/7 下午2:13
# @desc:

import get_config
import tensorflow as tf

gConfig = get_config.get_config()
print(gConfig)
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
            y.append(gConfig[b.lower()])
    return x_array, y_array


def tokenizer(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1000, oov_token=3)
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=gConfig['max_inp'], padding='post')
    return tensor, lang_tokenizer


if __name__ == '__main__':
    x_array, y_array = create_data(train_data)
    a_array, b_array = create_data(test_data)
    # a, b = tokenizer(x_array)
    print(y_array[0])









