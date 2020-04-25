#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: han.py
# @time: 2020/4/20 上午9:46
# @desc:

import tensorflow as tf
import pandas as pd
import jieba
import json
import numpy as np
# print(tf.__version__)


# text = ['she is a good student', 'xiaoming is good at basketball', 'he is Chinese']
# tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=50)
# tokenizer.fit_on_texts(text)
# # print(tokenizer.texts_to_sequences(text))
#
# tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer.to_json())
# print(tokenizer.texts_to_sequences(text))


def load_data():
    data = pd.read_csv('/Users/peihongyue/phy/project/dl/data/bc/bc.csv')
    data = data.fillna('')
    data['text'] = data.apply(lambda x: x['现病史（最近一次乳腺癌住院病历，后同）'] + x['诊疗过程描述'], axis=1)
    # data['text'] = data['text'].apply(lambda x: list(jieba.cut(str(x))))
    data['target'] = data['几线治疗'].apply(lambda x: 1 if x == '3' or x == 3 else 0)
    return data


def get_token(text_list=None):
    if text_list:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=9000)
        tokenizer.fit_on_texts(text_list)
        with open('/Users/peihongyue/phy/project/dl/data/bc/bc_tokenizer', 'w') as f:
            f.write(json.dumps(tokenizer.to_json()))
    else:
        with open('/Users/peihongyue/phy/project/dl/data/bc/bc_tokenizer') as f:
            line = f.readline()
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.loads(line))
    return tokenizer


class HierarchicalAttentionNetwork(tf.keras.layers.Layer):
    def __init__(self, attention_dim):
        super(HierarchicalAttentionNetwork, self).__init__()
        self.attention_dim = attention_dim
        self.dense1 = tf.keras.layers.Dense(self.attention_dim)
        self.activation1 = tf.keras.activations.tanh
        self.dense2 = tf.keras.layers.Dense(1)
        self.activation2 = tf.keras.activations.softmax

    def call(self, inputs):
        output = self.dense1(inputs)
        output = self.activation1(output)
        output = self.dense2(output)
        w = self.activation2(output)
        output = tf.reduce_mean(w * inputs, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]


def han_model(x_train):
    # sentence_encoder = tf.keras.Sequential([
    input1 = tf.keras.layers.Input(shape=(100,))
    a = tf.keras.layers.Embedding(9041, 512)(input1)
    a = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(100, return_sequences=True))(a)
    a1 = HierarchicalAttentionNetwork(128)(a)
    m1 = tf.keras.Model(input1, a1)
    m1.summary()
    # ])
    # model = tf.keras.Sequential([
    input2 = tf.keras.layers.Input(shape=(20, 100))
    a = tf.keras.layers.TimeDistributed(m1)(input2)
    a = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(100, return_sequences=True))(a)
    a = HierarchicalAttentionNetwork(128)(a)
    pred = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(a)
    # ])

    model = tf.keras.Model(input2, pred)
    model.summary()
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])


def load_data():
    pass


if __name__ == '__main__':
    max_sentence = 20
    max_word = 100
    # data = load_data()[['text', 'target']]
    # text_list = data['text'].tolist()
    # target = data['target'].tolist()
    # tokenizer = get_token()
    # word_idx = tokenizer.word_index
    # print(len(word_idx))
    # text_idx = tokenizer.texts_to_sequences(text_list)

    data = np.zeros(shape=(10, max_sentence, max_word))
    # for i, context in enumerate(text_list):
    #     print(i)
    #     if len(context.split('。')) > 200:
    #         print(context)
    #     for j, sentence in enumerate(context.split('。')):
    #         if len(list(jieba.cut(sentence))) > 1000:
    #             print(sentence)
    #         for k, word in enumerate(list(jieba.cut(sentence))):
    #             data[i, j, k] = word_idx.get(word) or 0
    # print(data[:2])
    han_model(np.array(data))










