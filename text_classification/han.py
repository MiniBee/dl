#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: han.py
# @time: 2020/4/20 上午9:46
# @desc:

import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
import jieba
import json
import sys
sys.path.append('/home/peihongyue/project/python/dl/')
sys.path.append('/Users/peihongyue/phy/project/dl')
import numpy as np

from nlp_util import data_flow
from nlp_util import pre_process
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
        self.output1 = self.dense1(inputs)
        output1 = self.activation1(self.output1)
        output1 = self.dense2(output1)
        self.weight = self.activation2(output1)
        output1 = tf.reduce_mean(self.weight * inputs, axis=1)
        return output1

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def han_model(x_train, y_train, x_test, y_test):
    # sentence_encoder = tf.keras.Sequential([
    input1 = tf.keras.layers.Input(shape=(100,))
    a = tf.keras.layers.Embedding(9041, 512)(input1)
    a = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(100, return_sequences=True))(a)
    han1 = HierarchicalAttentionNetwork(100)
    a1 = han1(a)
    m1 = tf.keras.Model(input1, a1)
    m1.summary()
    # ])
    # model = tf.keras.Sequential([
    input2 = tf.keras.layers.Input(shape=(20, 100))
    # print('input2: ', input2)
    a = tf.keras.layers.TimeDistributed(m1)(input2)
    # print('a1: ', a)
    a = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(100, return_sequences=True))(a)
    # print('a2: ', a)
    # a = HierarchicalAttentionNetwork(100)(a)
    han = HierarchicalAttentionNetwork(100)
    a = han(a)
    # print('a3: ', a)
    pred = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(a)
    # print('pred: ', pred)
    # ])

    model = tf.keras.Model(input2, pred)
    model.summary()
    adam = tf.keras.optimizers.Adam(0.001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    print('x_trian shape: ', x_train.shape)
    print('y_trian shape: ', y_train.shape)
    model.fit(x_train[:32], y_train[:32], batch_size=32, epochs=1, validation_data=(x_test, y_test))

    layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[3].output)
    layer_model2 = tf.keras.Model(inputs=m1.input, outputs=m1.layers[3].output)

    # sentence_w = layer_model.predict(x_train)
    # print(sentence_w)
    # word_w = layer_model2.predict()
    fn = K.function([input2], [han.weight])
    sentence_w = fn(x_train)
    print(sentence_w)


def load_data(path, max_sentence, max_word):
    x = []
    y = []
    with open(path) as f:
        for line in f:
            data = np.zeros(shape=(max_sentence, max_word))
            sentence_list = line.split('$$$3$$$')
            for i, sentence in enumerate(sentence_list):
                word_list = sentence.split('$$$')
                if i == 0:
                    y.append(float(word_list[0]))
                    continue
                if i - 1 >= max_sentence:
                    break
                for j, word in enumerate(word_list):
                    if j >= max_word:
                        break
                    data[i - 1, j] = float(word)
            x.append(data)
    return np.array(x), np.array(y)


def main():
    max_sentence = 20
    max_word = 100
    x_train, y_train = load_data('/Users/peihongyue/phy/project/dl/data/bc/bc_train.csv', max_sentence, max_word)
    x_test, y_test = load_data('/Users/peihongyue/phy/project/dl/data/bc/bc_test.csv', max_sentence, max_word)
    print('x_train shape: ', x_train.shape)
    print('y_train shape:', y_train.shape)
    han_model(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()










