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


class Han(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Han, self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    def call(self):
        pass



if __name__ == '__main__':
    max_sentence = 200
    max_word = 1000
    data = load_data()[['text', 'target']]
    text_list = data['text'].tolist()
    target = data['target'].tolist()
    tokenizer = get_token()
    word_idx = tokenizer.word_index
    # text_idx = tokenizer.texts_to_sequences(text_list)
    # data = np.zeros(len(text_list), max_sentence, max_word)

    for i, context in enumerate(text_list):
        print(i)
        if len(context.split('。')) > 200:
            print(context)
        for j, sentence in enumerate(context.split('。')):
            if len(list(jieba.cut(sentence))) > 1000:
                print(sentence)
            for k, word in enumerate(list(jieba.cut(sentence))):
                data[i, j, k] = word_idx.get(word) or 0
    print(data[:2])










