#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/3/14 下午4:46
#@Author  :hongyue pei 
#@FileName: data_flow.py
#@Software: PyCharm

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import jieba
import re
import os

jieba.load_userdict('/home/peihongyue/project/python/dl/data/bc/drug')


def text_clean(string):
    string = string.replace(' ', '')
    return list(jieba.cut(string))


def read_file(path):
    df = pd.read_csv(path)
    df['text'] = df.apply(lambda x: x['现病史（最近一次乳腺癌住院病历，后同）'] if x['现病史（最近一次乳腺癌住院病历，后同）'] else '' +
                                                                                                          x['诊疗过程描述'] if x['诊疗过程描述'] else '', axis=1)
    df.rename(columns={'几线治疗': 'target'}, inplace=True)
    df = df.fillna('')
    df = df[df['text'] != '']
    # df['text'] = df['text'].apply(lambda x: text_clean(x))
    df['text'] = df['text'].apply(lambda x: text_clean(x))
    df['target'] = df['target'].apply(lambda x: 1 if x == 3 else 0)
    return np.array(df[['text', 'target']])


def get_word_idx(origin_data):
    words = set()
    for line in origin_data:
        for word in line[0]:
            words.add(word)
    with open('/home/peihongyue/project/python/dl/data/bc/wordIdx', 'w') as f:
        for line in words:
            f.write(line + '\n')


def word2idx(origin_data):
    if not os.path.exists('/home/peihongyue/project/python/dl/data/bc/wordIdx'):
        get_word_idx(origin_data)
    wordIdx = []
    with open('/home/peihongyue/project/python/dl/data/bc/wordIdx') as f:
        for line in f:
            wordIdx.append(line.replace('\n', ''))
    return wordIdx


def sentence2idx(origin_data, wordIdx):
    ret_list = []
    for line in origin_data:
        sentence_idx = []
        word_list = line[0]
        for word in word_list:
            sentence_idx.append(str(wordIdx.index(word)))
        target = line[1]
        ret_list.append(','.join([str(target)] + sentence_idx))
    return ret_list


def save(path, data_list):
    with open(path, 'w') as f:
        for data in data_list:
            f.write(data + '\n')


if __name__ == '__main__':
    path = '/home/peihongyue/project/python/dl/data/bc/bc.csv'
    origin_data = read_file(path)
    wordIdx = word2idx(origin_data)
    sentenceIdx = sentence2idx(origin_data, wordIdx)
    train, test = train_test_split(sentenceIdx, test_size=0.3)
    save('/home/peihongyue/project/python/dl/data/bc/trainVec', train)
    save('/home/peihongyue/project/python/dl/data/bc/testVec', test)





