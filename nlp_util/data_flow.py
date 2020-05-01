#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/4/19 下午3:10
#@Author  :hongyue pei 
#@FileName: data_flow.py
#@Software: PyCharm

import sys
# sys.path.append('/home/peihongyue/project/python/dl/')
sys.path.append('/Users/peihongyue/project/python/dl/')
from nlp_util import word2idx
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data():
    file_path = '/home/peihongyue/project/python/dl/data/has_ae.csv'
    data = pd.read_csv(file_path)[['y_True', 'ori']]
    content_list = data['ori'].tolist()
    target_list = data['y_True'].tolist()
    content_word_list = word2idx.tokens(content_list)
    word_idx = word2idx.word_set(content_word_list)
    content_idx = word2idx.word2idx(content_word_list, word_idx)
    x_train, x_test, y_train, y_test = train_test_split(content_idx, target_list, test_size=0.3)
    target_contect_list = []
    for i, line in enumerate(x_train):
        target_contect_list.append([y_train[i]] + x_train[i])
    word2idx.save_list(target_contect_list, '/home/peihongyue/project/python/dl/data/bc_train.csv')

    target_contect_list = []
    for i, line in enumerate(x_test):
        target_contect_list.append([y_test[i]] + x_test[i])
    word2idx.save_list(target_contect_list, '/home/peihongyue/project/python/dl/data/bc_test.csv')


def get_data(path):
    x = []
    y = []
    with open(path) as f:
        for line in f:
            t_list = line.replace('\n', '').split('$$$')
            y.append(int(t_list[0]))
            x.append([int(i) for i in t_list[1:]])
    return x, y


def get_sentece_word(path, sentence_split_key, word_split_key, max_sentence, max_word):
    x = []
    y = []
    with open(path) as f:
        for line in f:
            sentence_list = line.replace('\n', '').split(sentence_split_key)
            for i, sentence in enumerate(sentence_list):
                if i > max_sentence:
                    break
                word_list = sentence.split(word_split_key)
                for j, word in enumerate(word_list):
                    pass


def pre_bc():
    file_path = '/Users/peihongyue/phy/project/dl/data/bc/'
    # file_path = '/home/peihongyue/project/python/dl/data/bc/'
    data = pd.read_csv(file_path + 'bc.csv')
    data = data.fillna('')
    data['text'] = data.apply(lambda x: x['现病史（最近一次乳腺癌住院病历，后同）'] + x['诊疗过程描述'], axis=1)
    data['target'] = data['几线治疗'].apply(lambda x: 1 if x == '3' or x == 3 else 0)

    content_list = data['text'].tolist()
    target_list = data['target'].tolist()

    content_word_list = word2idx.tokens(content_list)
    word_idx = word2idx.word_set(content_word_list, file_path + 'bc_word.idx')
    content_idx = word2idx.word2idx(content_word_list, word_idx)
    x_train, x_test, y_train, y_test = train_test_split(content_idx, target_list, test_size=0.3)
    target_contect_list = []
    for i, line in enumerate(x_train):
        target_contect_list.append([y_train[i]] + x_train[i])
    word2idx.save_list(target_contect_list, file_path + '/bc_train.csv')

    target_contect_list = []
    for i, line in enumerate(x_test):
        target_contect_list.append([y_test[i]] + x_test[i])
    word2idx.save_list(target_contect_list, file_path + '/bc_test.csv')


if __name__ == '__main__':
    pre_bc()
