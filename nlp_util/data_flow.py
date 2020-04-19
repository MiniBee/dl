#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/4/19 下午3:10
#@Author  :hongyue pei 
#@FileName: data_flow.py
#@Software: PyCharm

import sys
sys.path.append('/home/peihongyue/project/python/dl/')
from nlp_util.word2idx import word2idx
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
    word2idx.save_list(target_contect_list, '/home/peihongyue/project/python/dl/data/has_ae_train.csv')

    target_contect_list = []
    for i, line in enumerate(x_test):
        target_contect_list.append([y_test[i]] + x_test[i])
    word2idx.save_list(target_contect_list, '/home/peihongyue/project/python/dl/data/has_ae_test.csv')


def get_data(path):
    x = []
    y = []
    with open(path) as f:
        for line in f:
            t_list = line.replace('\n', '').split('$$$')
            y.append(int(t_list[0]))
            x.append([int(i) for i in t_list[1:]])
    return x, y



if __name__ == '__main__':
    split_data()
