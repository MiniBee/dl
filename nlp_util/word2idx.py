#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/4/19 上午10:13
#@Author  :hongyue pei 
#@FileName: word2idx.py
#@Software: PyCharm

import pandas as pd
import jieba
from collections import Counter

SPLIT_KEY = '$$$'


def save_list(content_list, save_path):
    with open(save_path, 'w') as f:
        for line in content_list:
            if isinstance(line, list):
                line = [str(i) for i in line]
                line = SPLIT_KEY.join(line)
            f.write(line + '\n')


def tokens(content_list, save_path=None):
    tokens_list = []
    for line in content_list:
        tokens_list.append(SPLIT_KEY.join(list(jieba.cut(line.strip()))))
    if save_path:
        save_list(tokens_list, save_path)
    return tokens_list


def word_set(content_list, save_path=None):
    tokens_list = []
    for line in content_list:
        if not isinstance(line, list):
            line = line.replace('\n', '。').split(SPLIT_KEY)
        tokens_list.extend(line)
    cnt = Counter(tokens_list)
    ret = [i[0] for i in cnt.most_common()]
    if save_path:
        save_list(ret, save_path)
    return ret


def word2idx(content_word_list, word_idx, save_path=None):
    content_word_idx = []
    for line in content_word_list:
        temp = []
        for word in line.split(SPLIT_KEY):
            if word in word_idx:
                temp.append(word_idx.index(word) + 1)
            else:
                temp.append(0)
        content_word_idx.append(temp)
    if save_path:
        save_list(content_word_idx, save_path)
    return content_word_idx


if __name__ == '__main__':
    file_path = '/home/peihongyue/project/python/dl/data/has_ae.csv'
    data = pd.read_csv(file_path)[['y_True', 'ori']]
    content_list = data['ori'].tolist()
    target_list = data['y_True'].tolist()
    content_word_list = tokens(content_list)
    word_idx = word_set(content_word_list)
    content_idx = word2idx(content_word_list, word_idx)
    target_contect_list = []
    for i, line in enumerate(content_idx):
        target_contect_list.append([target_list[i]] + content_idx[i])
    save_list(target_contect_list, '/home/peihongyue/project/python/dl/data/has_ae_xy.cs')


