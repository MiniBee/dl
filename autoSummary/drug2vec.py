#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/4/9 下午10:22
#@Author  :hongyue pei 
#@FileName: drug2vec.py
#@Software: PyCharm

import re
import pandas as pd
from gensim.models import Word2Vec
import matplotlib.pyplot as plt


def load_data(path):
    df = pd.read_excel(path)
    df = df[df['visit_type'] == '住院']
    df['inn_name'] = df['inn_name'].apply(lambda x: x.replace('"', '').replace('[', '').replace(']', '').replace(' ', '').split(','))

    inn_list = df['inn_name'].tolist()
    return inn_list
    # print(inn_list[:10])


def drug_vec(path):
    drug = []
    vec = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            drug.append(line.split(' ')[0])
            vec.append((line.split(' ')[1], line.split(' ')[2]))
    return drug, vec


if __name__ == '__main__':
    path = '/home/peihongyue/project/python/dl/data/drug2vec.xlsx'
    inn_list = load_data(path)
    model = Word2Vec(inn_list, min_count=1, size=16, window=32)
    model.wv.save_word2vec_format('/home/peihongyue/project/python/dl/data/drug2vec.model')
    # model = Word2Vec()
    # model.wv.load_word2vec_format('/home/peihongyue/project/python/dl/data/drug2vec.model')
    print(model.most_similar_cosmul('门冬胰岛素'))
    # drug, vec = drug_vec('/home/peihongyue/project/python/dl/data/drug2vec.model')
    # print(vec[:10])

    # x = []
    # y = []
    # plt.figure(figsize=(20, 20))
    # for i, d in enumerate(drug[:50]):
    #     x = float(vec[i][0])
    #     y = float(vec[i][1])
    #     plt.scatter(x, y)
    #     plt.annotate(d, xy=[x, y])
    # plt.show()



