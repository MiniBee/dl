#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/3/16 下午10:41
#@Author  :hongyue pei 
#@FileName: train.py
#@Software: PyCharm

import transformer


def load_data(path):
    x_train = []
    y_train = []
    with open(path) as f:
        i = f.readline().split(',')
        i = [int(a.strip()) for a in i]
        x_train.append(i[1:])
        y_train.append(i[0])
    return x_train, y_train

if __name__ == '__main__':
    x_train, y_train = load_data('/home/peihongyue/project/python/dl/data/bc/trainVec')
    print(x_train[:20])
