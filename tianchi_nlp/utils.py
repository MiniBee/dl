#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: utils.py
# @time: 2020/7/24 下午6:48
# @desc:


import os
import numpy as np

import get_config


def load_data(path):
    y_array = []
    x_array = []
    with open(path) as f:
        f.readline()
        for line in f:
            line = line.split('\t')
            y_array.append(int(line[0]))
            x_array.append([int(i) for i in line[1].split(' ')])
    return np.array(x_array), y_array


if __name__ == '__main__':
    p_config = get_config.get_config()
    train_file = p_config['train_data']
    x_array, y_array = load_data(train_file)
    vocab_set = set()
    for x in x_array:
        vocab_set = vocab_set | set(x)
    print(len(vocab_set))







