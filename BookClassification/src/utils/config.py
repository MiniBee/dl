#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: config.py
# @time: 2020/9/21 下午9:15
# @desc:

import torch
import os
import numpy as np

curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(curPath)[0])[0]

train_file = root_path + '/'
dev_file = root_path + '/'
test_file = root_path + '/'
stopWords_file = root_path + '/data/stopwords.txt'
log_dir = root_path + '/logs/'

vocab_size = 1000
embedding_dim = 1024
max_len = 200


if __name__ == '__main__':
    print(curPath)
    print(root_path)




