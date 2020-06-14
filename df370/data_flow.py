#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: data_flow.py
# @time: 2020/6/2 下午3:34
# @desc:

import sys
import pandas as pd
from nlp_util import word2idx


sys.path.append('/home/peihongyue/project/python/dl/')
sys.path.append('/Users/peihongyue/phy/project/dl')


if __name__ == '__main__':
    file_path = '/data/phy/datas/df370/'
    train = pd.read_csv(file_path + 'train.csv', sep='\t')[['comment', 'label']]
    comment_list = train['comment'].tolist()
    label_list = train['label'].tolist()
    content_word_list = word2idx.tokens(comment_list, save_path=file_path + 'tokens.csv')
    word_idx = word2idx.word_set(content_word_list, save_path=file_path + 'idx.csv')
    content_idx = word2idx.word2idx(content_word_list, word_idx, label_list=label_list, save_path=file_path + 'widx.csv')






