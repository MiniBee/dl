#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: mlData.py
# @time: 2020/9/24 下午2:46
# @desc:


import numpy as np
import pandas as pd
import json
import os
from __init__ import *
from src.utils import config
from src.utils.tools import create_logger, wam, query_cut
from src.word2vec.embedding import Embedding
logger = create_logger(config.log_dir + '/data.log')

class MLData(object):
    def __init__(self, debug_mode=False, train_mode=True):
        self.debug_mode = debug_mode
        self.em = Embedding()
        self.em.load()
        if train_mode:
            self.preprocessor()

    def preprocessor(self):
        logger.info('load data ... ')
        self.train = pd.read_csv(config.root_path + '/data/train.csv', sep=',', names=['label', 'text']).dropna()
        self.dev = pd.read_csv(config.root_path + '/data/test.csv', sep=',', names=['label', 'text']).dropna()
        if self.debug_mode:
            self.train = self.train.sample(n=1000).reset_index(drop=True)
            self.dev = self.dev.sample(n=100).reset_index(drop=True)
        self.train['queryCut'] = self.train['text'].apply(query_cut)
        self.dev['queryCut'] = self.dev['text'].apply(query_cut)
        # 去掉停用词
        self.train['queryCutRMStopWord'] = self.train['queryCut'].apply(lambda x: [word for word in x if word not in self.em.stopWords])
        self.dev['queryCutRMStopWord'] = self.dev['queryCut'].apply(lambda x: [word for word in x if word not in self.em.stopWords])
        if os.path.exists(config.root_path + '/data/label2id.json'):
            labelNameToIndex = json.load(open(config.root_path + '/data/label2id.json', encoding='utf-8'))
        else:
            labelName = self.train['label'].unique()
            labelIndex = list(range(len(labelName)))
            labelNameToIndex = dict(zip(labelName, labelIndex))
            with open(config.root_path + '/data/label2id.json', 'w', encoding='utf-8') as f:
                json.dump({k: v for k, v in labelNameToIndex.items()}, f)
        self.train['labelIndex'] = self.train['label'].map(labelNameToIndex)
        self.dev['labelIndex'] = self.dev['label'].map(labelNameToIndex)

    def process_data(self, method='word2vec'):
        x_train = self.get_features(self.train, method)
        x_test = self.get_features(self.dev, method)
        y_train = self.train['labelIndex']
        y_test = self.dev['labelIndex']
        return x_train, x_test, y_train, y_test

    def get_features(self, data, method='word2vec'):
        if method == 'tfidf':
            data = [' '.join(query) for query in data['queryCutRMStopWord']]
            return self.em.tfidf.transform(data)
        elif method == 'word2vec':
            return np.vstack(data['queryCutRMStopWord'].apply(lambda x: wam(x, self.em.w2v)[0]))
        elif method == 'fasttext':
            return np.vstack(data['queryCutRMStopWord'].apply(lambda x: wam(x, self.em.fast)[0]))
        else:
            NotImplementedError

if __name__ == '__main__':
    logger.info('hello, world')
    mlData = MLData()





