#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: fasttext.py
# @time: 2020/9/23 下午3:09
# @desc:


import pandas as pd 
from tqdm import tqdm
import fasttext

from __init__ import * 
from src.utils import config
from src.utils.tools import create_logger

logger = create_logger(config.root_path + '/logs/Fasttext.log')


class Fasttext(object):
    def __init__(self, train_raw_path=config.root_path + '/data/train.csv', 
                       test_raw_path=config.root_path + '/data/test.csv', 
                       model_train_file=config.root_path + '/data/fast_train.csv',
                       model_test_file=config.root_path + '/data/fast_test.csv',
                       model_path=None):
        if model_path is None:
            self.train_raw_data = pd.read_csv(train_raw_path, ',', names=['label', 'text'])
            self.test_raw_data = pd.read_csv(test_raw_path, ',', names=['label', 'text'])
            self.data_process(self.train_raw_data, model_train_file)
            self.data_process(self.test_raw_data, model_test_file)
        else:
            self.classifier = fasttext.load_model(model_path)

    def data_process(self, raw_data, model_data_file):
        with open(model_data_file, 'w') as f:
            for index, row in tqdm(raw_data.iterrows(), total=raw_data.shape[0]):
                outline = '__label__' + str(int(row['label'])) + ', \t' + str(row['text']) + '\n'
                f.write(outline)

    def train(self, model_train_file, model_test_file):
        self.classifier = fasttext.train_supervised(model_train_file, label='__label__', dim=100, epoch=100)
        self.test(model_train_file, model_test_file)
        self.classifier.save_model(config.root_path + '/model/fasttext.model')

    def test(self, model_train_file, model_test_file):
        train_result = self.classifier.test(model_train_file)
        test_result = self.classifier.test(model_test_file)

        print(test_result)
        # print(train_result[1], train_result[2])


if __name__ == "__main__":
    content = Fasttext()
    content.train(config.root_path + '/data/fast_train.csv', config.root_path + '/data/fast_test.csv')
    # content.test(config.root_path + '/data/fast_train.csv', config.root_path + '/data/fast_test.csv')




