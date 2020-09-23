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

logger = create_logger(config.log_path + '/logs/Fasttext.log')


class Fasttext(object):
    def __init__(self, train_raw_path=config.root_path + '/data/train_clean.tsv', 
                       test_raw_path=config.root_path + '/data/test_clean.tsv', 
                       model_train_file=config.root_path + '/data/fast_train.csv',
                       model_test_file=config.root_path + '/data/fast_test.csv',
                       model_path=None):
        if model_path is None:
            self.train_raw_data = pd.read_csv(train_raw_path, '\t')
            self.test_raw_data = pd.read_csv(test_raw_path, '\t')
            self.data_process(self.train_raw_data, model_train_file)
            self.data_process(self.test_raw_data, model_test_file)
        else:
            self.fast = fasttext.load_model(model_path)

    def data_process(self, raw_data, model_data_file):
        with open(model_data_file, 'w') as f:
            for index, row in tqdm(raw_data.iterrows(), total=raw_data.shape[0]):
                outline = row['text'] + '\t__label__' + str(int(row['category_id'])) + '\n'
                f.write(outline)

    def train(self, model_train_file, model_test_file):
        self.classifier = fasttext.train_supervised(model_train_file, label='__label__', dim=50, epoch=5)
        self.test(model_train_file, model_test_file)
        self.classifier.save_model(config.root_path + '/model/fasttext.model')

    def test(self, model_train_file, model_test_file):
        train_result = self.classifier.test(model_train_file)
        test_result = self.classifier.test(model_test_file)

        print(test_result[1], test_result[2])
        print(train_result[1], train_result[2])


if __name__ == "__main__":
    content = Fasttext()




