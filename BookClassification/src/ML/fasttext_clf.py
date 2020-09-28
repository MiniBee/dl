#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: fasttext.py
# @time: 2020/9/23 下午3:09
# @desc:


import pandas as pd 
from tqdm import tqdm
import fasttext
import jieba 

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
        stopWords = open(config.root_path + '/data/stopWords_cn.txt').readlines()
        jieba.load_userdict(config.root_path + '/data/ai100_words.txt')
        if model_path is None:
            self.train_raw_data = pd.read_csv(train_raw_path, ',', names=['label', 'text'])
            self.test_raw_data = pd.read_csv(test_raw_path, ',', names=['label', 'text'])
            # shuffle
            self.train_raw_data = self.train_raw_data.sample(frac=1).reset_index(drop=True)
            self.test_raw_data = self.test_raw_data.sample(frac=1).reset_index(drop=True)
            # ' ' split text
            self.train_raw_data['text'] = self.train_raw_data['text'].apply(lambda x: ' '.join([w for w in jieba.cut(x) if w not in stopWords]))
            self.test_raw_data['text'] = self.test_raw_data['text'].apply(lambda x: ' '.join([w for w in jieba.cut(x) if w not in stopWords]))
            self.data_process(self.train_raw_data, model_train_file)
            self.data_process(self.test_raw_data, model_test_file)
        elif model_path == '2':
            pass
        else:
            self.classifier = fasttext.load_model(model_path)

    def data_process(self, raw_data, model_data_file):
        with open(model_data_file, 'w') as f:
            for index, row in tqdm(raw_data.iterrows(), total=raw_data.shape[0]):
                outline = '__label__' + str(int(row['label'])) + '\t,\t' + str(row['text']) + '\n'
                f.write(outline)

    def train(self, model_train_file, model_test_file):
        self.classifier = fasttext.train_supervised(model_train_file, lr=1, label='__label__', dim=100, epoch=10)
        # self.test(model_train_file, model_test_file)
        self.classifier.save_model(config.root_path + '/model/fasttext.model')

    def test(self, model_train_file, model_test_file):
        # train_result = self.classifier.test(model_train_file)
        test_result = self.classifier.test(model_test_file)

        print(test_result)
        # print(train_result)


if __name__ == "__main__":
    content = Fasttext(model_path=config.root_path + '/model/fasttext.model')
    # print(config.root_path + '/data/fast_test.csv')
    # content.train(config.root_path + '/data/fast_train.csv', config.root_path + '/data/fast_test.csv')
    # content.test(config.root_path + '/data/fast_train.csv', config.root_path + '/data/fast_test.csv')
    res = content.classifier.predict('公司 以 绿色 、 环保 为 经营理念 ， 主要 从事 汽车 美容 养护 产品 的 研发 、 生产 和 销售 ， 目前 拥有 专利技术 4 项 ， 其中 发明专利 1 项 ， 另有 2 项 发明专利 已 进入 实质性 审查 阶段 。 公司 为 中国 标准化 协会 汽车用品 专业 委员会 的 汽车漆 面 镀膜 产品 标准 的 起草 单位 ， 掌握 了 汽车 镀膜 、 清洗 、 研磨 、 养护 等 产品 的 核心技术 或 配方 ， 经过 多年 的 积累 ， 公司 注册 了 优贝 、 博斯 威尔 等 商标 并 把 它们 发展 为 行业 的 知名品牌 ， 在 行业 内 树立 较 高 的 品牌 知名度 ， 产品 形象 深入 市场 ， 公司 连续 多年 被 慧聪网 等 专业 机构 评为 “ 汽车用品 十大 知名品牌 ” ')
    print(res)


