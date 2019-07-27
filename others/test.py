#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: test.py
# @time: 2019/7/22 上午10:20
# @desc:


import ant
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os


def load_data(path):
    columns = ['男', '女', '年龄', '*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶',
       '*碱性磷酸酶', '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白', '*球蛋白', '白球比例', '甘油三酯', '总胆固醇',
       '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '尿素', '肌酐', '尿酸',
               '白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积',
       '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度', '血小板计数', '血小板平均体积',
       '血小板体积分布宽度', '血小板比积', '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%',
       ]
    label_columns = ['血糖']
    data = pd.read_csv(path)
    # data = data.fillna(9999)
    for col in columns:
        if data[col].isnull().any():
            data[col].fillna(data[col].median())
    a = pd.get_dummies(data['性别'])
    data.drop(['性别'], axis=1, inplace=True)
    data = pd.concat([data, a], axis=1)
    return data[columns].values, data[label_columns].values


if __name__ == '__main__':
    train_path = '/home/peihongyue/project/python/dl/data/hpd/train_data.csv'
    if not os.path.exists(train_path):
        train_path = '/Users/peihongyue/phy/project/ai/dl/data/train_data.csv'
    test_path = '/home/peihongyue/project/python/dl/data/hpd/test_data.csv'
    if not os.path.exists(test_path):
        test_path = '/Users/peihongyue/phy/project/ai/dl/data/test_data.csv'
    train_x, train_y = load_data(train_path)
    train_x = StandardScaler().fit_transform(train_x)
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.3)
    agt = ant.Auto_Grow_Tree(train_x, test_x, train_y, test_y, num_n=1024, categories=1, solver='xgboost_reg')
    print(agt.k_mean_list)
    agt.auto_grow()
