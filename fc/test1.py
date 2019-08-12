#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019/7/18 下午9:39
#@Author  :hongyue pei 
#@FileName: hpd.py
#@Software: PyCharm

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import fc_net
import os
import re

def load_data(path):
    columns = ['男', '女', '年龄', '*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶',
       '*碱性磷酸酶', '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白', '*球蛋白', '白球比例', '甘油三酯', '总胆固醇',
       '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '尿素', '肌酐', '尿酸', '白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积',
       '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度', '血小板计数', '血小板平均体积',
       '血小板体积分布宽度', '血小板比积', '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%',
       ]
    columns = ['男', '女', '年龄', '*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶',
       '*碱性磷酸酶', '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白', '*球蛋白', '白球比例', '甘油三酯', '总胆固醇',
       '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '尿素', '肌酐', '尿酸', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原',
       '乙肝e抗体', '乙肝核心抗体', '白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积',
       '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度', '血小板计数', '血小板平均体积',
       '血小板体积分布宽度', '血小板比积', '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%']
    label_columns = ['血糖']
    data = pd.read_csv(path)
    print(data.columns)
    # data = data.fillna(0)
    a = pd.get_dummies(data['性别'])
    data = pd.concat([data, a], axis=1)
    data.drop(['性别'], axis=1, inplace=True)
    for col in columns:
        if data[col].isnull().any():
            if re.search('乙肝', col, re.IGNORECASE):
                data[col].fillna(999, inplace=True)
            else:
                data[col].fillna(data[col].median(), inplace=True)

    data['总酶'] = data['*天门冬氨酸氨基转换酶'] + data['*丙氨酸氨基转换酶'] + data['*碱性磷酸酶'] + data['*r-谷氨酰基转换酶']

    data['*天门冬氨酸氨基转换酶ratio'] = data['*天门冬氨酸氨基转换酶'] / np.maximum(data["总酶"].astype("float"), 1)
    data['*天门冬氨酸氨基转换酶ratio'].loc[data['*天门冬氨酸氨基转换酶ratio'] < 0] = 0
    data['*天门冬氨酸氨基转换酶ratio'].loc[data['*天门冬氨酸氨基转换酶ratio'] > 1] = 1

    data['*丙氨酸氨基转换酶ratio'] = data['*丙氨酸氨基转换酶'] / np.maximum(data["总酶"].astype("float"), 1)
    data['*丙氨酸氨基转换酶ratio'].loc[data['*丙氨酸氨基转换酶ratio'] < 0] = 0
    data['*丙氨酸氨基转换酶ratio'].loc[data['*丙氨酸氨基转换酶ratio'] > 1] = 1

    data['*碱性磷酸酶ratio'] = data['*碱性磷酸酶'] / np.maximum(data["总酶"].astype("float"), 1)
    data['*碱性磷酸酶ratio'].loc[data['*碱性磷酸酶ratio'] < 0] = 0
    data['*碱性磷酸酶ratio'].loc[data['*碱性磷酸酶ratio'] > 1] = 1

    data['*r-谷氨酰基转换酶ratio'] = data['*r-谷氨酰基转换酶'] / np.maximum(data["总酶"].astype("float"), 1)
    data['*r-谷氨酰基转换酶ratio'].loc[data['*r-谷氨酰基转换酶ratio'] < 0] = 0
    data['*r-谷氨酰基转换酶ratio'].loc[data['*r-谷氨酰基转换酶ratio'] > 1] = 1

    data['白蛋白ratio'] = data['白蛋白'] / np.maximum(data["*总蛋白"].astype("float"), 1)
    data['白蛋白ratio'].loc[data['白蛋白ratio'] < 0] = 0
    data['白蛋白ratio'].loc[data['白蛋白ratio'] > 1] = 1

    data['*球蛋白ratio'] = data['*球蛋白'] / np.maximum(data["*总蛋白"].astype("float"), 1)
    data['*球蛋白ratio'].loc[data['*球蛋白ratio'] < 0] = 0
    data['*球蛋白ratio'].loc[data['*球蛋白ratio'] > 1] = 1

    data['高密度脂蛋白胆固醇ratio'] = data['高密度脂蛋白胆固醇'] / np.maximum(data["总胆固醇"].astype("float"), 1)
    data['高密度脂蛋白胆固醇ratio'].loc[data['高密度脂蛋白胆固醇ratio'] < 0] = 0
    data['高密度脂蛋白胆固醇ratio'].loc[data['高密度脂蛋白胆固醇ratio'] > 1] = 1

    data['低密度脂蛋白胆固醇ratio'] = data['低密度脂蛋白胆固醇'] / np.maximum(data["总胆固醇"].astype("float"), 1)
    data['低密度脂蛋白胆固醇ratio'].loc[data['低密度脂蛋白胆固醇ratio'] < 0] = 0
    data['低密度脂蛋白胆固醇ratio'].loc[data['低密度脂蛋白胆固醇ratio'] > 1] = 1

    data['null_count'] = data.isnull().sum(axis=1)

    data['*r-谷氨酰基转换酶-尿酸'] = data['*r-谷氨酰基转换酶'] - data['尿酸']
    data['*r-谷氨酰基转换酶*年龄'] = data['*r-谷氨酰基转换酶'] * data['年龄']
    data['*r-谷氨酰基转换酶*总胆固醇'] = data['*r-谷氨酰基转换酶'] * data['总胆固醇']

    data['*丙氨酸氨基转换酶**天门冬氨酸氨基转换酶'] = data['*丙氨酸氨基转换酶'] * data['*天门冬氨酸氨基转换酶']
    data['*丙氨酸氨基转换酶+*天门冬氨酸氨基转换酶'] = data['*丙氨酸氨基转换酶'] + data['*天门冬氨酸氨基转换酶']
    data['*丙氨酸氨基转换酶/*天门冬氨酸氨基转换酶'] = data['*丙氨酸氨基转换酶'] / data['*天门冬氨酸氨基转换酶']

    data['*天门冬氨酸氨基转换酶/*总蛋白'] = data['*天门冬氨酸氨基转换酶'] / data['*总蛋白']
    data['*天门冬氨酸氨基转换酶-*球蛋白'] = data['*天门冬氨酸氨基转换酶'] - data['*球蛋白']

    data['*球蛋白/甘油三酯'] = data['*球蛋白'] / data['甘油三酯']

    data['年龄*红细胞计数/红细胞体积分布宽度-红细胞计数'] = data['年龄'] * data['红细胞计数'] / (data['红细胞体积分布宽度'] - data['红细胞计数'])

    data['尿酸/肌酐'] = data['尿酸'] / data['肌酐']

    data['肾'] = data['尿素'] + data['肌酐'] + data['尿酸']

    data['红细胞计数*红细胞平均血红蛋白量'] = data['红细胞计数'] * data['红细胞平均血红蛋白量']
    data['红细胞计数*红细胞平均血红蛋白浓度'] = data['红细胞计数'] * data['红细胞平均血红蛋白浓度']
    data['红细胞计数*红细胞平均体积'] = data['红细胞计数'] * data['红细胞平均体积']

    data['嗜酸细胞'] = data['嗜酸细胞%'] * 100

    data['年龄*中性粒细胞%/尿酸*血小板比积'] = data['年龄'] * data['中性粒细胞%'] / (data['尿酸'] * data['血小板比积'])

    columns.extend(['总酶', '*天门冬氨酸氨基转换酶ratio', '*丙氨酸氨基转换酶ratio', '*碱性磷酸酶ratio',
       '*r-谷氨酰基转换酶ratio', '白蛋白ratio', '*球蛋白ratio', '高密度脂蛋白胆固醇ratio',
       '低密度脂蛋白胆固醇ratio', 'null_count', '*r-谷氨酰基转换酶-尿酸', '*r-谷氨酰基转换酶*年龄',
       '*r-谷氨酰基转换酶*总胆固醇', '*丙氨酸氨基转换酶**天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶+*天门冬氨酸氨基转换酶',
       '*丙氨酸氨基转换酶/*天门冬氨酸氨基转换酶', '*天门冬氨酸氨基转换酶/*总蛋白', '*天门冬氨酸氨基转换酶-*球蛋白',
       '*球蛋白/甘油三酯', '年龄*红细胞计数/红细胞体积分布宽度-红细胞计数', '尿酸/肌酐', '肾',
       '红细胞计数*红细胞平均血红蛋白量', '红细胞计数*红细胞平均血红蛋白浓度', '红细胞计数*红细胞平均体积', '嗜酸细胞',
       '年龄*中性粒细胞%/尿酸*血小板比积'])
    return data[columns], data[label_columns].values

class Dataset(object):
    def __init__(self, x, y):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = x.shape[0]
        self.x = x
        self.y = y

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if start == 0 and self._index_in_epoch == 0:
            idx = np.arange(0, self._num_examples)
            np.random.shuffle(idx)
            self.x = self.x[idx]
            self.y = self.y[idx]
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            x_rest_part = self.x[start:self._num_examples]
            y_rest_part = self.y[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)
            np.random.shuffle(idx0)
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            x_new_part = self.x[start:self._index_in_epoch]
            y_new_part = self.y[start:self._index_in_epoch]
            return np.concatenate((x_rest_part, x_new_part), axis=0), np.concatenate((y_rest_part, y_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.x[start:end], self.y[start:end]


if __name__ == '__main__':
    train_model = True
    train_path = '/home/peihongyue/project/python/dl/data/hpd/train_data.csv'
    if not os.path.exists(train_path):
        train_path = '/Users/peihongyue/phy/project/ai/dl/data/train_data.csv'
    test_path = '/home/peihongyue/project/python/dl/data/hpd/test_data.csv'
    if not os.path.exists(test_path):
        test_path = '/Users/peihongyue/phy/project/ai/dl/data/test_data.csv'
    train_x, train_y = load_data(train_path)
    standardScaler = StandardScaler()
    train_x = standardScaler.fit_transform(train_x)
    test_x, test_y = load_data(test_path)
    test_x = standardScaler.transform(test_x)
    # train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.3)
    train_x = train_x.reshape(len(train_x), len(train_x[0]), 1)
    test_x = test_x.reshape(len(test_x), len(test_x[0]), 1)
    print(train_x.shape)
    data = Dataset(train_x, train_y)
    test_data = Dataset(test_x, test_y)
    test_x, test_y = test_data.next_batch(len(test_x))
    model = fc_net.Model(len(train_x[0]), 0.00005)
    best_loss = 999999.0
    if train_model:
        for i in range(40000):
            batch_x, batch_y = data.next_batch(512)
            model.sess.run(model.step, feed_dict={model.inputs: batch_x, model.y: batch_y})
            loss_train = model.sess.run(model.loss, feed_dict={model.inputs: batch_x, model.y: batch_y})
            loss_test = model.sess.run(model.loss, feed_dict={model.inputs: test_x, model.y: test_y})
            if i % 100 == 0:
                print('loss_test ' + str(i) + ': ', loss_test)
                print('loss_train ---------> ' + str(i) + ': ', loss_train)
            if best_loss > loss_test:
                best_loss = loss_test
    print(best_loss)


