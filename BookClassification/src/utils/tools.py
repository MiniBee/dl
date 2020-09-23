#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: tools.py
# @time: 2020/9/21 下午9:21
# @desc:

import logging
from logging import handlers

from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import jieba
from datetime import timedelta
import time
import re
import numpy as np

import tensorflow as tf
import torch


def get_score(train_label, test_label, train_predict_label, test_predict_label):
    return metrics.accuracy_score(train_label, train_predict_label), \
           metrics.accuracy_score(test_label, test_predict_label), \
           metrics.recall_score(test_label, test_predict_label), \
           metrics.f1_score(test_label, test_predict_label)


def query_cut(query):
    return list(jieba.cut(query))


def get_time_dif(start_time):
    end_time = time.time()
    time_diff = end_time - start_time
    return timedelta(seconds=int(round(time_diff)))


def create_logger(log_path):
    level_relations={
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }
    logger = logging.getLogger(log_path)
    fmt = '%(asctime)s - %(pathname)s[Line:%(lineno)d] - %(levelname)s: %(message)s'
    format_str = logging.Formatter(fmt)
    logger.setLevel(level_relations['info'])
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    th = handlers.TimedRotatingFileHandler(filename=log_path, when='D', backupCount=3, encoding='utf-8')
    th.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(th)
    return logger


def clean_str(string):
    string = re.sub(r'\s+', '', string)
    string = re.sub(r"[^\u4e00-\u9fa5^.^,^!^?^:^;^、^a-z^A-Z^0-9]", "", string)
    return string.strip()


def strQ2B(ustring):
    # 全角 半角 符号 转换
    ss = []
    for s in ustring:
        rstring = ''
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)


def wam(sentence, w2v_model, method='mean', aggregate=True):
    arr = np.array([w2v_model.wv.get_vector(s) for s in sentence if s in w2v_model.wv.vocab.keys()])
    if not aggregate:
        return arr
    if len(arr) > 0:
        if method == 'mean':
            return np.mean(np.array(arr), axis=0)
        elif method == 'max':
            return np.max(np.array(arr), axis=0)
        else:
            raise NotImplemented
    else:
        return np.zeros(300)


def padding(indice, max_length, pad_idx=0):
    pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
    return torch.tensor(pad_indice)


def grid_train_model(model, train_features, train_label, test_features, test_label):
    parameters = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05],
        'n_estimators': [1000, 2000],
        'subsample': [0.6, 0.75, 0.9],
        'colsample_bytree': [0.6, 0.75, 0.9],
        'reg_alpha': [5, 10],
        'reg_lambda': [10, 30, 50]
    }
    gsearch = GridSearchCV(model, param_grid=parameters, scoring='accuracy', vc=3, verbose=True)
    gsearch.fit(train_features, train_label)
    print('Best parameters set found on development set: {}'.format(gsearch.best_params_))
    return gsearch


def bayes_parameter_opt_lgb(trn_data, init_round=3, opt_round=5, n_folds=5, random_seed=6, n_estimators=10000, learning_rate=0.05):

    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):
        params = {
            'application': 'multiclass',
            'num_iterations': n_estimators,
            'learning_rate': learning_rate,
            'early_stopping_round': 100,
            'num_class': len([x.strip() for x in open('').readlines()]),
            'metric': 'multi_logloss'
        }


def format_data(data, max_features, max_len, tokenizer=None, shuffle=False):
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)
    data['text'] = data['text'].apply(lambda x: ' '.join(x))

    X = data['text']
    if not tokenizer:
        filters = "\"#$%&()*+./<=>@[\\]^_`{|}~\t\n"
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features, filters=filters)
        tokenizer.fit_on_texts(list(X))

    X = tokenizer.texts_to_sequences(X)
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_len)
    return X, tokenizer












