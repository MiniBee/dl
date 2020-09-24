#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: models.py
# @time: 2020/9/23 下午3:09
# @desc:

import os 
import lightgbm as lgb 
import numpy as np 
import torchvision
import json
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids

from sklearn.ensemble import RandomForestClassifier
import joblib
from transformers import BertModel, BertTokenizer

from __init__ import * 
from src.utils.tools import create_logger, bayes_parameter_opt_lgb
from src.data.mlData import MLData
from src.utils import config
from src.utils.feature import get_embedding_feature, get_autoencoder_feature, get_score

logger = create_logger(config.log_path + 'model.log')

class Model(object):
    def __init__(self, model_path=None, feature_engineer=False, train_mode=True):
        # self.res_model = torchvision.models.resnet152(pretrained=True)
        # self.res_model = self.res_model.to(config.device)

        self.ml_data = MLData(debug_mode=True, train_mode=train_mode)
        if train_mode:
            self.model = lgb.LGBMClassifier(objective='multiclass', num_class=33, seed=11)
        else:
            self.load(model_path)
            labelNameToIndex = json.load(open(config.root_path + '/data/label2id.json', encoding='utf-8'))
            self.id2label = {v: k for k, v in labelNameToIndex.items()}

    def feature_engineer(self):
        logger.info('generate embedding feature .. ')
        train_tfidf, train = get_embedding_feature(self.ml_data.train, self.ml_data.em.tfidf, self.ml_data.em.w2v)
        test_tfidf, test = get_embedding_feature(self.ml_data.test, self.ml_data.em.tfidf, self.ml_data.em.w2v)
        train = formate_data(train, train_tfidf, train_ae)
        test = formate_data(test, test_tfidf, test_ae)
        #  生成训练，测试的数据
        cols = [x for x in train.columns if str(x) not in ['labelIndex']]
        X_train = train[cols]
        X_test = test[cols]
        train["labelIndex"] = train["labelIndex"].astype(int)
        test["labelIndex"] = test["labelIndex"].astype(int)
        y_train = train["labelIndex"]
        y_test = test["labelIndex"]
        return X_train, X_test, y_train, y_test

    def param_search(self, search_method='grid'):
        if search_method == 'grid':
            logger.info('use grid search ... ')
            self.model = grid_train_model(self.model, self.x_train, self.x_test, self.y_train, self.y_test)
        elif search_method == 'bayesian':
            logger.info('use bayesian optimization ... ')
            trn_data = lgb.Dataset(data=self.x_train, label=self.y_train, free_raw_data=False)
            param = bayes_parameter_opt_lgb(trn_data)
            logger.info('best param', param)
            return param

    def unbanlance_helper(self, imbalance_method='under_sampling', search_method='grid'):
        logger.info('get all feature ... ')
        self.x_train, self.x_test, self.y_train, self.y_test = self.feature_engineer()
        model_name = None
        if imbalance_method == 'over_sampling':
            logger.info('Use SMOTE deal with unbalance data ... ')
            self.x_train, self.y_train = SMOTE().fit_resample(self.x_train, self.y_train)
            self.x_test, self.y_test = SMOTE().fit_resample(self.x_test, self.y_test)
            model_name = 'lgb_over_sampling'
        elif imbalance_method == 'under_sampling':
            logger.info('User ClusterCentroids deal with unbalance data ... ')
            self.x_train, self.y_train = ClusterCentroids(random_state=11).fit_resample(self.x_train, self.y_train)
            self.x_test, self.y_test = ClusterCentroids(random_state=11).fit_resample(self.x_test, self.y_test)
            model_name = 'lgb_under_sampling'

        logger.info('search best param ... ')
        param = {}
        param['params'] = {}
        param['params']['num_leaves'] = 3
        param['params']['max_depth'] = 5
        self.model = self.model.set_params(**param['params'])
        logger.info('fit model ... ')
        self.model.fit(self.x_train, self.y_train)
        test_predict_label = self.model.predict(self.x_test)
        train_predict_label = self.model.predict(self.x_train)
        per, acc, recall, f1 = get_score(self.y_train, self.y_test, train_predict_label, test_predict_label)

        logger.info('Train accuracy %s' % per)
        # 输出测试集的准确率
        logger.info('test accuracy %s' % acc)
        # 输出recall
        logger.info('test recall %s' % recall)
        # 输出F1-score
        logger.info('test F1_score %s' % f1)
        self.save(model_name)

    def save(self, model_name):
        joblib.dump(self.model, config.root_path + '/model/ml_model/' + model_name)

    def load(self, model_path):
        return joblib.load(model_path)





