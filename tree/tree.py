#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019/8/5 下午10:01
#@Author  :hongyue pei 
#@FileName: tree.py
#@Software: PyCharm

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import xgboost as xgb
import os


def load_data(path):
    columns = ['男', '女', '年龄', '*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶',
       '*碱性磷酸酶', '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白', '*球蛋白', '白球比例', '甘油三酯', '总胆固醇',
       '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '尿素', '肌酐', '尿酸', '白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积',
       '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度', '血小板计数', '血小板平均体积',
       '血小板体积分布宽度', '血小板比积', '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%',
       ]
    label_columns = ['血糖']
    data = pd.read_csv(path)
    # data = data.fillna(0)
    a = pd.get_dummies(data['性别'])
    data = pd.concat([data, a], axis=1)
    data.drop(['性别'], axis=1, inplace=True)
    for col in columns:
        if data[col].isnull().any():
            data[col].fillna(data[col].median(), inplace=True)

    return data[columns], data[label_columns].values


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

    rf = RandomForestRegressor(n_jobs=10)
    rf_param = {'n_estimators': [70, 80, 100, 150]}
    rf_grid = GridSearchCV(estimator=rf, param_grid=rf_param, cv=5)
    rf_grid.fit(train_x, train_y)
    rf = rf_grid.best_estimator_
    pred_y1 = rf.predict(test_x)


    gbd = GradientBoostingRegressor()
    gbd_param = {'n_estimators': [100, 150, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [1,2]}
    gbd_grid = GridSearchCV(estimator=gbd, param_grid=gbd_param, cv=5)
    gbd_grid.fit(train_x, train_y)
    gbd = gbd_grid.best_estimator_
    pred_y2 = gbd.predict(test_x)

    xgb_param = {'max_depth': [1,2,3], 'learning_rate': [0.05, 0.1], 'n_estimators': [100, 150, 200]}
    xgb_grid = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror'), param_grid=xgb_param, cv=5)
    xgb_grid.fit(train_x, train_y)
    xgb = xgb_grid.best_estimator_
    pred_y3 = xgb.predict(test_x)

    svc = SVR()
    svc.fit(train_x, train_y)
    pred_y4 = svc.predict(test_x)

    print(rf_grid.best_params_)
    print(gbd_grid.best_params_)
    print(xgb_grid.best_params_)

    print(mean_squared_error(test_y, pred_y1))
    print(mean_squared_error(test_y, pred_y2))
    print(mean_squared_error(test_y, pred_y3))
    print(mean_squared_error(test_y, pred_y4))

    print(mean_squared_error(test_y, (0.7*pred_y2 + 0.1*pred_y3 + 0.2*pred_y4)))
