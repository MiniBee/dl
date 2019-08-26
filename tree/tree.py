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
import lightgbm as lgb
import catboost as cb
import re
import os
import matplotlib.pyplot as plt


def load_data(path):
    columns = ['男', '女', '年龄', '*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶',
       '*碱性磷酸酶', '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白', '*球蛋白', '白球比例', '甘油三酯', '总胆固醇',
       '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '尿素', '肌酐', '尿酸', '白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积',
       '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度', '血小板计数', '血小板平均体积',
       '血小板体积分布宽度', '血小板比积', '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%']
    label_columns = ['血糖']

    data = pd.read_csv(path)
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
    columns_index = [54, 11, 15, 61, 18, 22, 45, 24, 46, 50, 43, 27, 52, 13, 35, 42, 5, 55, 2, 51, 53, 32, 16, 3, 56, 12, 28, 38, 31, 8, 33, 23, 30, 36, 14, 25, 58, 7, 17, 57, 21, 26, 39, 9, 37, 4, 19, 49, 47, 6, 29, 34, 48, 59, 44, 10, 0, 40, 20, 41, 1, 60]
    columns1 = [columns[i] for i in columns_index]
    print(columns1)
    return data[columns1], data[label_columns].values


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

    # rf = RandomForestRegressor(n_jobs=10)
    # rf_param = {'n_estimators': [140, 120, 130], "max_features": ["log2", "sqrt"]}
    # rf_grid = GridSearchCV(estimator=rf, param_grid=rf_param, cv=10)
    # rf_grid.fit(train_x, train_y)
    # rf = rf_grid.best_estimator_
    # pred_y1 = rf.predict(test_x)
    #
    # gbd = GradientBoostingRegressor()
    # gbd_param = {'n_estimators': [330, 340, 320], 'learning_rate': [0.11, 0.12, 0.07], 'max_depth': [1 ,2, 3]}
    # gbd_grid = GridSearchCV(estimator=gbd, param_grid=gbd_param, cv=10)
    # gbd_grid.fit(train_x, train_y)
    # gbd = gbd_grid.best_estimator_
    # pred_y2 = gbd.predict(test_x)
    #
    # xgb_param = {'max_depth': [2,1,3], 'learning_rate': [0.12, 0.14, 0.11], 'n_estimators': [100, 40, 50]}
    # xgb_grid = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror'), param_grid=xgb_param, cv=10)
    # xgb_grid.fit(train_x, train_y)
    # xgb = xgb_grid.best_estimator_
    # pred_y3 = xgb.predict(test_x)
    #
    # svc = SVR()
    # svc.fit(train_x, train_y)
    # pred_y4 = svc.predict(test_x)
    #
    # gbm = lgb.LGBMRegressor(objective='regression')
    # gbm_param = {'n_estimators': [145, 140], 'learning_rate': [0.02, 0.03]}
    # gbm_grid = GridSearchCV(estimator=gbm, param_grid=gbm_param, cv=10)
    # gbm_grid.fit(train_x, train_y)
    # gbm = gbm_grid.best_estimator_
    # '''
    # # booster = gbm.booster_
    # # importance = booster.feature_importance(importance_type='split')
    # # feature_name = booster.feature_name()
    # # ret = []
    # # for (feature_name, importance) in zip(feature_name, importance):
    # #     print (feature_name, importance)
    # #     ret.append([feature_name, importance])
    # # ret.sort(key=lambda x: x[1], reverse=True)
    # # columns_index = [int(i[0].split('_')[1]) for i in ret]
    # # print(columns_index)
    # # lgb.plot_importance(gbm, max_num_features=30)
    # # plt.title("Featurertances")
    # # plt.show()
    # '''
    # pred_y5 = gbm.predict(test_x)

    cbst = cb.CatBoostRegressor(iterations=2, depth=2, learning_rate=1, loss_function='RMSE')
    # cbst_param = {'learning_rate': [0.1, 0.05, 0.15], 'depth': [1,2,3], 'iterations': [20, 50, 100]}
    # cb_grid = GridSearchCV(estimator=cbst, param_grid=cbst_param, cv=10)
    train_pool = cb.Pool(train_x,
                      train_y,
                      cat_features=[1])
    test_pool = cb.Pool(test_x,
                     cat_features=[1])
    cbst.fit(train_pool)
    # cbst = cb_grid.best_estimator_
    pred_y6 = cbst.predict(test_pool)

    # print(rf_grid.best_params_)
    # print(gbd_grid.best_params_)
    # print(xgb_grid.best_params_)
    # print(gbm_grid.best_params_)
    print(cbst_grid.best_params_)

    # print(mean_squared_error(test_y, pred_y1))
    # print(mean_squared_error(test_y, pred_y2))
    # print(mean_squared_error(test_y, pred_y3))
    # print(mean_squared_error(test_y, pred_y4))
    # print(mean_squared_error(test_y, pred_y5))
    print(mean_squared_error(test_y, pred_y6))
    #
    print(mean_squared_error(test_y, (0.1*pred_y2 + 0*pred_y3 + 1*pred_y1 + 0.8*pred_y5)))
