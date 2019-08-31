#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: test2.py
# @time: 2019/8/31 上午10:17
# @desc:

import pandas as pd
import numpy as np
import seaborn as sns
from pylab import mpl
import lightgbm as lgb
import xgboost as xgb
from sklearn import preprocessing
from sklearn import svm
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import warnings
warnings.filterwarnings('ignore')

def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label,pred)*0.5
    return ('0.5mse',score,False)

data_path = 'datalab/3964/'
df_train = pd.read_csv(data_path+'d_train_20180102w.csv',encoding='gbk',parse_dates=['体检日期'])
df_test_A = pd.read_csv(data_path+'d_test_A_20180102w.csv',encoding='gbk',parse_dates=['体检日期'])
df_test_B = pd.read_csv(data_path+'d_test_B_20180128w.csv',encoding='gbk',parse_dates=['体检日期'])
df_test_A_answer = pd.read_csv(data_path+'d_answer_a_20180128.csv',header=-1)
df_test_A['血糖'] = df_test_A_answer[0]
data = pd.concat([df_train,df_test_A,df_test_B])
data = data.reset_index(drop=True)


data["yearmonth"] = data['体检日期'].dt.year*100 + data['体检日期'].dt.month
data["yearweek"] = data['体检日期'].dt.year*100 + data['体检日期'].dt.weekofyear
data["month_of_year"] = data['体检日期'].dt.month
data["week_of_year"] = data['体检日期'].dt.weekofyear
data["day_of_week"] =  data['体检日期'].dt.weekday

data['总酶'] = data['*天门冬氨酸氨基转换酶']+data['*丙氨酸氨基转换酶']+data['*碱性磷酸酶']+data['*r-谷氨酰基转换酶']

data['*天门冬氨酸氨基转换酶ratio'] = data['*天门冬氨酸氨基转换酶']/np.maximum(data["总酶"].astype("float"),1)
data['*天门冬氨酸氨基转换酶ratio'].loc[data['*天门冬氨酸氨基转换酶ratio']<0]=0
data['*天门冬氨酸氨基转换酶ratio'].loc[data['*天门冬氨酸氨基转换酶ratio']>1]=1

data['*丙氨酸氨基转换酶ratio'] = data['*丙氨酸氨基转换酶']/np.maximum(data["总酶"].astype("float"),1)
data['*丙氨酸氨基转换酶ratio'].loc[data['*丙氨酸氨基转换酶ratio']<0]=0
data['*丙氨酸氨基转换酶ratio'].loc[data['*丙氨酸氨基转换酶ratio']>1]=1

data['*碱性磷酸酶ratio'] = data['*碱性磷酸酶']/np.maximum(data["总酶"].astype("float"),1)
data['*碱性磷酸酶ratio'].loc[data['*碱性磷酸酶ratio']<0]=0
data['*碱性磷酸酶ratio'].loc[data['*碱性磷酸酶ratio']>1]=1

data['*r-谷氨酰基转换酶ratio'] = data['*r-谷氨酰基转换酶']/np.maximum(data["总酶"].astype("float"),1)
data['*r-谷氨酰基转换酶ratio'].loc[data['*r-谷氨酰基转换酶ratio']<0]=0
data['*r-谷氨酰基转换酶ratio'].loc[data['*r-谷氨酰基转换酶ratio']>1]=1

data['白蛋白ratio'] = data['白蛋白']/np.maximum(data["*总蛋白"].astype("float"),1)
data['白蛋白ratio'].loc[data['白蛋白ratio']<0]=0
data['白蛋白ratio'].loc[data['白蛋白ratio']>1]=1

data['*球蛋白ratio'] = data['*球蛋白']/np.maximum(data["*总蛋白"].astype("float"),1)
data['*球蛋白ratio'].loc[data['*球蛋白ratio']<0]=0
data['*球蛋白ratio'].loc[data['*球蛋白ratio']>1]=1

data['高密度脂蛋白胆固醇ratio'] = data['高密度脂蛋白胆固醇']/np.maximum(data["总胆固醇"].astype("float"),1)
data['高密度脂蛋白胆固醇ratio'].loc[data['高密度脂蛋白胆固醇ratio']<0]=0
data['高密度脂蛋白胆固醇ratio'].loc[data['高密度脂蛋白胆固醇ratio']>1]=1

data['低密度脂蛋白胆固醇ratio'] = data['低密度脂蛋白胆固醇']/np.maximum(data["总胆固醇"].astype("float"),1)
data['低密度脂蛋白胆固醇ratio'].loc[data['低密度脂蛋白胆固醇ratio']<0]=0
data['低密度脂蛋白胆固醇ratio'].loc[data['低密度脂蛋白胆固醇ratio']>1]=1

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

data['年龄*红细胞计数/红细胞体积分布宽度-红细胞计数'] = data['年龄'] * data['红细胞计数'] / (data['红细胞体积分布宽度']-data['红细胞计数'])

data['尿酸/肌酐'] = data['尿酸'] / data['肌酐']

data['肾'] = data['尿素'] + data['肌酐'] + data['尿酸']

data['红细胞计数*红细胞平均血红蛋白量'] = data['红细胞计数'] * data['红细胞平均血红蛋白量']
data['红细胞计数*红细胞平均血红蛋白浓度'] = data['红细胞计数'] * data['红细胞平均血红蛋白浓度']
data['红细胞计数*红细胞平均体积'] = data['红细胞计数'] * data['红细胞平均体积']

data['嗜酸细胞'] = data['嗜酸细胞%'] * 100

data['年龄*中性粒细胞%/尿酸*血小板比积'] = data['年龄'] * data['中性粒细胞%']/ (data['尿酸']*data['血小板比积'])

predictors1 = ['年龄',
               '性别',
               '高密度脂蛋白胆固醇',
               '甘油三酯',
               '尿素',
               '低密度脂蛋白胆固醇',
               '*天门冬氨酸氨基转换酶',
               '*丙氨酸氨基转换酶',
               '*r-谷氨酰基转换酶',
               '*碱性磷酸酶',
               '尿酸',
               '中性粒细胞%',
               '红细胞体积分布宽度',
               '红细胞平均体积',
               '红细胞平均血红蛋白浓度',
               '红细胞平均血红蛋白量',
               '红细胞计数',
               '血小板体积分布宽度',
               '血小板比积',
               'yearweek',
               'week_of_year',
               'day_of_week',
               '*天门冬氨酸氨基转换酶ratio',
               '*碱性磷酸酶ratio',
               '*r-谷氨酰基转换酶-尿酸',
               '*r-谷氨酰基转换酶*年龄',
               '*r-谷氨酰基转换酶*总胆固醇',
               '*丙氨酸氨基转换酶**天门冬氨酸氨基转换酶'
	, '*丙氨酸氨基转换酶+*天门冬氨酸氨基转换酶'
	, '*丙氨酸氨基转换酶/*天门冬氨酸氨基转换酶'
	, '*天门冬氨酸氨基转换酶/*总蛋白'
	, '*天门冬氨酸氨基转换酶-*球蛋白'
	, '*球蛋白/甘油三酯'

               # 下面是麻婆豆腐开源的部分特征
	, '尿酸/肌酐'
	, '红细胞计数*红细胞平均血红蛋白浓度'
	, '红细胞计数*红细胞平均体积'
	, '肾'
	, '总酶'
	, '嗜酸细胞%'
	, '淋巴细胞%'

               ]


predictors = predictors1
df_feature = data[predictors]
train_feat = df_feature[0:(len(df_train)+len(df_test_A))]
train_target = data[0:(len(df_train)+len(df_test_A))]['血糖']
train_feat['血糖'] = train_target
test_feat = df_feature[(len(df_train)+len(df_test_A)):len(data)]


kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)

#poisson regression

lgb_params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'poisson',
    'bagging_fraction': 0.8,
    'bagging_freq':1,
    'num_leaves': 12,
    'colsample_bytree': 0.6,
    'max_depth': 6,
    'min_data': 5,
    'min_hessian': 1,
    'verbose': -1
}

train_preds_lgb = np.zeros(train_feat.shape[0])
test_preds_lgb = np.zeros((test_feat.shape[0], 5))

for i, (train_index, test_index) in enumerate(kf):
	print('\n')
	print('第{}次训练...'.format(i))

	train_feat11 = train_feat.iloc[train_index]
	train_feat12 = train_feat.iloc[test_index]

	print('lightgbm')

	lgb_train1 = lgb.Dataset(train_feat11[predictors], train_feat11['血糖'])
	lgb_train2 = lgb.Dataset(train_feat12[predictors], train_feat12['血糖'])

	gbm = lgb.train(lgb_params,
	                lgb_train1,
	                num_boost_round=20000,
	                valid_sets=lgb_train2,
	                verbose_eval=500,
	                feval=evalerror,
	                early_stopping_rounds=200)

	train_preds_lgb[test_index] += gbm.predict(train_feat12[predictors])
	test_preds_lgb[:, i] = gbm.predict(test_feat)
	print('\n')



