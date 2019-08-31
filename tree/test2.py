#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: test2.py
# @time: 2019/8/31 上午10:17
# @desc:

import pandas as pd
import numpy as np
import seaborn as sns
import re
from pylab import mpl
import lightgbm as lgb
import xgboost as xgb
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def evalerror(pred, df):
	label = df.get_label().values.copy()
	score = mean_squared_error(label,pred)*0.5
	return ('0.5mse',score,False)

data_path = 'datalab/3964/'
df_train = pd.read_csv('/home/peihongyue/project/python/dl/data/hpd/train_data.csv', parse_dates=['体检日期'])
df_test = pd.read_csv('/home/peihongyue/project/python/dl/data/hpd/test_data.csv', parse_dates=['体检日期'])
data = pd.concat([df_train, df_test], ignore_index=True)
data = data.reset_index(drop=True)

data['性别'] = data['性别'].apply(lambda x: 1 if x == '男' else 0)
data["yearmonth"] = data['体检日期'].dt.year*100 + data['体检日期'].dt.month
data["yearweek"] = data['体检日期'].dt.year*100 + data['体检日期'].dt.weekofyear
data["month_of_year"] = data['体检日期'].dt.month
data["week_of_year"] = data['体检日期'].dt.weekofyear
data["day_of_week"] = data['体检日期'].dt.weekday

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
for col in predictors:
    if data[col].isnull().any():
        if re.search('乙肝', col, re.IGNORECASE):
            data[col].fillna(-999, inplace=True)
        else:
            data[col].fillna(data[col].median(), inplace=True)


df_feature = data[predictors]
train_feat = data[0:len(df_train)]
# train_target = data[0:(len(df_train))]['血糖']
# train_feat['血糖'] = train_target
test_feat = data[len(df_train):]

train_x, train_y = train_feat[predictors], train_feat['血糖']
test_x, test_y = test_feat[predictors], test_feat['血糖']

rf = RandomForestRegressor(n_jobs=10)
rf_param = {'n_estimators': [140, 120, 130], "max_features": ["log2", "sqrt"]}
rf_grid = GridSearchCV(estimator=rf, param_grid=rf_param, cv=5)
rf_grid.fit(train_x, train_y)
rf = rf_grid.best_estimator_
pred_y1 = rf.predict(test_x)

gbd = GradientBoostingRegressor()
gbd_param = {'n_estimators': [330, 340, 350], 'learning_rate': [0.11, 0.06, 0.07], 'max_depth': [1 ,2, 3]}
gbd_grid = GridSearchCV(estimator=gbd, param_grid=gbd_param, cv=5)
gbd_grid.fit(train_x, train_y)
gbd = gbd_grid.best_estimator_
pred_y2 = gbd.predict(test_x)

xgb_param = {'max_depth': [2,3,4], 'learning_rate': [0.12, 0.14, 0.11], 'n_estimators': [100, 110, 120]}
xgb_grid = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror'), param_grid=xgb_param, cv=5)
xgb_grid.fit(train_x, train_y)
xgb = xgb_grid.best_estimator_
pred_y3 = xgb.predict(test_x)

svc = SVR()
svc.fit(train_x, train_y)
pred_y4 = svc.predict(test_x)

gbm = lgb.LGBMRegressor(objective='regression')
gbm_param = {'n_estimators': [150, 145, 140], 'learning_rate': [0.02, 0.03, 0.01], 'max_depth': [5, 6, 7]}
gbm_grid = GridSearchCV(estimator=gbm, param_grid=gbm_param, cv=5)
gbm_grid.fit(train_x, train_y)
gbm = gbm_grid.best_estimator_
'''
# booster = gbm.booster_
# importance = booster.feature_importance(importance_type='split')
# feature_name = booster.feature_name()
# ret = []
# for (feature_name, importance) in zip(feature_name, importance):
#     print (feature_name, importance)
#     ret.append([feature_name, importance])
# ret.sort(key=lambda x: x[1], reverse=True)
# columns_index = [int(i[0].split('_')[1]) for i in ret]
# print(columns_index)
# lgb.plot_importance(gbm, max_num_features=30)
# plt.title("Featurertances")
# plt.show()
'''
pred_y5 = gbm.predict(test_x)

# cbst = cb.CatBoostRegressor(iterations=2, depth=2, learning_rate=1, loss_function='RMSE')
# # cbst_param = {'learning_rate': [0.1, 0.05, 0.15], 'depth': [1,2,3], 'iterations': [20, 50, 100]}
# # cb_grid = GridSearchCV(estimator=cbst, param_grid=cbst_param, cv=10)
# # train_pool = cb.Pool(train_x,
# #                   train_y)
# # test_pool = cb.Pool(test_x)
# cbst.fit(train_x, train_y)
# # cbst = cb_grid.best_estimator_
# pred_y6 = cbst.predict(test_x)

print(rf_grid.best_params_)
print(gbd_grid.best_params_)
print(xgb_grid.best_params_)
print(gbm_grid.best_params_)
# print(cbst_grid.best_params_)

print(mean_squared_error(test_y, pred_y1))
print(mean_squared_error(test_y, pred_y2))
print(mean_squared_error(test_y, pred_y3))
print(mean_squared_error(test_y, pred_y4))
print(mean_squared_error(test_y, pred_y5))
# print(mean_squared_error(test_y, pred_y6))
#
print(mean_squared_error(test_y, (0*pred_y2 + 0.1*pred_y3 + 0.1*pred_y1 + 0.8*pred_y5)))




# kf = KFold(len(train_feat), n_folds = 5, shuffle=True)
#
# lgb_params = {
# 	'learning_rate': 0.01,
# 	'boosting_type': 'gbdt',
# 	'objective': 'poisson',
# 	'bagging_fraction': 0.8,
# 	'bagging_freq':1,
# 	'num_leaves': 12,
# 	'colsample_bytree': 0.6,
# 	'max_depth': 6,
# 	'min_data': 5,
# 	'min_hessian': 1,
# 	'verbose': -1
# }
#
# train_preds_lgb = np.zeros(train_feat.shape[0])
# test_preds_lgb = np.zeros((test_feat.shape[0], 5))
#
# for i, (train_index, test_index) in enumerate(kf):
# 	print('\n')
# 	print('第{}次训练...'.format(i))
# 	train_feat11 = train_feat.iloc[train_index]
# 	train_feat12 = train_feat.iloc[test_index]
# 	print('lightgbm')
# 	lgb_train1 = lgb.Dataset(train_feat11[predictors], train_feat11['血糖'])
# 	lgb_train2 = lgb.Dataset(train_feat12[predictors], train_feat12['血糖'])
# 	gbm = lgb.train(lgb_params,
# 					lgb_train1,
# 					num_boost_round=20000,
# 					valid_sets=lgb_train2,
# 					verbose_eval=500,
# 					feval=evalerror,
# 					early_stopping_rounds=200)
# 	train_preds_lgb[test_index] += gbm.predict(train_feat12[predictors])
# 	test_pred = gbm.predict(test_feat[predictors])
# 	print('test')
# 	print(mean_squared_error(test_pred, test_feat['血糖']))
# 	print('\n')


# print(mean_squared_error())




