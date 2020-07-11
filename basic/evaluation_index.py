#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: evaluation_index.py
# @time: 2020/7/10 上午10:43
# @desc:

import numpy as np

# 分类
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
y_pred = [0, 1, 0, 1]
y_real = [0, 1, 1, 1]
print('ACC: ', accuracy_score(y_pred, y_real))
print('Precision: ', precision_score(y_pred, y_real))
print('Recall', recall_score(y_pred, y_real))
print('F1-score', f1_score(y_pred, y_real))
print('AUC: ', roc_auc_score(y_real, y_pred))


# 回归
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_true = np.array([1.0, 5.0, 4.0, 3.0, 2.0, 5.0, -3.0])
y_pred = np.array([1.0, 4.5, 3.8, 3.2, 3.0, 4.8, -2.2])

# MSE
print('MSE:',mean_squared_error(y_true, y_pred))
# RMSE
print('RMSE:',np.sqrt(mean_squared_error(y_true, y_pred)))
# MAE
print('MAE:',mean_absolute_error(y_true, y_pred))
# R2-score
print('R2-score:', r2_score(y_true, y_pred))


