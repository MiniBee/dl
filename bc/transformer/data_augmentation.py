#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/3/21 下午9:02
#@Author  :hongyue pei 
#@FileName: data_augmentation.py
#@Software: PyCharm

import numpy as np
import random
import copy


def random_delete(x_train, y_train):
    x_ret = []
    y_ret = []
    for j in range(3):
        a = list(copy.deepcopy(x_train))
        for k in range(10):
            del_idx = random.choice([i for i in range(len(a))])
            if len(a) < 10:
                break
            a.pop(del_idx)
        x_ret.append(a)
        y_ret.append(y_train)
    return x_ret, y_ret

