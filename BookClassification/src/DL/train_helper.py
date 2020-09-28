#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_helper.py
@Time    :   2020/09/27 15:48:08
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from sklearn import metrics
import time
from tqdm import tqdm 
from transformers import AdamW, get_linear_schedule_with_warmup

from __init__ import * 
from src.utils.tools import get_time_dif


# 网络权重初始化，xavier
def init_netword(model, method='xavier', exclude='embedding', seed=11):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass 


def train(config, model, train_iter, dev_iter, test_iter):
    start_time  = time.time()
    model.train()
    if config.model_name.isupper():
        print('User Adam ... ')
        print(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        pass




