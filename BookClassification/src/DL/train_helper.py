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
        print('User AdamW ... ')
        print(config.device)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        }, {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.eps)
    
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0 
    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch {} / {}'.format(epoch + 1, config.num_epochs))
        for i, (trains, mask, tokens, labels) in tqdm(enumerate(train_iter)):
            trains = trains.to(config.device)




