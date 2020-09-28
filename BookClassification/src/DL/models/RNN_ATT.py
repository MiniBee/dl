#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   RNN_ATT.py
@Time    :   2020/09/27 15:19:25
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_size, config.num_layers, 
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.Tensor(config.hidden_size * 2))  # 初始化 attention 的W
        self.tanh2 = nn.Tanh()

        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x):
        x = self.embedding(x[0])
        H, _ = self.lstm(x)
        M = self.tanh1(H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        x = H * alpha
        x = torch.sum(x, 1)
        x = F.relu(x)
        x = self.fc1(x)
        x = self.fc(x)
        return x 
        






