#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   CNN.py
@Time    :   2020/09/27 14:26:23
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torchsnooper   # pytorch 代码调试神器
import numpy as np 
from __init__ import * 

# squeeze 对数据进行维度压缩
# unsqueeze 对数据进行维度扩充

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        # input_channel, output_channel, kernel_size
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embedding_dim)) for k in config.filter_sizes]
        )
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x 

    def forward(self, x):
        print('before embedding ... :', x.shape)
        x = self.embedding(x[0])
        print('after embedding ... :', x.shape)
        x = x.unsqueeze(1)
        x = torch.cat(
            [self.conv_and_pool(x, conv) for conv in self.convs], 1
        )
        x = self.dropout(x)
        x = self.fc(x)
        return x 
    





