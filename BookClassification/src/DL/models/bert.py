#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   bert.py
@Time    :   2020/09/27 14:19:36
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''


import torch.nn as nn 
from transformers import BertConfig, BertModel

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        model_config = BertConfig.from_pretrained(config.bert_path, num_labels=config.num_classes)
        self.bert = BertModel.from_pretrained(config.bert_path, config=model_config)

        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]
        maxk = x[1]
        token_type_ids = x[2]
        _, pooled = self.bert(context, attention_mask=mask, token_type_ids=token_type_ids)
        out = self.fc(pooled)
        return out 





