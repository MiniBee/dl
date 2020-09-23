#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: models.py
# @time: 2020/9/23 下午3:09
# @desc:

import os 
import lightgbm as lgb 
import numpy as np 
import torchvision
import json
import pandas as pd 

from sklearn.ensemble import RandomForestClassifier
import joblib
from transformers import BertModel, BertTokenizer

from __init__ import * 
from src.utils.tools import create_logger
from src.utils import config 

logger = create_logger(config.log_path + 'model.log')

class Model(object):
    def __init__(self, model_path=None, feature_engineer=False, train_mode=True):
        self.res_model = torchvision.models.resnet152(pretrained=True)




