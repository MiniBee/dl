#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: execute.py
# @time: 2020/7/9 下午2:04
# @desc:

import tensorflow as tf

# import getConfig
#
# gConfig = getConfig.get_config()
# batch_size = gConfig['batch_size']
# epochs = gConfig['epochs']


from cloudia import Cloudia
text1 = "text data..."
text2 = "text data..."

# from str
Cloudia(text1).plot()

# from list
Cloudia([text1, text2]).plot()








