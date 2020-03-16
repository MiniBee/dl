#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/3/16 下午9:36
#@Author  :hongyue pei 
#@FileName: test.py
#@Software: PyCharm

import re

print(re.findall('[\w]+', 'a哈-哈哈，哈123', re.IGNORECASE))
