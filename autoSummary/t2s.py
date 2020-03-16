#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/2/16 下午3:02
#@Author  :hongyue pei 
#@FileName: t2s.py
#@Software: PyCharm

from openccpy.opencc import *
import json
import pandas as pd
import os


if __name__ == '__main__':
    input_path = '/home/peihongyue/data/kaikeba/zhwiki/zhwikidata/'
    dirs = os.listdir(input_path)
    for dir in dirs:
        if dir == 'data' or dir == 'sqlResult_1558435.csv' or dir == 'news.csv':
            continue
        dir_ = os.path.join(input_path, dir)
        files = os.listdir(dir_)
        for file in files:
            file_ = os.path.join(dir_, file)
            print(file_)
            text = []
            with open(file_) as f:
                for i, line in enumerate(f):
                    text.append(''.join([Opencc.to_simple(x) for x in line]))
            with open(os.path.join(input_path, 'data/' + file + dir), 'w') as f:
                for t in text:
                    f. write(t)

