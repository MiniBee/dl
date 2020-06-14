#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: eda.py
# @time: 2020/6/2 下午3:28
# @desc:

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_path = '/data/phy/datas/df370/'
    df = pd.read_csv(file_path + 'train.csv', sep='\t')
    df['label'].value_counts().plot.bar()
    plt.show()





