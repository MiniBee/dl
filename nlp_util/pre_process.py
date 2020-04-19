#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/4/19 下午3:24
#@Author  :hongyue pei 
#@FileName: pre_process.py
#@Software: PyCharm


def mask(token_list, max_len):
    tokens_len = len(token_list)
    if tokens_len < max_len:
        return token_list + [0] * (max_len - tokens_len)
    else:
        return token_list


def padding(x_train, x_test):
    len_list = [len(i) for i in x_train + x_test]
    max_len = max(len_list)
    return [mask(i, max_len) for i in x_train], [mask(i, max_len) for i in x_test]


