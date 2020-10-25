#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_flow.py
@Time    :   2020/10/19 10:33:37
@Author  :   Hongyue Pei 
@Version :   1.0
@Contact :   
@Desc    :   None
'''

import pandas as pd 


def get_data(path, encoding=None, sep='\t', names=None):
    df = pd.read_csv(path, sep=sep, encoding=encoding, names=names)
    return df 


def merge_data(data_path):
    train_query = get_data(data_path + 'train/train.query.tsv', sep='\t', names=['query_id', 'query'])
    train_reply = get_data(data_path + 'train/train.reply.tsv', sep='\t', names=['query_id', 'reply_id', 'reply', 'is_match'])
    
    train_df = train_reply.merge(train_query, on='query_id', how='right')[['query_id', 'reply_id', 'query', 'reply', 'is_match']]
    
    test_query = get_data(data_path + 'test/test.query.tsv', sep='\t', encoding='gbk', names=['query_id', 'query'])
    test_reply = get_data(data_path + 'test/test.reply.tsv', sep='\t', encoding='gbk', names=['query_id', 'reply_id', 'reply'])

    test_df = test_reply.merge(test_query, on='query_id', how='right')[['query_id', 'reply_id', 'query', 'reply']]

    train_df.to_csv(data_path + 'train_df.csv')
    test_df.to_csv(data_path + 'test_df.csv')


def data_cut(file_path, columns):
    pass




if __name__ == "__main__":
    pd.set_option('display.width', 1800)
    pd.set_option('display.max_colwidth', 200)
    pd.set_option('display.max_columns', 800)
    pd.set_option('display.max_rows', 800)

    data_path = '/data/phy/datas/beike_qa/'
    # merge_data(data_path)




