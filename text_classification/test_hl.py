#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: test_hl.py
# @time: 2020/5/4 下午4:49
# @desc:

from xlsxwriter.workbook import Workbook


workbook = Workbook('test.xlsx')
worksheet = workbook.add_worksheet('Sheet 1')

red = workbook.add_format({'color': 'red'})
black = workbook.add_format({'color': 'black'})

sequences = ['AAABBB', 'BBBCCC', 'CCCDDD', 'DDDEEE']

worksheet.set_column('A:A', 20)

for row_num, sequence in enumerate(sequences):
    format_pairs = []
    # for i, base in enumerate(sequence):
    #     if i == 2 or i == 3:
    #         format_pairs.extend((red, base))
    #     else:
    #         format_pairs.extend((black, base))
    if row_num == 2:
        format_pairs.extend((red, sequence))
        format_pairs.extend((black, sequence))
    else:
        format_pairs.extend((black, sequence))
        format_pairs.extend((red, sequence))
    for i in format_pairs:
        print(i)
    worksheet._write_rich_string(row_num, 1, *format_pairs)
workbook.close()



