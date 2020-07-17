#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: teat.py
# @time: 2020/5/27 下午1:45
# @desc:


def find_pairs(array, k):
    res = []
    array.sort()
    left, right = 0, len(array) - 1
    while left < right:
        while right > 1 and array[right] == array[right-1]:
            right -= 1
        while left < len(array) - 1 and array[left] == array[left + 1]:
            left += 1
        if array[right] - array[left] == k:
            res.append((array[left], array[right]))






