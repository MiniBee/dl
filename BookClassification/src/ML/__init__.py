#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: __init__.py.py
# @time: 2020/9/23 下午3:09
# @desc:
import os
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(curPath)[0])[0]
import sys
sys.path.append(root_path)

