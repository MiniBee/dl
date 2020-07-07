#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: get_config.py
# @time: 2020/7/7 下午2:03
# @desc:
from configparser import ConfigParser


def get_config(conf_path='project.ini'):
    parser = ConfigParser()
    parser.read(conf_path)
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]
    return dict(_conf_ints + _conf_strings)


if __name__ == '__main__':
    print(get_config())
