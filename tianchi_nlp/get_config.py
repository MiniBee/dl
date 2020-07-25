#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: get_config.py
# @time: 2020/7/24 下午6:50
# @desc:


from configparser import ConfigParser


def get_config(config_path='project.ini'):
    parser = ConfigParser()
    parser.read(config_path)
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]
    return dict(_conf_ints + _conf_strings)


if __name__ == '__main__':
    print(get_config())

