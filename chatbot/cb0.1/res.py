#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: res.py
# @time: 2020/7/7 上午11:15
# @desc:

import json
import requests
import pprint


def get_weather():
    weatherJsonUrl = 'http://wthrcdn.etouch.cn/weather_mini?city=北京'
    response = requests.get(weatherJsonUrl)
    response.raise_for_status()
    weather_data = json.loads(response.text)
    weather_data = weather_data['data']['forecast'][0]
    return '北京，今天天气{}，最高温度{},最低温度{}, {}{}级'.format(weather_data['type'],
                  weather_data['high'][2:],
                  weather_data['low'][2:],
                  weather_data['fengxiang'],
                  weather_data['fengli'])



if __name__ == '__main__':
    weather_data = get_weather()
    print(weather_data)



