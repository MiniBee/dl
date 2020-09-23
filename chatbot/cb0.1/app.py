#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: app.py
# @time: 2020/7/6 下午6:11
# @desc:

from flask_api import Flask, render_template, request, jsonify
# import execute
import time
import threading
import jieba
import random
import res


def heartbeat():
    print(time.strftime('%Y-%m-%d %H:%M:$S - heartbeat'), time.localtime(time.time()))
    timer = threading.Timer(60, heartbeat)
    timer.start()

app = Flask(__name__, static_url_path="/static")

no_res = ['走开。。', '我还不会说话。。。', '我听不懂。。']

@app.route('/message', methods=['POST'])
def reply():
    req_msg = request.form['msg']
    req_msg = ' '.join(jieba.cut(req_msg))
    if '天气' in req_msg:
        res_msg = res.get_weather()
    else:
        # res_msg = execute.predict(reg_msg)
        res_msg = random.choice(no_res)
    res_msg = res_msg.replace('_UNK', '^_^')
    res_msg = res_msg.strip()
    if res_msg == ' ':
        res_msg = 'Do you speak chinese?'
    return jsonify({'text': res_msg})

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8808)





