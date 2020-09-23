#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: flast_manager.py
# @time: 2020/9/21 上午10:06
# @desc:


from flask import Flask
from flask import request
from load_model import Model

cls = Model()

app = Flask(__name__)


@app.route('/api/classifier', methods=['POST'])
def predict_api():
    input_json = request.get_json(force=False, silent=False, cache=True)
    response = cls.predict(input_json['input_str'])
    return response


if __name__ == '__main__':
    app.run()










