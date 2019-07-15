#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: get_data.py
# @time: 2019/7/8 9:53
# @desc:

import requests
from bs4 import BeautifulSoup
import os
import traceback


def download(url, filename):
	if os.path.exists(filename):
		print('file exists ... ')
		return
	try:
		r = requests.get(url, stream=True, timeout=60)
		r.raise_for_status()
		with open(filename, 'wb') as f:
			for chunk in r.iter_content(chunk_size=1024):
				if chunk:
					f.write(chunk)
					f.flush()
	except KeyboardInterrupt:
		if os.path.exists(filename):
			os.remove(filename)
			raise KeyboardInterrupt
	except Exception:
		traceback.print_exc()
		if os.path.exists(filename):
			os.remove(filename)


if os.path.exists('./data/imgs/') == False:
	os.makedirs('./data/imgs')

for i in range(10):
	url = 'http://konachan.net/post?page=%d&tags=' % i
	html = requests.get(url).text
	soup = BeautifulSoup(html, 'html.parser')
	for img in soup.find_all('img', class_="preview"):
		target_url = img['src']
		filename = os.path.join('./data/imgs', target_url.split('/')[-1])
		download(target_url, filename)
		print('page: ', i)




