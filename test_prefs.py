# -*- coding: utf-8 -*-
import requests
API = 'http://127.0.0.1:5000/api/dialog'
UID = 'debug_oily_1'
# 重置后第一次
d = requests.post(API, json={'message': '推荐几个家常菜', 'user_id': UID, 'reset': True}, timeout=120).json()
print('轮1 prefs:', d.get('user_preferences', {}))
print('轮1 recs:', [x.get('name') for x in d.get('recommendations', [])])
print()
d2 = requests.post(API, json={'message': '不要太油腻，清淡一点', 'user_id': UID}, timeout=120).json()
print('轮2 prefs:', d2.get('user_preferences', {}))
print('轮2 recs:', [x.get('name') for x in d2.get('recommendations', [])])
print()
d3 = requests.post(API, json={'message': '太油了，换点清蒸的', 'user_id': UID}, timeout=120).json()
print('轮3 prefs:', d3.get('user_preferences', {}))
print('轮3 recs:', [x.get('name') for x in d3.get('recommendations', [])])
