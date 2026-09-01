# -*- coding: utf-8 -*-
import requests

API = 'http://127.0.0.1:5000/api/dialog'


def talk(msg, uid, reset=True):
    body = {'message': msg, 'user_id': uid}
    if reset:
        body['reset'] = True
    d = requests.post(API, json=body, timeout=90).json()
    n = len(d.get('recommendations', []))
    print(f"[{msg}] intent={d.get('intent')} | {n}道")
    print('  回复:', (d.get('response', '') or '')[:130].replace('\n', ' '))
    print()


print('=== 营养咨询 ===')
talk('鸡胸肉的营养怎么样', 'nutr1')
talk('虾的热量是多少', 'nutr2')
talk('红烧肉的卡路里', 'nutr3')
print('=== 食材替换 ===')
talk('家里只有鸡蛋和番茄，能做什么', 'fridge1')
print('=== 老人(复核) ===')
talk('给老人做的，要软烂', 'elder1')
