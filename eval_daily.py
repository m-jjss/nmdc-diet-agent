# -*- coding: utf-8 -*-
"""
日常对话场景模拟测试——覆盖高频日常使用需求。
通过真实后端 /api/dialog 逐场景实测，输出每轮的推荐与回复。
"""
import requests
import json
import time
import sys

API = 'http://127.0.0.1:5000/api/dialog'


def talk(msg, user_id=None, reset=False, user_ids=None):
    body = {'message': msg}
    if user_id:
        body['user_id'] = user_id
    if user_ids:
        body['user_ids'] = user_ids
    if reset:
        body['reset'] = True
    t0 = time.time()
    try:
        r = requests.post(API, json=body, timeout=120)
        d = r.json()
    except Exception as e:
        return {'error': str(e), '_ms': 0}
    d['_ms'] = round((time.time() - t0) * 1000)
    return d


def show(tag, d, show_rec=True):
    recs = [x.get('name') for x in d.get('recommendations', [])]
    resp = d.get('response', '')
    line = f"[{tag}] intent={d.get('intent','?')} | {len(recs)}道 | {d.get('_ms',0)}ms"
    print(line)
    if show_rec and recs:
        print('    推荐:', '、'.join(recs[:6]))
    print('    回复:', resp[:160].replace('\n', ' '))
    print('    ---')
    return d


if __name__ == '__main__':
    print("=" * 60)
    print("日常对话场景模拟测试")
    print("=" * 60)

    scenarios = [
        ("1.开场闲聊", "你好", 'u_greet', True),
        ("2.模糊点餐", "今晚吃什么", 'u_vague', True),
        ("3.具体想吃", "想吃点辣的", 'u_spicy', True),
        ("4.指定食材", "我想吃虾", 'u_shrimp', True),
        ("5.忌口排除", "我不吃香菜", 'u_coriander', True),
        ("6.过敏声明", "我对花生过敏", 'u_allergy', True),
        ("7.健康需求", "我要控糖，推荐午餐", 'u_sugar', True),
        ("8.减肥需求", "我在减肥，吃什么好", 'u_diet', True),
        ("9.老人用餐", "给老人做的，要软烂", 'u_elder', True),
        ("10.儿童用餐", "给小孩做点爱吃的", 'u_kid', True),
        ("11.早餐场景", "明天早餐吃什么", 'u_breakfast', True),
        ("12.夜宵场景", "半夜饿了，想吃夜宵", 'u_night', True),
        ("13.食材替换", "家里只有鸡蛋和番茄", 'u_fridge', True),
        ("14.菜单详情", "怎么做红烧肉", 'u_detail', True),
        ("15.营养咨询", "鸡胸肉的营养怎么样", 'u_nutri', True),
        ("16.换一道菜", "推荐3道菜", 'u_subst', True),
        ("17.情感问候", "谢谢你的推荐", 'u_thanks', True),
        ("18.能力边界", "今天天气怎么样", 'u_weather', True),
        ("19.数量指定", "今天只吃四道菜", 'u_four', True),
        ("20.多吃几道", "我想多吃几道菜", 'u_more', True),
    ]

    for tag, msg, uid, reset in scenarios:
        d = talk(msg, user_id=uid, reset=reset)
        show(tag, d)
