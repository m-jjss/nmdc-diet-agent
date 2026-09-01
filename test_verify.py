# -*- coding: utf-8 -*-
import requests

API = 'http://127.0.0.1:5000/api/dialog'
UID = 'verify_001'


def talk(msg, reset=False):
    body = {'message': msg, 'user_id': UID}
    if reset:
        body['reset'] = True
    d = requests.post(API, json=body, timeout=120).json()
    recs = [x.get('name') for x in d.get('recommendations', [])]
    print(f"  用户: {msg}")
    print(f"  推荐: {'、'.join(recs[:6])}")
    print(f"  助手: {((d.get('response') or '').replace(chr(10), ' '))[:120]}")
    print('  ' + '-' * 50)
    return recs


print("=== 验证：清淡约束是否真正生效 ===")
talk("推荐几个家常菜", reset=True)
r2 = talk("不要太油腻，清淡一点")
r3 = talk("太油了，换点清蒸的")

print("=== 验证：不吃辣 ===")
talk("推荐菜", reset=True)
talk("我不吃辣")
