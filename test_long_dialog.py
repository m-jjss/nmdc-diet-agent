# -*- coding: utf-8 -*-
"""
长连续多轮对话测试——模拟用户一天完整的饮食规划。
验证上下文承接、约束记忆、否定换菜、追加约束、多人场景等逻辑通畅性。
"""
import requests

API = 'http://127.0.0.1:5000/api/dialog'
UID = 'daily_user_001'


def talk(msg, reset=False, user_ids=None):
    body = {'message': msg, 'user_id': UID}
    if reset:
        body['reset'] = True
    if user_ids:
        body['user_ids'] = user_ids
    d = requests.post(API, json=body, timeout=120).json()
    recs = [x.get('name') for x in d.get('recommendations', [])]
    resp = (d.get('response', '') or '').replace('\n', ' ')
    print(f"  用户: {msg}")
    if recs:
        print(f"  推荐({len(recs)}): {'、'.join(recs[:6])}")
    print(f"  助手: {resp[:140]}")
    print('  ' + '-' * 60)
    return d


print("=" * 70)
print("长连续多轮对话：一个普通用户一天从早到晚")
print("=" * 70)

# 早上：起床问候 + 早餐
print("\n【早上】")
talk("早上好", reset=True)
talk("给我推荐个早餐吧")

# 上午：追加忌口
print("\n【上午·追加约束】")
talk("我不吃花生，记得啊")

# 中午：午餐 + 不想太油
print("\n【中午】")
talk("中午吃什么好")
talk("不要太油腻，清淡点")

# 下午：否定换菜
print("\n【下午·否定重推】")
talk("刚才那几个不太想吃，换一批清淡的")

# 晚上：晚餐 + 数量 + 加个汤
print("\n【晚上】")
talk("晚上三个人吃饭，推荐一桌")
talk("只要四道菜就行")
talk("再加一个汤")

# 夜宵：控制热量
print("\n【夜宵·健康关注】")
talk("半夜饿了，想吃点不胖的夜宵")

# 确认是否记住花生过敏
print("\n【记忆校验】")
talk("今天还有什么推荐的")
