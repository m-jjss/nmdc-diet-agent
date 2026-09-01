# -*- coding: utf-8 -*-
"""
多场景日常对话压力测试——模拟不同用户的真实使用场景。
覆盖：特殊人群、慢性病、过敏、多人聚餐、减脂、营养咨询、菜谱详情、
否定换菜、追加约束、数量控制、记忆校验等日常对话。
"""
import requests
import re

API = 'http://127.0.0.1:5000/api/dialog'

issues = []
turns = 0


def talk(msg, uid, reset=False, user_ids=None):
    global turns
    body = {'message': msg, 'user_id': uid}
    if reset:
        body['reset'] = True
    if user_ids:
        body['user_ids'] = user_ids
    d = requests.post(API, json=body, timeout=120).json()
    turns += 1
    recs = [x.get('name') for x in d.get('recommendations', [])]
    resp = (d.get('response', '') or '').replace('\n', ' ')
    n = len(recs)
    flag = ''
    # 质量检查点
    if not d.get('success'):
        flag = ' ❌失败'
        issues.append(f"[{uid}] '{msg}' -> 接口失败: {d.get('error')}")
    if not resp.strip():
        flag += ' ❌空回复'
        issues.append(f"[{uid}] '{msg}' -> 空回复")
    if n > 8:
        flag += ' ❌数量过多'
        issues.append(f"[{uid}] '{msg}' -> 返回{n}道菜")
    if n == 0 and d.get('intent') not in ('greet', 'cancel', 'confirm', 'ask_nutrition',
                                          'ask_recipe_detail', 'vague_query', 'set_preferences',
                                          'farewell'):
        # 食材替换咨询（"没有猪肉了，能换成鸡肉吗"）是建议性回复，无需推荐
        _is_swap_advice = bool(re.search(r'没有[\u4e00-\u9fa5]{1,6}[了]?|换成|替代|替换', msg)) \
            and d.get('intent') == 'request_substitute'
        if not _is_swap_advice:
            flag += ' ⚠️无推荐'
            issues.append(f"[{uid}] '{msg}' -> 无推荐(intent={d.get('intent')})")
    print(f"[{uid}] {msg}")
    print(f"  intent={d.get('intent')} | {n}道{flag}")
    if recs:
        print(f"  菜: {'、'.join(recs)}")
    print(f"  助手: {resp[:150]}")
    print('  ' + '-' * 62)
    return d


print("=" * 70)
print("场景1：孕妇+家庭聚餐（多人约束合并）")
print("=" * 70)
talk("周末请朋友来家里吃饭，六个人，其中有个孕妇", 'preg1', reset=True)
talk("她怀孕三个月，有什么要注意的吗？", 'preg1')
talk("再加两个清淡的素菜", 'preg1')
talk("不要螃蟹和生冷的东西", 'preg1')

print("=" * 70)
print("场景2：减脂白领（多轮叠加）")
print("=" * 70)
talk("我最近在减脂，帮我安排今天的午饭", 'diet1', reset=True)
talk("要高蛋白低脂肪的", 'diet1')
talk("主食换成粗粮", 'diet1')
talk("帮我查一下鸡胸肉的热量", 'diet1')
talk("刚才那几道能再给我看看怎么做吗", 'diet1')

print("=" * 70)
print("场景3：糖尿病+给老人做")
print("=" * 70)
talk("我有糖尿病，今天晚饭吃点啥", 'dia1', reset=True)
talk("不要太甜，米饭也少点", 'dia1')
talk("这是我妈吃的，她牙口不好，要软烂的", 'dia1')

print("=" * 70)
print("场景4：海鲜过敏 + 食材替换")
print("=" * 70)
talk("帮我推荐晚餐，我虾过敏", 'al1', reset=True)
talk("我还不吃香菜", 'al1')
talk("把第二道菜换掉，换成清淡点的", 'al1')
talk("家里没有猪肉了，能换成鸡肉吗", 'al1')

print("=" * 70)
print("场景5：儿童挑食")
print("=" * 70)
talk("孩子不爱吃蔬菜，怎么搭配能让他多吃点？", 'kid1', reset=True)
talk("不要太辣，孩子吃不了", 'kid1')
talk("还要有营养，孩子正在长身体", 'kid1')

print("=" * 70)
print("场景6：生病恢复期（暖胃清淡）")
print("=" * 70)
talk("这两天胃不舒服，想吃点暖胃的", 'sick1', reset=True)
talk("不要油腻的，清淡点", 'sick1')
talk("来点好消化的粥或者汤", 'sick1')

print("=" * 70)
print("场景7：周末宴客（多人不同口味）")
print("=" * 70)
talk("周末请同事来家里，八个人", 'party1', reset=True)
talk("有人吃辣有人不吃辣，能兼顾吗", 'party1')
talk("来一桌，荤素搭配好一点", 'party1')
talk("再来个硬菜", 'party1')

print("=" * 70)
print("场景8：营养咨询 + 模糊交互 + 能力边界")
print("=" * 70)
talk("番茄炒蛋的营养怎么样", 'nut1', reset=True)
talk("那牛肉的营养呢", 'nut1')
talk("明天天气怎么样", 'nut1')
talk("谢谢，再见", 'nut1')

print("=" * 70)
print("场景9：老年人一日三餐（长连续）")
print("=" * 70)
talk("早上好，帮我看看今天吃什么", 'eld2', reset=True)
talk("我有高血压，不能太咸", 'eld2')
talk("中午了，吃啥", 'eld2')
talk("晚上想吃点粥，配个菜", 'eld2')

print("=" * 70)
print("场景10：追加忌口 + 记忆校验（长连续）")
print("=" * 70)
talk("帮我推荐个早餐", 'mem1', reset=True)
talk("我不吃鸡蛋，记住了", 'mem1')
talk("中午吃啥", 'mem1')
talk("我想吃肉，但不要肥肉", 'mem1')
talk("你还记得我不吃鸡蛋吧？", 'mem1')

print("\n" + "=" * 70)
print(f"总轮次: {turns} | 问题数: {len(issues)}")
for i in issues:
    print("  -", i)
print("=" * 70)
