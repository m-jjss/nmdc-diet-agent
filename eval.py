"""
竞赛自动评分脚本

评测维度（总分100分）：
(1) 基础推荐(20分)：硬约束满足率、菜谱真实性
(2) 复杂场景(20分)：多人约束、搭配合理性、营养均衡
(3) 多轮交互(30分)：上下文一致性、最小化修改、交互自然度
(4) 性能效率(30分)：首Token延迟、端到端响应、多轮平均

验收形式：自动评分占60% + 专家评审占40%
"""
import requests
import json
import os
import time
import sys
import re
from typing import List, Dict, Set, Optional

BASE_URL = "http://127.0.0.1:5000"
DIALOG_URL = f"{BASE_URL}/api/dialog"

# ─── 数据加载 ───────────────────────────────────────────

def load_recipes() -> tuple:
    """加载菜谱库：返回 (名称集合, 完整菜谱列表)"""
    try:
        with open('recipes_parsed.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            name_set = {r.get('name', '') for r in data if r.get('name')}
            recipe_dict = {r.get('name', ''): r for r in data if r.get('name')}
            return name_set, recipe_dict
    except Exception:
        return set(), {}


def load_user_profiles() -> Dict:
    """加载用户健康档案：返回 {user_id: profile_dict}"""
    try:
        with open('50个用户健康档案_详细版7.13.json', 'r', encoding='utf-8') as f:
            profiles = json.load(f)
            return {str(p['id']): p for p in profiles}
    except Exception:
        return {}


ALL_RECIPES, RECIPE_DATA = load_recipes()
USER_PROFILES = load_user_profiles()
NUTRITION_DB = {}  # 营养数据库
_nut_path = os.path.join(os.path.dirname(__file__), 'nutrition_database.json')
if os.path.exists(_nut_path):
    with open(_nut_path, 'r', encoding='utf-8') as f:
        NUTRITION_DB = json.load(f)

# 疾病对应的禁忌关键词
DISEASE_FORBIDDEN = {
    '高血压': ['高盐', '腌制', '腊肉', '咸鱼', '腐乳', '盐焗', '卤肉', '酱菜'],
    '糖尿病': ['糖', '甜', '蜜', '冰糖', '红糖', '白糖', '蜂蜜', '炼乳', '果酱', '糖醋', '拔丝'],
    '痛风': ['内脏', '肝', '腰', '脑', '沙丁', '凤尾鱼', '嘌呤', '啤酒', '浓汤'],
    '高血脂': ['肥肉', '猪油', '奶油', '黄油', '油炸', '炸', '油条', '油饼', '酥皮'],
    '高尿酸': ['海鲜', '虾', '蟹', '扇贝', '干贝', '鲜贝', '鱿鱼', '沙丁', '内脏', '啤酒', '浓汤'],
}

# 肉类关键词（用于荤素比检测）
MEAT_KEYWORDS = ['鸡', '鸭', '鹅', '猪', '牛', '羊', '鱼', '虾', '蟹', '贝',
                 '鱿', '肉', '排', '翅', '腿', '蹄', '骨', '肚', '肝', '肠',
                 '蛋', '芝士', '火腿', '培根', '腊', '卤', '熏', '鲍', '蛤', '鳗', '蚝']

# 热菜烹饪方式关键词  
HOT_COOKING = ['炒', '烧', '炖', '蒸', '煮', '炸', '煎', '烤', '焖', '煲',
               '焗', '烩', '爆', '熘', '煨', '焯', '烫', '熬', '卤']

# 冷菜关键词
COLD_KEYWORDS = ['凉拌', '沙拉', '刺身', '冷', '冰', '冻', '醉', '糟']


def extract_cooking_methods(recipe: Dict) -> Set[str]:
    """从菜谱步骤中提取烹饪方式"""
    methods = set()
    steps = recipe.get('steps', '')
    name = recipe.get('name', '')
    label = recipe.get('label', '')
    combined = steps + name + label
    for m in HOT_COOKING:
        if m in steps:
            methods.add(m)
    for c in COLD_KEYWORDS:
        if c in steps or c in name:
            methods.add(c)
    # 饮品/甜点/烘焙特殊处理
    if any(kw in combined for kw in ['奶昔', '奶盖', '茶', '咖啡', '果汁', '冰沙']):
        methods.add('饮品调制')
    if any(kw in combined for kw in ['饼干', '蛋糕', '面包', '吐司', '糕点']):
        methods.add('烘焙')
    if any(kw in combined for kw in ['粥', '汤', '羹']):
        methods.add('炖煮')
    return methods


def is_meat_dish(recipe: Dict) -> bool:
    """判断是否为荤菜"""
    name = recipe.get('name', '')
    ingredients = recipe.get('ingredients', '')
    tags = recipe.get('tags', [])
    label = recipe.get('label', '')
    combined = name + ingredients + ' '.join(tags) + label
    return any(kw in combined for kw in MEAT_KEYWORDS)


def is_cold_dish(recipe: Dict) -> bool:
    """判断是否为冷菜（凉拌、沙拉、冷盘、冰品等）"""
    name = recipe.get('name', '')
    steps = recipe.get('steps', '')
    label = recipe.get('label', '')
    combined = name + steps + label
    return any(kw in combined for kw in COLD_KEYWORDS)


def estimate_recipe_nutrition(recipe_name: str) -> Dict:
    """根据菜谱食材粗略估算营养素（热量/蛋白质/脂肪/碳水）"""
    recipe = RECIPE_DATA.get(recipe_name, {})
    if not recipe:
        return {'热量': 0, '蛋白质': 0, '脂肪': 0, '碳水': 0}
    ingredients = recipe.get('ingredients', '')
    total = {'热量': 0, '蛋白质': 0, '脂肪': 0, '碳水': 0}
    for ing, data in NUTRITION_DB.items():
        if ing in ingredients:
            total['热量'] += data.get('热量', 0)
            total['蛋白质'] += data.get('蛋白质', 0)
            total['脂肪'] += data.get('脂肪', 0)
            total['碳水'] += data.get('碳水', 0)
    return total


def check_constraint_violation(recipe: Dict, user_profile: Dict) -> List[str]:
    """检查单个菜谱是否违反用户约束，返回违反项列表"""
    violations = []
    name = recipe.get('name', '')
    ingredients = recipe.get('ingredients', '')
    allergens_in_recipe = recipe.get('allergens', [])
    label = recipe.get('label', '')
    combined = name + ingredients + label

    # 1. 过敏检查
    user_allergies = user_profile.get('过敏食材', [])
    for allergen in user_allergies:
        if allergen and allergen in combined:
            violations.append(f"过敏({allergen})")
        # 也检查菜谱自带的过敏原标记
        for a in allergens_in_recipe:
            if a and allergen in a:
                violations.append(f"过敏标记({allergen})")

    # 2. 疾病禁忌检查（从体检指标推断疾病）
    indicators = user_profile.get('体检指标', {})
    for disease, forbidden_kw in DISEASE_FORBIDDEN.items():
        # 先判断用户是否有此疾病
        has_disease = False
        if disease == '高血压':
            bp = indicators.get('血压_mmHg', '')
            if bp:
                parts = bp.split('/')
                if len(parts) >= 2:
                    has_disease = int(parts[0]) >= 140 or int(parts[1]) >= 90
        elif disease in ('高血脂',):
            tc = indicators.get('总胆固醇_mmol/L', 0)
            has_disease = tc > 6.2
        elif disease in ('糖尿病',):
            glu = indicators.get('空腹血糖_mmol/L', 0)
            has_disease = glu >= 7.0
        elif disease in ('痛风', '高尿酸'):
            ua = indicators.get('尿酸_umol/L', 0)
            has_disease = ua > 420

        if has_disease:
            for kw in forbidden_kw:
                if kw in combined:
                    violations.append(f"{disease}禁忌({kw})")
                    break

    # 3. 特殊人群检查
    special_groups = user_profile.get('特殊人群', [])
    for group in special_groups:
        if group == '孕妇' and any(kw in combined for kw in ['白酒', '啤酒', '红酒', '薏米', '山楂', '螃蟹', '甲鱼']):
            violations.append(f"孕妇禁忌(特殊食材)")

    return violations


# ─── 评分器 ─────────────────────────────────────────────

class AutoScorer:
    def __init__(self):
        self.results = []
        self.category_scores = {
            'basic':      {'score': 0, 'max': 20},
            'complex':    {'score': 0, 'max': 20},
            'multi_turn': {'score': 0, 'max': 30},
            'performance':{'score': 0, 'max': 30},
        }
        self.timings = []
        self.constraint_violations_total = 0
        self.hallucination_total = 0
        self.first_tokens = []  # 首Token延迟列表

    def call_dialog(self, message: str, user_id: str = None,
                    user_ids: List[str] = None) -> Dict:
        payload = {"message": message}
        if user_id:
            payload["user_id"] = user_id
        if user_ids:
            payload["user_ids"] = user_ids

        t0 = time.perf_counter()
        try:
            resp = requests.post(DIALOG_URL, json=payload, timeout=60)
            t_total = (time.perf_counter() - t0) * 1000
            data = resp.json() if resp.status_code == 200 else {}
            data['_t_total'] = t_total
            # 提取服务端返回的首Token延迟
            timing = data.get('timing', {})
            data['_first_token_ms'] = timing.get('first_token_ms', 0)
            self.timings.append(t_total)
            ft = timing.get('first_token_ms', 0)
            if ft:
                self.first_tokens.append(ft)
            return data
        except Exception as e:
            t_total = (time.perf_counter() - t0) * 1000
            return {'success': False, 'error': str(e), '_t_total': t_total}

    def _get_recs(self, result: Dict) -> List[Dict]:
        return result.get('recommendations', [])

    def _get_response(self, result: Dict) -> str:
        return result.get('response', '')

    # ==================== (1) 基础推荐 20分 ====================
    def run_basic_tests(self):
        print("\n" + "=" * 60)
        print("(1) 基础推荐测试 (20分) — 硬约束满足率 + 菜谱真实性")
        print("    扣分规则: 每违反一项忌口/过敏扣5分, 幻觉菜品扣5分/项")
        print("=" * 60)

        tests = [
            {
                "name": "高血压用户(13)",
                "user_id": "13",
                "message": "推荐几道菜",
                "points": 3,
            },
            {
                "name": "高血糖用户(3)—高血压+高血糖",
                "user_id": "3",
                "message": "推荐几道菜",
                "points": 3,
            },
            {
                "name": "高尿酸用户(23)—尿酸482",
                "user_id": "23",
                "message": "推荐几道菜",
                "points": 3,
            },
            {
                "name": "海鲜过敏用户(1)",
                "user_id": "1",
                "message": "推荐几道菜",
                "points": 3,
            },
            {
                "name": "鸡蛋过敏用户(6)—海鲜+鸡蛋+哺乳期",
                "user_id": "6",
                "message": "推荐几道菜",
                "points": 3,
            },
            {
                "name": "孕妇用户(20)—12周",
                "user_id": "20",
                "message": "推荐几道菜",
                "points": 3,
            },
            {
                "name": "哺乳期用户(6)—多重过敏+哺乳",
                "user_id": "6",
                "message": "推荐一份营养午餐",
                "points": 2,
            },
        ]

        score = 0
        for test in tests:
            print(f"\n  [{test['name']}] — 满分{test['points']}分")
            result = self.call_dialog(test['message'], user_id=test['user_id'])
            recs = self._get_recs(result)
            t_ms = round(result.get('_t_total', 0), 0)
            earned = test['points']
            penalty_items = []

            if not recs:
                print(f"    - 无推荐（Agent反问中） | 0分 | {t_ms:.0f}ms")
                continue

            # 获取用户档案
            profile = USER_PROFILES.get(test['user_id'], {})

            # 逐道检查：硬约束违反 + 幻觉
            for r in recs:
                rname = r.get('name', '')
                if not rname:
                    continue

                # 幻觉检测
                if rname not in ALL_RECIPES:
                    penalty_items.append(f"幻觉: {rname}")
                    self.hallucination_total += 1
                    continue  # 幻觉菜品无法进一步做约束检查

                # 约束违反检测 — 用完整菜谱数据
                full_recipe = RECIPE_DATA.get(rname)
                if full_recipe and profile:
                    violations = check_constraint_violation(full_recipe, profile)
                    for v in violations:
                        penalty_items.append(f"{rname}: {v}")
                        self.constraint_violations_total += 1

            # 扣分：每违反一项扣5分
            deducted = len(penalty_items) * 5
            earned = max(0, earned - deducted)

            if penalty_items:
                print(f"    [警告] 违反{len(penalty_items)}项 → 扣{deducted}分 → 得{earned}分 | {t_ms:.0f}ms")
                for item in penalty_items[:5]:
                    print(f"       - {item}")
                if len(penalty_items) > 5:
                    print(f"       ... 还有{len(penalty_items) - 5}项")
            else:
                print(f"    [通过] 零违反 | {earned}分 | {t_ms:.0f}ms")

            score += earned

        self.category_scores['basic']['score'] = min(score, 20)
        print(f"\n  >> 基础推荐得分: {min(score, 20)}/20")
        print(f"     幻觉共计: {self.hallucination_total}项 | 约束违反: {self.constraint_violations_total}项")

    # ==================== (2) 复杂场景 20分 ====================
    def run_complex_tests(self):
        print("\n" + "=" * 60)
        print("(2) 复杂场景测试 (20分) — 多人约束 + 搭配合理性 + 营养均衡")
        print("=" * 60)

        tests = [
            {
                "name": "多人约束—高血压(2)+痛风(3)合并",
                "user_ids": ["2", "3"],
                "user_id": "eval_multi",
                "message": "三人晚餐，一位高血压一位痛风，推荐合适的菜",
                "points": 8,
                "check_diversity": True,
            },
            {
                "name": "荤素搭配+数量要求",
                "user_id": "eval_complex",
                "message": "推荐5道菜，要有荤有素有汤，家常口味",
                "min_count": 3,
                "points": 6,
                "check_ratio": True,
            },
            {
                "name": "烹饪方式多样性",
                "user_id": "eval_complex2",
                "message": "推荐4道菜，做法不要重复，有炒的有蒸的有炖的",
                "min_count": 3,
                "points": 6,
                "check_methods": True,
            },
        ]

        score = 0
        for test in tests:
            print(f"\n  [{test['name']}] — 满分{test['points']}分")
            kwargs = {"message": test['message']}
            if test.get('user_ids'):
                kwargs["user_ids"] = test['user_ids']
            if test.get('user_id'):
                kwargs["user_id"] = test['user_id']
            result = self.call_dialog(**kwargs)
            recs = self._get_recs(result)
            t_ms = round(result.get('_t_total', 0), 0)
            earned = test['points']
            deductions = 0
            notes = []

            # — 数量检查 —
            min_count = test.get('min_count', 0)
            if min_count and len(recs) < min_count:
                deductions += 2
                notes.append(f"菜品不足({len(recs)}<{min_count})扣2分")

            if not recs:
                print(f"    - 无推荐 | 0分 | {t_ms:.0f}ms")
                continue

            # — 幻觉检测 —
            with_names = []
            for r in recs:
                rname = r.get('name', '')
                if rname and rname in ALL_RECIPES:
                    with_names.append(rname)
                elif rname:
                    deductions += 1
                    notes.append(f"幻觉: {rname}")
                    self.hallucination_total += 1

            # — 多人约束检查 —
            if test.get('user_ids'):
                uid_list = test['user_ids']
                for uid in uid_list:
                    profile = USER_PROFILES.get(uid, {})
                    if profile:
                        for rname in with_names:
                            full = RECIPE_DATA.get(rname)
                            if full:
                                violations = check_constraint_violation(full, profile)
                                if violations:
                                    deductions += 1
                                    notes.append(f"用户{uid}违反: {violations[0]}")
                                    self.constraint_violations_total += 1

            # — 荤素比检查 —
            if test.get('check_ratio') and len(with_names) >= 2:
                meat_count = sum(1 for n in with_names
                                 if RECIPE_DATA.get(n) and is_meat_dish(RECIPE_DATA[n]))
                veg_count = len(with_names) - meat_count
                ratio = f"{meat_count}荤/{veg_count}素"
                if veg_count == 0:
                    deductions += 1
                    notes.append(f"无素菜({ratio})扣1分")
                elif meat_count == 0:
                    deductions += 1
                    notes.append(f"无荤菜({ratio})扣1分")
                else:
                    notes.append(f"荤素搭配 {ratio}")

            # — 烹饪方式多样性 —
            if test.get('check_methods') and len(with_names) >= 2:
                all_methods = set()
                for n in with_names:
                    full = RECIPE_DATA.get(n)
                    if full:
                        all_methods |= extract_cooking_methods(full)
                if len(all_methods) >= 3:
                    notes.append(f"烹饪方式{len(all_methods)}种")
                elif len(all_methods) >= 1:
                    deductions += 1
                    notes.append(f"烹饪方式仅{len(all_methods)}种扣1分")
                else:
                    deductions += 2
                    notes.append("无法识别烹饪方式扣2分")

            # — 冷热比检查 —
            if test.get('check_ratio') and len(with_names) >= 3:
                cold_count = sum(1 for n in with_names
                                 if RECIPE_DATA.get(n) and is_cold_dish(RECIPE_DATA[n]))
                hot_count = len(with_names) - cold_count
                if hot_count == len(with_names):
                    deductions += 1
                    notes.append(f"全热菜({hot_count}热/{cold_count}冷)扣1分")
                else:
                    notes.append(f"冷热搭配 {hot_count}热/{cold_count}冷")

            # — 按人营养均衡度（多人场景）—
            if test.get('user_ids') and NUTRITION_DB and len(with_names) >= 2:
                for uid in test['user_ids']:
                    total_cal = 0
                    total_protein = 0
                    for rname in with_names:
                        nut = estimate_recipe_nutrition(rname)
                        total_cal += nut['热量']
                        total_protein += nut['蛋白质']
                    per_cal = total_cal / len(test['user_ids'])
                    per_protein = total_protein / len(test['user_ids'])
                    notes.append(f"用户{uid}: ~{per_cal:.0f}kcal/餐, ~{per_protein:.0f}g蛋白")

            # — 营养概览输出验证（Agent是否在响应中主动提供营养数据）—
            if test.get('check_nutrition_output', True) and result.get('nutrition_summary'):
                notes.append("[通过] Agent已输出营养概览")

            earned = max(0, earned - deductions)
            score += earned

            status = "[通过]" if deductions == 0 else "[警告]"
            names_str = ', '.join(with_names[:4]) if with_names else '无推荐'
            extra = f" ({len(with_names)}道)" if len(with_names) > 4 else ""
            print(f"    {status} {names_str}{extra} | {earned}分 | {t_ms:.0f}ms")
            for note in notes:
                print(f"       {note}")

        self.category_scores['complex']['score'] = min(score, 20)
        print(f"\n  >> 复杂场景得分: {min(score, 20)}/20")

    # ==================== (3) 多轮交互 30分 ====================
    def run_multi_turn_tests(self):
        print("\n" + "=" * 60)
        print("(3) 多轮交互测试 (30分) — 上下文一致性 + 最小修改 + 交互自然度")
        print("=" * 60)

        session_id = f"eval_mt_{int(time.time())}"
        score = 0

        # — 场景A：完整多轮对话 —
        print("\n  [场景A] 完整推荐→追加约束→替换→确认 (满分15)")
        turns_a = [
            ("推荐3道晚餐菜", "init"),
            ("不要太油腻", "constraint"),
            ("把第一道菜换一个", "replace"),
        ]
        prev_recs = []
        first_recs = []
        a_score = 0

        for i, (msg, check_type) in enumerate(turns_a):
            result = self.call_dialog(msg, user_id=session_id)
            recs = self._get_recs(result)
            response = self._get_response(result)
            t_ms = round(result.get('_t_total', 0), 0)
            print(f"    轮{i+1}: \"{msg}\" → {response[:60]}... | {t_ms:.0f}ms")

            if i == 0:
                first_recs = recs

            if check_type == 'constraint' and prev_recs and recs:
                # 检查Agent是否在响应中提及保留/保留原有/不换/继续保持
                retention_words = ['保留', '保留原有', '不换', '继续保持', '不变', '维持',
                                   '符合', '继续', '依然推荐', '可以继续', '不用换']
                has_retention = any(kw in response for kw in retention_words)

                if has_retention:
                    a_score += 5
                    print(f"       [通过] 最小修改: 提及保留 (+5)")
                else:
                    print(f"       [警告] 未提及保留，可能全量替换")
            elif check_type == 'replace' and prev_recs and recs:
                # 检查是否明确说了"替换了/换掉X道"
                replace_words = ['替换', '替代', '换掉', '换成了', '改成', '换成',
                                 '换成别的', '改了一下', '调整']
                has_replace = any(kw in response for kw in replace_words)
                if has_replace:
                    a_score += 5
                    print(f"       [通过] 单道替换: 明确说明替换了某道菜 (+5)")
                else:
                    print(f"       [警告] 未提及替换，可能全量重推")

            prev_recs = recs if recs else prev_recs

        # 初始推荐质量检查（轮1结果）
        if first_recs:
            if len(first_recs) >= 3:
                a_score += 5
                print(f"       [通过] 初始推荐质量: ≥3道菜 (+5)")
            else:
                print(f"       [警告] 初始推荐不足3道")
        score += a_score

        # — 场景B：交互自然度 —
        print("\n  [场景B] 模糊需求→主动追问 (满分8)")
        session_b = f"eval_nat_{int(time.time())}"
        b_score = 0
        vague_turns = [
            ("今晚吃什么", "应反问偏好"),
            ("随便", "应追问具体需求"),
        ]
        for i, (msg, expected) in enumerate(vague_turns):
            result = self.call_dialog(msg, user_id=session_b)
            response = self._get_response(result)
            t_ms = round(result.get('_t_total', 0), 0)
            has_question = any(kw in response for kw in
                               ['？', '?', '什么', '偏好', '口味', '喜欢', '想吃什么', '忌口', '过敏'])
            if has_question:
                b_score += 4
                print(f"    轮{i+1}: \"{msg}\" → 主动追问 [通过] (+4) | {t_ms:.0f}ms")
            else:
                print(f"    轮{i+1}: \"{msg}\" → 未追问 [警告] | {t_ms:.0f}ms")
                print(f"       回复: {response[:80]}...")
        score += b_score

        # — 场景C：上下文一致性 —
        print("\n  [场景C] 上下文记忆测试 (满分7)")
        session_c = f"eval_ctx_{int(time.time())}"
        c_score = 0
        turns_c = [
            ("我对花生过敏，推荐午餐", "set_constraint"),
            ("有什么推荐的", "should_remember"),
        ]
        for i, (msg, check_type) in enumerate(turns_c):
            result = self.call_dialog(msg, user_id=session_c)
            response = self._get_response(result)
            recs = self._get_recs(result)
            t_ms = round(result.get('_t_total', 0), 0)
            if check_type == 'should_remember' and recs:
                risky = [r['name'] for r in recs if '花生' in r.get('name', '')]
                if not risky:
                    c_score += 7
                    print(f"    轮{i+1}: \"{msg}\" → 未遗忘花生过敏 [通过] (+7) | {t_ms:.0f}ms")
                else:
                    print(f"    轮{i+1}: \"{msg}\" → 违反忌口({risky}) [失败] | {t_ms:.0f}ms")
            else:
                print(f"    轮{i+1}: \"{msg}\" → {response[:60]}... | {t_ms:.0f}ms")
        score += c_score

        # — 场景D：需求矛盾检测 —
        print("\n  [场景D] 需求矛盾检测 — 素食→要肉 / 减肥→增肌 / 少辣→川菜 (满分5)")
        d_score = 0
        # D1: 素食 vs 肉类
        session_d1 = f"eval_contra1_{int(time.time())}"
        r = self.call_dialog("我是素食者，推荐午餐", user_id=session_d1)
        rsp1 = self._get_response(r)
        print(f"    轮1: \"我是素食者\" → {rsp1[:50]}...")
        r2 = self.call_dialog("我想吃红烧肉", user_id=session_d1)
        rsp2 = self._get_response(r2)
        has_alert = any(kw in rsp2 for kw in ['素食', '矛盾', '冲突', '之前', '改为', '调整为'])
        if has_alert:
            d_score += 2
            print(f"    轮2: \"红烧肉\" → 检测到素食矛盾 [通过] (+2)")
        else:
            print(f"    轮2: \"红烧肉\" → 未检测矛盾 [警告]")
            print(f"         回复: {rsp2[:80]}...")
        # D2: 减肥 vs 增肌
        session_d2 = f"eval_contra2_{int(time.time())}"
        r = self.call_dialog("我在减肥，推荐午饭", user_id=session_d2)
        print(f"    轮1: \"减肥\" → {self._get_response(r)[:50]}...")
        r2 = self.call_dialog("我要增肌", user_id=session_d2)
        rsp = self._get_response(r2)
        has_alert = any(kw in rsp for kw in ['减肥', '增肌', '矛盾', '冲突', '选择', '目标'])
        if has_alert:
            d_score += 2
            print(f"    轮2: \"增肌\" → 检测到减肥/增肌矛盾 [通过] (+2)")
        else:
            print(f"    轮2: \"增肌\" → 未检测矛盾 [警告]")
        # D3: 少辣 vs 川菜
        session_d3 = f"eval_contra3_{int(time.time())}"
        r = self.call_dialog("不要辣的，推荐几道菜", user_id=session_d3)
        print(f"    轮1: \"不要辣\" → {self._get_response(r)[:50]}...")
        r2 = self.call_dialog("来点川菜", user_id=session_d3)
        rsp = self._get_response(r2)
        has_alert = any(kw in rsp for kw in ['不要辣', '少辣', '不辣', '辣', '之前', '川菜属'])
        if has_alert:
            d_score += 1
            print(f"    轮2: \"川菜\" → 检测到辣度矛盾 [通过] (+1)")
        else:
            print(f"    轮2: \"川菜\" → 未检测矛盾 [警告]")
        score += d_score

        # — 场景E：否定推荐后追加具体要求 —
        print("\n  [场景E] 否定→给理由→Agent重新推荐 (满分5)")
        session_e = f"eval_reject_{int(time.time())}"
        e_score = 0
        r = self.call_dialog("推荐3道晚餐", user_id=session_e)
        rsp1 = self._get_response(r)
        recs1 = self._get_recs(r)
        print(f"    轮1: \"推荐3道晚餐\" → {rsp1[:50]}... ({len(recs1)}道)")
        r2 = self.call_dialog("不好吃，太清淡了，想吃口味重一点的", user_id=session_e)
        rsp2 = self._get_response(r2)
        recs2 = self._get_recs(r2)
        t_ms = round(r2.get('_t_total', 0), 0)
        # 检查是否理解了"太清淡→口味重"的含义
        understands_reason = any(kw in rsp2 for kw in ['重口味', '重口', '浓郁', '下饭', '红烧', '麻辣', '香辣', '换', '调整'])
        if understands_reason and recs2:
            e_score += 3
            print(f"    轮2: \"太清淡→口味重\" → Agent理解了拒绝原因 [通过] (+3) | {t_ms:.0f}ms")
        else:
            print(f"    轮2: \"太清淡→口味重\" → 未理解原因 [警告] | {t_ms:.0f}ms")
            print(f"         回复: {rsp2[:80]}...")
        # 检查新推荐是否和旧推荐不同
        if recs1 and recs2:
            old_names = {r.get('name','') for r in recs1}
            new_names = {r.get('name','') for r in recs2}
            overlap = old_names & new_names
            if len(overlap) < len(new_names) * 0.5:  # 至少换了一半
                e_score += 2
                print(f"       [通过] 推荐大幅更新 (重叠{len(overlap)}/{len(new_names)}道) (+2)")
            else:
                print(f"       [警告] 推荐变化不大 (重叠{len(overlap)}/{len(new_names)}道)")
        score += e_score

        # — 场景F：多人场景——隐含冲突 —
        print("\n  [场景F] 多人隐含冲突：三人口味对立 (满分5)")
        session_f = f"eval_familyconflict_{int(time.time())}"
        f_score = 0
        # 用预置用户档案：高血压(2) + 痛风(3)
        r = self.call_dialog("推荐4人晚餐", user_id=session_f, user_ids=["2", "3"])
        rsp = self._get_response(r)
        recs = self._get_recs(r)
        t_ms = round(r.get('_t_total', 0), 0)
        if recs:
            # 检查是否有荤素搭配意识
            f_score += 2
            print(f"    多人→有推荐结果 ({len(recs)}道) (+2) | {t_ms:.0f}ms")
        else:
            print(f"    多人→无推荐，Agent在反问 | {t_ms:.0f}ms")
        # 检查是否在回复中提到了多人的约束考虑
        if any(kw in rsp for kw in ['高血压', '痛风', '低盐', '低嘌呤', '兼顾', '平衡', '不同']):
            f_score += 3
            print(f"       [通过] 回复体现了多人约束意识 (+3)")
            print(f"         回复: {rsp[:100]}...")
        else:
            print(f"       [警告] 回复未体现多人约束考虑")
        score += f_score

        self.category_scores['multi_turn']['score'] = min(score, 30)
        print(f"\n  >> 多轮交互得分: {min(score, 30)}/30")
        print(f"     场景A(最小修改)={a_score}/15  场景B(交互自然)={b_score}/8  场景C(上下文)={c_score}/7  场景D(需求矛盾)={d_score}/5")
        print(f"     场景E(否定理解)={e_score}/5  场景F(多人冲突)={f_score}/5")

    # ==================== (4) 性能效率 30分 ====================
    def run_performance_eval(self):
        print("\n" + "=" * 60)
        print("(4) 性能效率测试 (30分)")
        print("=" * 60)

        if not self.timings:
            print("  无耗时数据，跳过")
            return

        sorted_t = sorted(self.timings)
        n = len(sorted_t)

        def pct(p):
            idx = int(n * p / 100)
            return round(sorted_t[min(idx, n - 1)], 0)

        avg_total = round(sum(self.timings) / n, 0)
        p50 = pct(50)
        p95 = pct(95)

        print(f"  请求数: {n}")
        print(f"  平均总耗时: {avg_total}ms")
        print(f"  P50: {p50}ms  P95: {p95}ms")

        score = 0

        # 首Token延迟 (10分) — 优秀<2s 合格<5s 超时0分
        if self.first_tokens:
            ft_avg = round(sum(self.first_tokens) / len(self.first_tokens), 0)
            ft_p50 = sorted(self.first_tokens)[len(self.first_tokens)//2] if self.first_tokens else 0
            print(f"\n  首Token延迟 (Agent首次LLM调用): 平均{ft_avg}ms, P50{ft_p50}ms")
            if ft_avg < 2000:
                ft_score = 10
                ft_level = "优秀 (<2s)"
            elif ft_avg < 5000:
                ft_score = 7
                ft_level = "合格 (2~5s)"
            else:
                ft_score = 3
                ft_level = "超时 (≥5s)"
        else:
            print(f"\n  首Token延迟: 无Agent调用数据（回退路径不可测）")
            ft_score = 7
            ft_level = "接口限制扣3分"
        score += ft_score
        print(f"    得分: {ft_score}/10（{ft_level}）")

        # 端到端响应 (10分) — 优秀<8s 合格<15s 超时0分
        if avg_total < 8000:
            e2e_score = 10
            e2e_level = "优秀 (<8s)"
        elif avg_total < 15000:
            e2e_score = 7
            e2e_level = "合格 (8~15s)"
        else:
            e2e_score = 0
            e2e_level = "超时 (≥15s)"
        score += e2e_score
        print(f"  端到端响应: {avg_total}ms → {e2e_level} ({e2e_score}/10)")

        # 多轮平均 (10分) — 优秀<6s 合格<12s 超时0分
        if p50 < 6000:
            mt_score = 10
            mt_level = "优秀 (<6s)"
        elif p50 < 12000:
            mt_score = 7
            mt_level = "合格 (6~12s)"
        else:
            mt_score = 0
            mt_level = "超时 (≥12s)"
        score += mt_score
        print(f"  多轮平均(P50): {p50}ms → {mt_level} ({mt_score}/10)")

        self.category_scores['performance']['score'] = min(score, 30)
        print(f"\n  >> 性能效率得分: {min(score, 30)}/30")

    # ==================== (5) 对话用例全覆盖 ====================
    def run_dialog_cases(self, cases_file: str = '对话用例.json'):
        print("\n" + "=" * 60)
        print("(5) 对话用例全覆盖测试 — 25组多轮对话场景")
        print("=" * 60)

        try:
            with open(cases_file, 'r', encoding='utf-8') as f:
                cases = json.load(f)
        except Exception as e:
            print(f"  无法加载 {cases_file}: {e}")
            return

        total_turns = 0
        passed = 0
        failed = 0
        empty_responses = 0
        case_timings = []

        for case in cases:
            case_id = case['id']
            turns = case['user_messages']
            session_id = f"eval_case_{case_id}_{int(time.time())}"
            case_ms = 0
            case_hallucinations = 0
            case_ok = True

            print(f"\n  [用例{case_id}] {turns[0][:40]}... ({len(turns)}轮)")

            for i, msg in enumerate(turns):
                result = self.call_dialog(msg, user_id=session_id)
                response = self._get_response(result)
                recs = self._get_recs(result)
                t_ms = round(result.get('_t_total', 0), 0)
                case_ms += t_ms
                total_turns += 1

                has_answer = len(response.strip()) > 0
                status = "[通过]" if has_answer else "[失败]"

                if not has_answer:
                    empty_responses += 1
                    case_ok = False

                # 幻觉检测
                hinfo = ""
                if recs:
                    fake = [r['name'] for r in recs
                            if r.get('name') and r['name'] not in ALL_RECIPES]
                    if fake:
                        hinfo = f" [幻觉:{len(fake)}]"
                        case_hallucinations += len(fake)
                        self.hallucination_total += len(fake)
                        case_ok = False

                rec_info = f" → {len(recs)}道" if recs else ""
                truncated = response[:50].replace('\n', ' ')
                print(f"    轮{i+1}: {status} {truncated}...{rec_info}{hinfo} | {t_ms:.0f}ms")

            case_timings.append(case_ms)
            if case_ok:
                passed += 1
                print(f"    >> 通过 [通过] ({case_ms:.0f}ms)")
            else:
                failed += 1
                print(f"    >> 未通过 [警告] ({case_ms:.0f}ms, 幻觉{case_hallucinations}项)")

        avg_case = sum(case_timings) / len(case_timings) if case_timings else 0
        print(f"\n  ── 对话用例汇总 ──")
        print(f"  总用例: {len(cases)}  通过: {passed}  未通过: {failed}")
        print(f"  总轮次: {total_turns}  空响应: {empty_responses}")
        print(f"  幻觉总计: {self.hallucination_total}项")
        print(f"  平均每用例耗时: {avg_case:.0f}ms")
        return {'total': len(cases), 'passed': passed, 'failed': failed,
                'empty_responses': empty_responses, 'avg_ms': avg_case}

    # ==================== 汇总 ====================
    def print_summary(self):
        print("\n" + "=" * 60)
        print("            竞 赛 评 分 汇 总")
        print("=" * 60)

        auto_total = 0
        for cat, info in self.category_scores.items():
            s = info['score']
            m = info['max']
            bar = '█' * int(s / m * 20) if m > 0 else ''
            auto_total += s
            labels = {
                'basic': '(1)基础推荐', 'complex': '(2)复杂场景',
                'multi_turn': '(3)多轮交互', 'performance': '(4)性能效率'
            }
            print(f"  {labels[cat]:14s}  {bar:<20s}  {s:2d}/{m:2d}")

        print(f"  {'─' * 44}")
        print(f"  {'自动评分小计':14s}  {'':20s}  {auto_total:2d}/100")

        # 专家评审（40%）提示
        print(f"\n  [警告] 注意：以上为自动评分部分（占验收60%）。")
        print(f"  专家评审（占验收40%）需人工评估：")
        print(f"    - 菜品搭配合理性")
        print(f"    - 交互质量")
        print(f"    - 推荐理由解释能力")
        print(f"  最终得分 = 自动评分×60% + 专家评审×40%")

        # 折算60%
        auto_60 = round(auto_total * 0.6, 1)
        print(f"\n  自动评分折算: {auto_total} × 60% = {auto_60}/60")
        print("=" * 60)
        return auto_total


def check_health() -> bool:
    try:
        resp = requests.get(f"{BASE_URL}/api/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


if __name__ == '__main__':
    print("方太膳食规划Agent — 竞赛自动评分")
    print(f"菜谱库: {len(ALL_RECIPES)} 道")
    print(f"用户档案: {len(USER_PROFILES)} 人")
    print(f"服务地址: {BASE_URL}")

    if not check_health():
        print("\n[FATAL] 服务未启动，请先运行 python app.py")
        print("       在项目根目录执行: python app.py")
        sys.exit(1)

    scorer = AutoScorer()

    scorer.run_basic_tests()
    scorer.run_complex_tests()
    scorer.run_multi_turn_tests()
    scorer.run_performance_eval()
    scorer.run_dialog_cases()

    scorer.print_summary()
