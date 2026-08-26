"""
营养缺口检测与膳食规划模块

根据本餐（人均）实际营养摄入，与用户的目标日需营养（RDI）对比，
检测蛋白质 / 热量 / 脂肪 / 碳水等关键营养的缺口或超标，
并给出可执行的增补 / 替代建议，帮助用户把"一餐"真正规划到符合身体目标。

原则：
- 一餐通常约占全天摄入的 1/3（晚/午餐），本文档用 meal_share 折算每餐参考值。
- 若提供用户档案（年龄 / 体重 / 目标），优先按真实对象估算；否则退回通用成人参考值。
- 只做"提醒 + 可执行建议"，不做诊断性断言，措辞保持温和。
"""

from typing import Dict, List, Optional

# 通用成人日需参考（用于未提供档案时的回退评估）
RDI_DEFAULT = {
    '热量_kcal': 2000,
    '蛋白质_g': 65,
    '脂肪_g': 60,
    '碳水_g': 250,
}

# 每餐约占全天摄入的比例（午/晚餐按 1/3 折算）
MEAL_SHARE = 0.35


def _to_float(value) -> Optional[float]:
    """把可能的字符串数字（'体重(kg)': '70kg' / '70' / 70）转为 float，失败返回 None。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace('kg', '').replace('公斤', '').strip())
    except (TypeError, ValueError):
        return None


def estimate_rdi(profile: Optional[Dict]) -> Dict:
    """
    根据用户档案估算每日营养需量。
    - 能以体重估算蛋白质（约 1.2g/kg）、热量（约 30 kcal/kg）；
    - 目标里若含"增肌 / 减脂 / 控糖"等，对该项做微调；
    - 无有效档案时回退通用参考值。
    """
    rdi = dict(RDI_DEFAULT)
    if profile:
        weight = _to_float(profile.get('体重(kg)', profile.get('weight')))
        if weight and weight > 20:
            rdi['热量_kcal'] = round(weight * 30, 0)          # 约 30 kcal/kg/日
            rdi['蛋白质_g'] = round(weight * 1.2, 0)          # 约 1.2 g/kg/日
        goals = ' '.join(str(profile.get('目标', profile.get('dietary_goals', []))))
        if not isinstance(goals, str):
            goals = str(goals)
        if '增肌' in goals and weight:
            rdi['蛋白质_g'] = round(weight * 1.6, 0)
        if '减脂' in goals:
            rdi['热量_kcal'] = round(rdi['热量_kcal'] * 0.85, 0)
        # 蛋白/脂/碳占比回退：蛋白 0.13, 脂 0.27, 碳 0.60（大致基于热量推算）
        kcal = rdi['热量_kcal']
        rdi['蛋白质_g'] = max(rdi['蛋白质_g'], round(kcal * 0.13 / 4, 0))
        rdi['脂肪_g'] = round(kcal * 0.27 / 9, 0)
        rdi['碳水_g'] = round(kcal * 0.60 / 4, 0)
    return rdi


def _gap_hint(key: str, actual: float, target: float) -> Optional[str]:
    """单个营养素的实际 vs 目标，返回一句话缺口/超标提示；在合理区间内返回 None。"""
    if target <= 0:
        return None
    ratio = actual / target
    name = {'热量_kcal': '热量', '蛋白质_g': '蛋白质', '脂肪_g': '脂肪', '碳水_g': '碳水'}[key]
    if ratio < 0.8:
        gap = round(target - actual)
        return f"{name}偏低（约差 {gap}g/kcal）"
    if ratio > 1.35:
        extra = round(actual - target)
        return f"{name}偏高（约多 {extra}g/kcal）"
    return None


def _remedy_for(short_keys: List[str], full_keys: List[str]) -> List[str]:
    """针对缺失/超标的营养素，给出可执行的食物调整建议。"""
    suggestions = []
    if '蛋白质_g' in short_keys:
        suggestions.append("可加一份优质蛋白：鸡蛋、虾仁、去皮鸡腿、豆腐或豆浆")
    if '碳水_g' in short_keys:
        suggestions.append("可补一份主食：杂粮饭、玉米、红薯或全麦馒头")
    if '热量_kcal' in short_keys and '脂肪_g' not in full_keys:
        suggestions.append("可适量增加健康油脂：坚果、牛油果或橄榄油拌菜")
    if '脂肪_g' in full_keys:
        suggestions.append("少用煎炸，改蒸煮炖，肉可去皮")
    if '碳水_g' in full_keys:
        suggestions.append("主食减半，替换为更多蔬菜来控制碳水")
    if '热量_kcal' in full_keys:
        suggestions.append("整体份量可略减一档")
    # 去重
    seen = set()
    uniq = [s for s in suggestions if not (s in seen or seen.add(s))]
    return uniq[:4]


def plan_nutrition_gaps(per_person_totals: Dict,
                        user_profiles: Optional[List[Dict]] = None,
                        meal_share: float = MEAL_SHARE) -> str:
    """
    对比本餐人均摄入与目标每餐参考值，输出营养缺口检测与规划建议。

    Args:
        per_person_totals: {'热量': float, '蛋白质': float, '脂肪': float, '碳水': float}
            人均一餐实际摄入（单位：kcal / g）。
        user_profiles: 可选，用餐人档案列表；用于按真实对象估算 RDI。
        meal_share: 一餐占全天比例，默认 0.35。

    Returns:
        一段 markdown 文本；若无可评估营养则返回空串。
    """
    if not per_person_totals or not any(per_person_totals.values()):
        return ""

    actual_map = {
        '热量_kcal': per_person_totals.get('热量', 0) or 0,
        '蛋白质_g': per_person_totals.get('蛋白质', 0) or 0,
        '脂肪_g': per_person_totals.get('脂肪', 0) or 0,
        '碳水_g': per_person_totals.get('碳水', 0) or 0,
    }

    # 计算目标每餐参考值：若有多人，取均值档案估算
    if user_profiles:
        rdi = {'热量_kcal': 0, '蛋白质_g': 0, '脂肪_g': 0, '碳水_g': 0}
        n = 0
        for prof in user_profiles:
            if not prof:
                continue
            est = estimate_rdi(prof)
            for k in rdi:
                rdi[k] += est.get(k, 0)
            n += 1
        if n > 0:
            rdi = {k: v / n for k, v in rdi.items()}
        else:
            rdi = estimate_rdi(None)
    else:
        rdi = estimate_rdi(None)

    target_map = {k: (v * meal_share) for k, v in rdi.items()}

    short_keys = []
    full_keys = []
    lines = []
    for key, target in target_map.items():
        hint = _gap_hint(key, actual_map[key], target)
        if not hint:
            continue
        name = {'热量_kcal': '热量', '蛋白质_g': '蛋白质', '脂肪_g': '脂肪', '碳水_g': '碳水'}[key]
        unit = 'kcal' if '热量' in name else 'g'
        lines.append(
            f"- {name}目标约 {round(target)}{unit}，本餐 {round(actual_map[key])}{unit}，{hint.split('（')[0]}。"
        )
        if '偏高' in hint:
            full_keys.append(key)
        else:
            short_keys.append(key)

    if not lines and not short_keys and not full_keys:
        return ""

    block = ["", "**营养规划**", ""]
    if lines:
        block.append("本餐营养缺口检测：")
        block.extend(lines)
    remedies = _remedy_for(short_keys, full_keys)
    if remedies:
        block.append("可这样微调：")
        block.extend(f"  {i+1}. {t}" for i, t in enumerate(remedies))
    return "\n".join(block)