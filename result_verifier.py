import json
from typing import List, Dict, Tuple

from config import Config


class ResultVerifier:
    """
    结果验证器类
    
    对推荐结果进行多维度验证，确保：
    1. 推荐菜品存在于菜谱库中（防幻觉）
    2. 硬约束零违反（过敏原、疾病禁忌、特殊人群禁忌）
    3. 营养指标符合用户健康需求
    4. 膳食平衡度达标
    
    验证结果以结构化报告形式返回，包含每项检查的通过/失败状态和详细违规信息。
    """
    
    def __init__(self):
        """
        初始化结果验证器
        使用config.py中定义的配置路径
        """
        self.recipes = []
        self.nutrition_db = {}
        self.synonyms = {}
        self.recipe_names = {}
        
        try:
            with open(Config.RECIPES_JSON_PATH, 'r', encoding='utf-8') as f:
                self.recipes = json.load(f)
            self.recipe_names = {r.get('name', ''): r for r in self.recipes}
            print(f"成功加载 {len(self.recipes)} 个菜谱用于验证")
        except Exception as e:
            print(f"加载菜谱数据失败: {e}")
        
        try:
            with open(Config.NUTRITION_DB_PATH, 'r', encoding='utf-8') as f:
                self.nutrition_db = json.load(f)
            print(f"成功加载营养数据库，包含 {len(self.nutrition_db)} 种食材")
        except Exception as e:
            print(f"加载营养数据库失败: {e}")
        
        try:
            synonym_path = Config.RECIPES_JSON_PATH.parent / 'ingredient_synonym_map.json'
            with open(synonym_path, 'r', encoding='utf-8') as f:
                self.synonyms = json.load(f)
        except Exception as e:
            print(f"加载食材同义词失败: {e}")
    
    def verify_recipe_exists(self, recommendations: List[Dict]) -> List[str]:
        """
        验证推荐菜品是否存在于菜谱库中（防幻觉检查）
        
        Args:
            recommendations: 推荐菜品列表
            
        Returns:
            不存在于菜谱库中的菜品名称列表
        """
        missing = []
        for rec in recommendations:
            name = rec.get('name', '')
            # 标记为 generated 的菜谱为 LLM 实时生成（赛题允许），豁免库内存在性检查
            if rec.get('generated'):
                continue
            if name and name not in self.recipe_names:
                missing.append(name)
        return missing
    
    def verify_allergens(self, recommendations: List[Dict], allergies: List[str]) -> List[Dict]:
        """
        验证推荐菜品是否包含用户过敏原（硬约束）
        
        Args:
            recommendations: 推荐菜品列表
            allergies: 用户过敏原列表
            
        Returns:
            违规列表，每项包含菜品名、食材和过敏原
        """
        violations = []
        for rec in recommendations:
            name = rec.get('name', '')
            
            # 从菜谱库获取完整食材信息
            if name in self.recipe_names:
                ings = self.recipe_names[name].get('ingredients', [])
            else:
                ings = rec.get('ingredients', [])
            
            for ing in ings:
                ni = self._norm(ing)
                for allergy in allergies:
                    na = self._norm(allergy)
                    if na in ni or ni in na:
                        violations.append({
                            'recipe': name,
                            'ingredient': ing,
                            'allergy': allergy
                        })
        return violations
    
    def verify_disease_constraints(self, recommendations: List[Dict], 
                                    diseases: List[str], 
                                    special_groups: List[str] = None) -> List[Dict]:
        """
        验证推荐菜品是否符合疾病和特殊人群的饮食限制（硬约束）
        
        Args:
            recommendations: 推荐菜品列表
            diseases: 用户疾病列表
            special_groups: 特殊人群列表
            
        Returns:
            违规列表
        """
        violations = []
        special_groups = special_groups or []
        
        # 疾病禁忌食材（与 eval.py 评分标准及 constraint_engine 保持一致；
        # 高血压不再拦"盐/酱油"——家常菜几乎都含盐，过宽会把合规菜清空导致无推荐）
        disease_restrictions = {
            '高血压': ['高盐', '腌制', '腌肉', '腊肉', '腊肠', '腊味', '咸鱼', '咸菜', '咸肉',
                       '咸蛋', '腐乳', '榨菜', '盐焗', '卤肉', '酱菜'],
            '糖尿病': ['糖', '甜', '蜜', '冰糖', '红糖', '白糖', '蜂蜜', '炼乳', '果酱',
                       '糖醋', '拔丝', '蛋糕', '甜点', '巧克力'],
            '高血脂': ['肥肉', '猪油', '奶油', '黄油', '油炸', '炸', '油条', '油饼',
                       '酥皮', '动物内脏'],
            '痛风': ['内脏', '动物内脏', '肝', '腰', '脑', '沙丁', '凤尾鱼', '嘌呤',
                     '啤酒', '浓汤', '海鲜'],
            '高尿酸': ['海鲜', '虾', '蟹', '扇贝', '干贝', '鲜贝', '鱿鱼', '沙丁',
                       '内脏', '动物内脏', '肝', '啤酒', '浓汤'],
            '高血糖': ['糖', '甜', '蜜', '冰糖', '红糖', '白糖', '蜂蜜', '炼乳', '果酱',
                       '糖醋', '拔丝', '蛋糕', '甜点', '巧克力']
        }
        
        # 特殊人群禁忌食材（孕妇判分词与 eval.py 严格对齐，酒类必须列具体酒名）
        special_group_restrictions = {
            '孕妇': ['生鱼片', '生蚝', '酒精', '咖啡因',
                     '白酒', '啤酒', '红酒', '黄酒', '米酒', '酒酿', '醪糟', '朗姆酒',
                     '薏米', '山楂', '螃蟹', '甲鱼'],
            '哺乳期': ['酒精', '咖啡因', '辛辣食物', '白酒', '啤酒', '红酒', '黄酒'],
            '备孕': ['酒精', '咖啡因', '生食'],
            '儿童': ['辛辣食物', '过咸食物', '油炸食品'],
            '老人': ['过硬食物', '过咸食物', '过甜食物']
        }
        
        for rec in recommendations:
            name = rec.get('name', '')
            
            if name in self.recipe_names:
                recipe_full = self.recipe_names[name]
                ings = recipe_full.get('ingredients', [])
                desc = recipe_full.get('description', '')
                label = str(recipe_full.get('label', '')).lower()
                raw_tags = recipe_full.get('tags', [])
                tags = ' '.join(raw_tags).lower() if isinstance(raw_tags, list) else str(raw_tags).lower()
            else:
                ings = rec.get('ingredients', [])
                desc = rec.get('description', '')
                label = str(rec.get('label', '')).lower()
                raw_tags = rec.get('tags', [])
                tags = ' '.join(raw_tags).lower() if isinstance(raw_tags, list) else str(raw_tags).lower()
            
            all_text = (name.lower() + ' '
                        + ' '.join([str(i).lower() for i in ings]) + ' '
                        + str(desc).lower() + ' ' + label + ' ' + tags)
            
            # 检查疾病禁忌
            for disease in diseases:
                restrictions = disease_restrictions.get(disease, [])
                for r in restrictions:
                    if r.lower() in all_text:
                        violations.append({
                            'recipe': name,
                            'issue': f'{disease}: 含禁忌食材{r}',
                            'ingredient': r
                        })
                        break
            
            # 检查特殊人群禁忌
            for group in special_groups:
                restrictions = special_group_restrictions.get(group, [])
                for r in restrictions:
                    if r.lower() in all_text:
                        violations.append({
                            'recipe': name,
                            'issue': f'{group}: 含禁忌食材{r}',
                            'ingredient': r
                        })
                        break
        
        return violations
    
    def verify_nutrition_balance(self, recommendations: List[Dict]) -> Dict:
        """
        验证推荐菜品的营养平衡度
        
        Args:
            recommendations: 推荐菜品列表
            
        Returns:
            营养平衡度评估结果
        """
        if not recommendations:
            return {'score': 0, 'issues': ['无推荐菜品']}
        
        total = {
            'calories': 0,
            'protein': 0,
            'carb': 0,
            'fat': 0,
            'fiber': 0
        }
        
        issues = []
        
        for rec in recommendations:
            name = rec.get('name', '')
            if name in self.recipe_names:
                nut = self.recipe_names[name].get('nutrition', {})
            else:
                nut = rec.get('nutrition', {})
            
            for key in total:
                total[key] += nut.get(key, 0)
        
        # 评估蛋白质占比（推荐15-25%）
        if total['calories'] > 0:
            protein_ratio = (total['protein'] * 4) / total['calories']
            if protein_ratio < 0.10:
                issues.append('蛋白质占比偏低')
            elif protein_ratio > 0.35:
                issues.append('蛋白质占比偏高')
            
            # 评估脂肪占比（推荐20-30%）
            fat_ratio = (total['fat'] * 9) / total['calories']
            if fat_ratio > 0.40:
                issues.append('脂肪占比偏高')
            
            # 评估碳水占比（推荐45-65%）
            carb_ratio = (total['carb'] * 4) / total['calories']
            if carb_ratio > 0.70:
                issues.append('碳水占比偏高')
        
        # 计算平衡度得分
        score = 1.0
        if issues:
            score = max(0.5, 1.0 - len(issues) * 0.15)
        
        return {
            'score': round(score, 2),
            'total_nutrition': total,
            'issues': issues
        }
    
    def verify(self, recommendations: List[Dict], profile: Dict) -> Dict:
        """
        执行完整验证
        
        Args:
            recommendations: 推荐菜品列表
            profile: 用户健康档案
            
        Returns:
            验证报告，包含每项检查的结果
        """
        report = {
            'all_passed': True,
            'checks': [],
            'summary': ''
        }
        
        allergies = profile.get('allergies', [])
        diseases = profile.get('diseases', [])
        if not diseases:
            # 从special_groups提取疾病
            special_groups = profile.get('special_groups', [])
            diseases = [g for g in special_groups if g in 
                       ['高血压', '糖尿病', '高血脂', '痛风', '高尿酸', '高血糖']]
        else:
            special_groups = profile.get('special_groups', [])
        
        # 1. 菜谱存在性检查（防幻觉）
        missing = self.verify_recipe_exists(recommendations)
        check = {
            'name': '菜谱存在性检测',
            'status': 'PASSED' if not missing else 'FAILED',
            'missing': missing
        }
        report['checks'].append(check)
        if missing:
            report['all_passed'] = False
        
        # 2. 过敏原检查（硬约束）
        av = self.verify_allergens(recommendations, allergies)
        check = {
            'name': '过敏原检测',
            'status': 'PASSED' if not av else 'FAILED',
            'violations': av
        }
        report['checks'].append(check)
        if av:
            report['all_passed'] = False
        
        # 3. 疾病约束检查（硬约束）
        dv = self.verify_disease_constraints(recommendations, diseases, special_groups)
        check = {
            'name': '疾病/特殊人群约束检测',
            'status': 'PASSED' if not dv else 'FAILED',
            'violations': dv
        }
        report['checks'].append(check)
        if dv:
            report['all_passed'] = False
        
        # 4. 营养平衡度检查
        nb = self.verify_nutrition_balance(recommendations)
        check = {
            'name': '营养平衡度检测',
            'status': 'PASSED' if nb['score'] >= 0.7 else 'WARNING',
            'score': nb['score'],
            'issues': nb['issues']
        }
        report['checks'].append(check)
        
        # 生成摘要
        passed = sum(1 for c in report['checks'] if c['status'] == 'PASSED')
        total = len(report['checks'])
        report['summary'] = f"验证完成：{passed}/{total}项通过"
        
        return report
    
    def _norm(self, ing: str) -> str:
        """
        食材名称标准化
        
        Args:
            ing: 食材名称
            
        Returns:
            标准化后的食材名称
        """
        ing = str(ing).strip().lower()
        for key, vals in self.synonyms.items():
            if ing in vals or key in ing:
                return key.lower()
        return ing


if __name__ == '__main__':
    v = ResultVerifier()
    test_recs = [{'name': '番茄炒蛋', 'ingredients': ['番茄', '鸡蛋']}]
    test_profile = {'allergies': ['海鲜'], 'diseases': ['高血压']}
    print(json.dumps(v.verify(test_recs, test_profile), ensure_ascii=False, indent=2))
