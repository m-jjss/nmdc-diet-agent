from typing import List, Dict, Optional, Tuple
import re
import json


class DialogManager:
    """
    对话管理器类
    
    负责维护用户对话上下文，支持多轮对话和上下文理解。
    
    核心功能：
    - 对话历史管理：记录用户和系统的对话消息
    - 上下文提取：从对话历史中提取用户偏好和需求
    - 对话状态追踪：管理对话的当前状态
    - 意图识别：分析用户当前意图
    - 约束追加/修改：支持多轮对话中逐步细化约束
    - 方案否定：支持用户否定之前的推荐并要求重新推荐
    - 上下文回溯：支持用户引用之前的对话内容
    """
    
    def __init__(self):
        """
        初始化对话管理器
        """
        # 对话历史
        self.history: List[Dict] = []
        
        # 用户提取的偏好信息
        self.user_preferences: Dict = {
            'allergies': [],
            'preferences': {},
            'dietary_goals': [],
            'excluded_ingredients': [],  # 用户明确不吃的食材
            'cooking_time_limit': None,   # 烹饪时间限制（分钟）
            'servings': None,             # 用餐人数
            'meal_type': None,            # 餐次：早餐/午餐/晚餐/夜宵
            'cuisine_preference': None,   # 菜系偏好
            'difficulty': None            # 制作难度：简单/中等/复杂
        }
        
        # 当前对话状态
        self.dialog_state: str = 'initial'
        
        # 对话轮数计数器
        self.turn_count: int = 0
        
        # 已推荐的菜品列表（用于避免重复推荐或支持方案否定）
        self.recommended_recipes: List[str] = []
        
        # 用户否定的菜品列表
        self.rejected_recipes: List[str] = []
        
        # 当前轮次提取的约束变更类型
        self.last_action: str = None
        
        # 新用户引导状态
        self.onboarding_step: int = 0  # 0=未开始, 1=问过敏, 2=问偏好, 3=问健康, 4=完成
        self.is_onboarding: bool = False
        self.onboarding_data: Dict = {
            'allergies': [],
            'taste': '',
            'health': '',
            'excluded': [],
        }
        
        # 性能统计
        self.timing_history: List[Dict] = []
    
    def add_message(self, role: str, content: str):
        """
        添加对话消息
        
        将用户或系统消息添加到对话历史中。
        
        Args:
            role: 消息角色，可选值: 'user', 'system', 'assistant'
            content: 消息内容
        """
        self.history.append({
            'role': role,
            'content': content
        })
        self.turn_count += 1
    
    def get_context(self) -> List[Dict]:
        """
        获取对话上下文
        
        返回完整的对话历史，用于LLM理解上下文。
        
        Returns:
            对话历史消息列表
        """
        return self.history
    
    def _extract_preferences_keyword(self, user_message: str) -> Dict:
        """
        从用户消息中提取偏好信息
        
        通过关键词匹配识别用户的饮食偏好、过敏信息、膳食目标、
        烹饪时间限制、用餐人数、餐次等。
        
        Args:
            user_message: 用户消息文本
            
        Returns:
            提取的偏好信息
        """
        extracted = {
            'allergies': [],
            'preferences': {},
            'dietary_goals': [],
            'excluded_ingredients': [],
            'cooking_time_limit': None,
            'servings': None,
            'meal_type': None,
            'cuisine_preference': None,
            'difficulty': None
        }
        
        message_lower = user_message.lower()
        
        # 识别过敏信息
        allergy_keywords = {
            '花生': ['花生', '花生酱'],
            '海鲜': ['海鲜', '虾', '蟹', '鱼'],
            '鸡蛋': ['鸡蛋', '蛋'],
            '牛奶': ['牛奶', '奶酪'],
            '大豆': ['大豆', '豆腐', '豆浆'],
            '坚果': ['坚果', '核桃', '杏仁']
        }
        
        for allergy, keywords in allergy_keywords.items():
            for keyword in keywords:
                if keyword in message_lower and ('过敏' in message_lower or '不吃' in message_lower or '不要' in message_lower):
                    if allergy not in extracted['allergies']:
                        extracted['allergies'].append(allergy)
                    break
        
        # 识别明确排除的食材
        exclude_patterns = [
            r'不要(.+?)(?:[，。,]|$)',
            r'不吃(.+?)(?:[，。,]|$)',
            r'别放(.+?)(?:[，。,]|$)',
            r'没有(.+?)(?:[，。,]|$)',
            r'家里只有(.+?)(?:[，。,]|$)'
        ]
        
        for pattern in exclude_patterns[:3]:  # 前3个是排除食材
            matches = re.findall(pattern, user_message)
            for match in matches:
                ingredients = re.split(r'[、和与还有]', match.strip())
                for ing in ingredients:
                    ing = ing.strip()
                    if ing and len(ing) <= 10 and ing not in extracted['excluded_ingredients']:
                        extracted['excluded_ingredients'].append(ing)
        
        # 识别饮食偏好
        preference_keywords = {
            'vegetarian': ['素食', '不吃肉', '素菜'],
            'low_fat': ['低脂', '少油', '清淡'],
            'low_spicy': ['不辣', '少辣', '微辣', '别做辣的', '不要辣', '别辣', '不吃辣', '别太辣', '不要太辣'],
            'low_sugar': ['低糖', '无糖', '别太甜', '不要太甜'],
            'light': ['清淡']
        }
        
        for pref, keywords in preference_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    extracted['preferences'][pref] = True
                    break
        
        # 额外：排除食材中含"辣的"也应触发 low_spicy
        if not extracted['preferences'].get('low_spicy'):
            for ing in extracted.get('excluded_ingredients', []):
                if '辣' in ing:
                    extracted['preferences']['low_spicy'] = True
                    break
        
        # 识别膳食目标
        goal_keywords = {
            '减肥': ['减肥', '减脂', '瘦身', '减重'],
            '增肌': ['增肌', '健身', '高蛋白'],
            '养生': ['养生', '健康', '滋补', '补气血', '补气', '补血'],
            '控糖': ['控糖', '糖尿病'],
            '暖胃': ['暖胃', '养胃', '胃不好']
        }
        
        for goal, keywords in goal_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    if goal not in extracted['dietary_goals']:
                        extracted['dietary_goals'].append(goal)
                    break
        
        # 识别烹饪时间限制
        time_patterns = [
            r'(\d+)\s*分钟',
            r'(\d+)\s*分钟能搞定',
            r'半小时内',
            r'(\d+)分钟内能搞定',
            r'(\d+)分钟搞定'
        ]
        for pattern in time_patterns:
            match = re.search(pattern, user_message)
            if match:
                if '半小时' in pattern:
                    extracted['cooking_time_limit'] = 30
                else:
                    extracted['cooking_time_limit'] = int(match.group(1))
                break
        
        # 识别用餐人数
        serving_patterns = [
            r'(\d+)\s*个人',
            r'(\d+)\s*人吃',
            r'一家(\d+)口',
            r'(\d+)人份'
        ]
        for pattern in serving_patterns:
            match = re.search(pattern, user_message)
            if match:
                extracted['servings'] = int(match.group(1))
                break
        
        # 识别餐次
        meal_keywords = {
            '早餐': ['早餐', '早点', '早饭', '早上吃'],
            '午餐': ['午餐', '午饭', '中午吃'],
            '晚餐': ['晚餐', '晚饭', '晚上吃', '今晚'],
            '夜宵': ['夜宵', '宵夜', '晚上饿了']
        }
        for meal, keywords in meal_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    extracted['meal_type'] = meal
                    break
            if extracted['meal_type']:
                break
        
        # 识别制作难度
        if any(kw in message_lower for kw in ['简单', '快速', '方便', '快手']):
            extracted['difficulty'] = '简单'
        elif any(kw in message_lower for kw in ['复杂', '难做', '正式', '仪式感']):
            extracted['difficulty'] = '复杂'
        
        # 识别菜系偏好（仅在非否定语境中匹配）
        cuisine_keywords = {
            '川菜': ['川菜', '麻辣'],
            '粤菜': ['粤菜', '广东'],
            '江浙菜': ['江浙', '本帮', '上海菜'],
            '家常菜': ['家常', '家里做', '家常便饭']
        }
        # 单独处理"辣"：只有明确想要辣(而非不要辣)时才匹配川菜
        if '辣' in message_lower:
            if not re.search(r'(不要|不吃|别|不想要|不想吃).*辣', message_lower):
                extracted['cuisine_preference'] = '川菜'
        
        # 更新用户偏好
        self._update_preferences(extracted)
        
        return extracted
    
    def _update_preferences(self, new_preferences: Dict):
        """
        更新用户偏好信息
        
        将新提取的偏好信息合并到已存储的用户偏好中。
        支持约束追加（增量更新）和约束修改（覆盖更新）。
        
        Args:
            new_preferences: 新提取的偏好信息
        """
        # 合并过敏信息（追加，不覆盖）
        for allergy in new_preferences.get('allergies', []):
            if allergy not in self.user_preferences['allergies']:
                self.user_preferences['allergies'].append(allergy)
        
        # 合并偏好设置（覆盖更新）
        self.user_preferences['preferences'].update(new_preferences.get('preferences', {}))
        
        # 合并膳食目标（追加）
        for goal in new_preferences.get('dietary_goals', []):
            if goal not in self.user_preferences['dietary_goals']:
                self.user_preferences['dietary_goals'].append(goal)
        
        # 合并排除食材（追加）
        for ing in new_preferences.get('excluded_ingredients', []):
            if ing not in self.user_preferences['excluded_ingredients']:
                self.user_preferences['excluded_ingredients'].append(ing)
        
        # 更新烹饪时间限制（覆盖）
        if new_preferences.get('cooking_time_limit') is not None:
            self.user_preferences['cooking_time_limit'] = new_preferences['cooking_time_limit']
        
        # 更新用餐人数（覆盖）
        if new_preferences.get('servings') is not None:
            self.user_preferences['servings'] = new_preferences['servings']
        
        # 更新餐次（覆盖）
        if new_preferences.get('meal_type') is not None:
            self.user_preferences['meal_type'] = new_preferences['meal_type']
        
        # 更新菜系偏好（覆盖）
        if new_preferences.get('cuisine_preference') is not None:
            self.user_preferences['cuisine_preference'] = new_preferences['cuisine_preference']
        
        # 更新制作难度（覆盖）
        if new_preferences.get('difficulty') is not None:
            self.user_preferences['difficulty'] = new_preferences['difficulty']
    
    def _detect_intent_keyword(self, user_message: str) -> str:
        """
        检测用户意图
        
        通过关键词匹配识别用户的当前意图。
        
        支持的意图类型：
        - recommend: 请求菜谱推荐
        - set_preferences: 设置饮食偏好
        - modify_preferences: 修改饮食偏好
        - add_constraint: 追加约束条件
        - reject_recommendation: 否定推荐方案
        - ask_nutrition: 询问营养信息
        - confirm: 确认推荐
        - cancel: 取消操作
        - greet: 问候
        - ask_recipe_detail: 询问菜品详情
        - request_substitute: 请求替换某道菜
        - request_more: 请求更多推荐
        
        Args:
            user_message: 用户消息文本
            
        Returns:
            识别出的意图类型
        """
        message_lower = user_message.lower()
        
        # 意图关键词映射（按优先级排序）
        intent_keywords = {
            'greet': ['你好', '嗨', '您好', '早上好', '晚上好', '在吗'],
            'cancel': ['取消', '不要了', '重来', '重新开始', '算了', '不选了'],
            'confirm': ['好的', '确认', '就要这个', '可以', '不错', '挺好', '就这个', '确定'],
            'reject_recommendation': ['不要这个', '换一个', '不喜欢', '不要这道', '换掉', '不好吃', '不太行'],
            'request_substitute': ['替换', '换个', '换掉.*菜', '把.*换成', '换成.*菜'],
            'request_more': ['再来一个', '再推荐', '多一些', '还有吗', '再多几个', '再来几个'],
            'ask_nutrition': ['热量', '卡路里', '成分', '多少卡', '营养成分', '营养.*多少', '营养.*吗', '营养.*如何', '有什么营养'],
            'ask_recipe_detail': ['怎么做', '做法', '步骤', '食材', '配料', '详细', '步骤是什么'],
            'set_preferences': ['设置', '偏好', '过敏', '不吃', '不要'],
            'modify_preferences': ['修改', '更改', '重新设置', '调整'],
            'add_constraint': ['别太', '不要', '清淡', '不要太', '少放', '加个', '再来个', '还要', '最好'],
            'recommend': ['推荐', '吃什么', '菜谱', '菜品', '午餐', '晚餐', '早餐', '夜宵', '安排', '帮我想']
        }
        
        # 特殊模式检测（提高识别准确率）
        # 检测方案否定（排除替换意图）
        if re.search(r'(整个|全部|所有).*不要|都不要|都不好|都不喜欢', message_lower):
            return 'reject_recommendation'
        
        # 检测替换请求（优先于否定）
        if re.search(r'(替换|换成|换掉|把.*换成|换.*成|把.*换掉|换一道)', message_lower):
            # 检查是否明确提到替换某道菜
            for recipe_name in self.recommended_recipes:
                if recipe_name in message_lower:
                    return 'request_substitute'
            # 如果有推荐历史但没明确提到菜品名，也视为替换请求
            if self.recommended_recipes:
                return 'request_substitute'
        
        # 检测否定推荐（检查是否提到具体菜品）
        if self.recommended_recipes:
            for recipe_name in self.recommended_recipes:
                if recipe_name in message_lower:
                    if re.search(rf'(不要.*{recipe_name}|{recipe_name}.*不要|不喜欢.*{recipe_name}|{recipe_name}.*不好)', message_lower):
                        return 'reject_recommendation'
        
        # 检测追加约束（在已有推荐的情况下）
        if self.recommended_recipes and self.turn_count > 0:
            # 排除明确的替换和否定意图
            if not re.search(r'(替换|换成|换掉)', message_lower):
                constraint_keywords = ['别太', '不要太', '不要', '不吃', '清淡', '少油', '少盐', '少辣', '不辣', '别辣', '别放',
                                       '再加', '还要', '最好', '偏', '改成', '换成口味', '换个口味', '换个风格']
                if any(kw in message_lower for kw in constraint_keywords):
                    # 确保不是替换整桌菜的情况
                    if not re.search(r'(全部|所有|整个|整套).*不要', message_lower):
                        return 'add_constraint'
        
        # 检测请求更多
        if re.search(r'(再来|再推荐|多一些|还有吗|再多)', message_lower):
            return 'request_more'
        
        # 按优先级检测意图
        # 0. 先检查模糊查询（优先级最高，避免被 confirm 等截胡）
        vague_keywords_full = ['随便', '都行', '都可以', '无所谓', '随便推荐', '随便几个', '什么都行', '看着办',
                               '吃点好的', '好吃的', '啥好吃', '有什么', '推荐一下', '来看看', '看看', '给点建议']
        if any(kw in message_lower for kw in vague_keywords_full):
            if not any(kw in message_lower for kw in ['辣', '素', '肉', '鱼', '鸡', '汤', '减肥', '餐', '不要辣']):
                return 'vague_query'
        
        for intent, keywords in intent_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    # 特殊处理：避免"不要辣"被误识别为set_preferences
                    if intent == 'set_preferences' and self.recommended_recipes and re.search(r'别太|不要太|清淡|少油', message_lower):
                        continue
                    # reject_recommendation 只在有推荐历史时触发
                    if intent == 'reject_recommendation' and not self.recommended_recipes:
                        continue
                    return intent
        
        # 检测模糊查询（在推荐之前）
        vague_keywords = ['随便', '都行', '都可以', '无所谓', '随便推荐', '随便几个', '什么都行', '看着办']
        if any(kw in message_lower for kw in vague_keywords):
            return 'vague_query'
        
        # 默认意图为推荐
        return 'recommend'
    
    def is_vague_query(self, user_message: str) -> bool:
        """
        检测是否为模糊查询（需要引导用户澄清偏好）
        
        Args:
            user_message: 用户消息文本
            
        Returns:
            是否需要澄清
        """
        message_lower = user_message.strip().lower()
        
        # 极短消息（<=4个字）且无明确需求关键词
        if len(user_message.strip()) <= 4:
            specific_keywords = ['辣', '素', '肉', '鱼', '鸡', '猪', '牛', '羊', '汤', '面', '饭', '蒸', '炒',
                                '减肥', '增肌', '控糖', '过敏', '不吃', '清淡', '少油', '少盐']
            if not any(kw in message_lower for kw in specific_keywords):
                return True
        
        # 明显的模糊关键词
        vague_keywords = ['随便', '都行', '都可以', '无所谓', '随便推荐', '随便几个', '什么都行', '看着办',
                         '吃点好的', '好吃的', '啥好吃', '有什么', '推荐一下', '来看看', '看看', '给点建议']
        if any(kw in message_lower for kw in vague_keywords):
            if not any(kw in message_lower for kw in ['辣', '素', '肉', '鱼', '鸡', '汤', '减肥', '餐']):
                return True
        
        return False
    
    def generate_clarification_question(self, user_message: str) -> str:
        """
        生成澄清问题，引导用户明确偏好
        
        Args:
            user_message: 用户原始消息
            
        Returns:
            澄清引导语
        """
        # 先检测缺失信息
        missing = self.detect_missing_info(user_message)
        
        if missing:
            questions = []
            if 'meal_type' in missing:
                questions.append('想吃哪一餐？早餐、午餐还是晚餐？')
            if 'servings' in missing:
                questions.append('几个人一起吃呀？')
            if 'cooking_time' in missing:
                questions.append('赶时间吗？比如半小时内要搞定？')
            if 'style' in missing:
                questions.append('想要什么风格？家常小炒、清淡养生还是丰盛一点？')
            if missing == ['none']:
                return ("嗯，你的偏好还比较模糊，多告诉我一些吧～\n"
                        "比如：想吃什么餐？几个人？喜欢什么口味？")
            if questions:
                return '还差一点点信息：\n' + '\n'.join(f'  - {q}' for q in questions)
        
        if self.turn_count <= 1:
            return ("好的！为了更好地为您推荐，可以告诉我您的偏好吗？\n"
                    "比如：\n"
                    "  - 口味偏好：清淡/少辣/素食\n"
                    "  - 膳食目标：减肥/增肌/控糖\n"
                    "  - 用餐场景：早餐/午餐/晚餐/多人聚餐\n"
                    "  - 烹饪时间：简单的/不限制\n\n"
                    '您也可以直接说"今晚想吃清淡的晚餐"哦～')
        else:
            return ("我理解您想要一些新选择。可以多告诉我一些偏好吗？\n"
                    "比如：想吃什么口味？有没有忌口食材？\n"
                    "这样我能给出更精准的推荐～")
    
    def detect_missing_info(self, user_message: str) -> list:
        """
        检测当前对话中缺少的关键信息
        
        Args:
            user_message: 用户消息
            
        Returns:
            缺失信息列表，如 ['meal_type', 'servings']
        """
        missing = []
        msg = user_message.lower()
        prefs = self.user_preferences
        
        # 餐次
        has_meal = prefs.get('meal_type') is not None
        if not has_meal and not any(k in msg for k in ['早餐', '午餐', '晚饭', '晚餐', '夜宵', '中午', '早上', '宵夜', '早饭']):
            # 检查历史消息
            has_meal_history = any(
                any(k in (m.get('content','') if isinstance(m,dict) else str(m)) 
                    for k in ['早餐', '午餐', '晚饭', '晚餐', '夜宵', '中午', '早上'])
                for m in self.history
            )
            if not has_meal_history:
                missing.append('meal_type')
        
        # 人数
        has_servings = prefs.get('servings') is not None
        has_people_msg = any(k in msg for k in ['人', '个', '自己', '聚餐', '家庭', '两人', '三人', '一人'])
        if not has_servings and not has_people_msg:
            missing.append('servings')
        
        # 烹饪时间（只在用户提到了"快/简单/赶时间"这类词但没有具体时间时追问）
        needs_time = any(k in msg for k in ['快', '简单', '容易', '方便', '速'])
        has_time = prefs.get('cooking_time_limit') is not None
        if needs_time and not has_time and 'cooking_time' not in missing:
            missing.append('cooking_time')
        
        # 整体：如果以上都缺且没有明确风格，追问
        if len(missing) >= 2 and not any(k in msg for k in ['辣', '清淡', '重口', '随便', '推荐']):
            missing.append('style')
        
        return missing if missing else ['none']
    
    def get_user_profile(self) -> Dict:
        """
        获取用户档案
        
        将提取的偏好信息转换为标准用户档案格式。
        
        Returns:
            标准用户档案格式，包含过敏、偏好和膳食目标
        """
        return {
            'allergies': self.user_preferences['allergies'],
            'preferences': self.user_preferences['preferences'],
            'dietary_goals': self.user_preferences['dietary_goals'],
            'excluded_ingredients': self.user_preferences['excluded_ingredients'],
            'cooking_time_limit': self.user_preferences['cooking_time_limit'],
            'servings': self.user_preferences['servings'],
            'meal_type': self.user_preferences['meal_type'],
            'cuisine_preference': self.user_preferences['cuisine_preference'],
            'difficulty': self.user_preferences['difficulty'],
            'dialog_turns': self.turn_count,
            'rejected_recipes': self.rejected_recipes
        }
    
    def add_recommended_recipes(self, recipe_names: List[str]):
        """
        记录已推荐的菜品
        
        Args:
            recipe_names: 推荐的菜品名称列表
        """
        for name in recipe_names:
            if name not in self.recommended_recipes:
                self.recommended_recipes.append(name)
    
    def add_rejected_recipe(self, recipe_name: str):
        """
        记录用户否定的菜品
        
        Args:
            recipe_name: 被否定的菜品名称
        """
        if recipe_name not in self.rejected_recipes:
            self.rejected_recipes.append(recipe_name)
    
    def get_excluded_recipe_names(self) -> List[str]:
        """
        获取需要排除的菜品名称列表（已推荐且被否定的）
        
        Returns:
            需要排除的菜品名称列表
        """
        return self.rejected_recipes.copy()
    
    def get_search_filters(self) -> Dict:
        """
        根据用户偏好生成检索过滤条件
        
        Returns:
            检索过滤条件字典
        """
        filters = {}
        
        # 排除食材
        excluded = []
        excluded.extend(self.user_preferences.get('allergies', []))
        excluded.extend(self.user_preferences.get('excluded_ingredients', []))
        if excluded:
            filters['exclude_ingredients'] = excluded
        
        # 排除已否定的菜品
        if self.rejected_recipes:
            filters['exclude_recipes'] = self.rejected_recipes.copy()
        
        # 口味偏好
        prefs = self.user_preferences.get('preferences', {})
        if prefs.get('light') or prefs.get('low_fat'):
            filters['low_fat'] = True
        if prefs.get('low_spicy'):
            filters['low_spicy'] = True
        if prefs.get('low_sugar'):
            filters['low_sugar'] = True
        if prefs.get('vegetarian'):
            filters['vegetarian'] = True
        
        # 烹饪时间限制
        if self.user_preferences.get('cooking_time_limit'):
            filters['max_cooking_time'] = self.user_preferences['cooking_time_limit']
        
        # 用餐人数
        if self.user_preferences.get('servings'):
            filters['servings'] = self.user_preferences['servings']
        
        # 餐次
        if self.user_preferences.get('meal_type'):
            filters['meal_type'] = self.user_preferences['meal_type']
        
        return filters
    
    def resolve_reference(self, user_message: str) -> str:
        """
        上下文回溯：解析用户对之前对话内容的引用
        
        支持的引用类型：
        - 代词引用：那个菜、这个菜、它、那个、这个
        - 位置引用：第一个、第二个、最后一个、上次推荐的
        - 模糊引用：刚才推荐的、之前说的、之前的那个
        
        Args:
            user_message: 用户消息文本
            
        Returns:
            解析后的消息文本（替换引用为具体菜品名称）
        """
        message = user_message
        
        # 如果没有推荐历史，直接返回原消息
        if not self.recommended_recipes:
            return message
        
        # 获取最近推荐的菜品列表（按时间顺序）
        recent_recipes = self.recommended_recipes[-5:]  # 最多保留最近5个
        
        # 代词引用替换
        pronoun_patterns = [
            (r'(那个菜|这个菜)', '最近推荐的菜品'),
            (r'(那个|这个)', '最近推荐的'),
            (r'(它)', '这道菜')
        ]
        
        for pattern, replacement in pronoun_patterns:
            if re.search(pattern, message):
                # 如果只有一个推荐，直接替换为具体名称
                if len(recent_recipes) == 1:
                    message = re.sub(pattern, recent_recipes[0], message)
                else:
                    message = re.sub(pattern, replacement, message)
        
        # 位置引用替换
        position_patterns = [
            (r'第(一|1)[个]?', 0),
            (r'第(二|2)[个]?', 1),
            (r'第(三|3)[个]?', 2),
            (r'第(四|4)[个]?', 3),
            (r'第(五|5)[个]?', 4),
            (r'最后[一]?个', -1),
            (r'上[一]?个', -2)
        ]
        
        for pattern, index in position_patterns:
            match = re.search(pattern, message)
            if match:
                if 0 <= index < len(recent_recipes):
                    message = message.replace(match.group(0), recent_recipes[index])
                elif index < 0 and len(recent_recipes) >= abs(index):
                    message = message.replace(match.group(0), recent_recipes[index])
        
        # 模糊引用替换
        if '刚才' in message or '之前' in message:
            if len(recent_recipes) > 0:
                # 尝试提取具体菜品名称
                for recipe in recent_recipes:
                    if recipe in message:
                        break
                else:
                    # 如果没有具体菜品，添加最近推荐的菜品到消息中
                    message += f"（最近推荐的菜品：{', '.join(recent_recipes)}）"
        
        return message
    
    def detect_contradiction(self, new_preferences: Dict) -> Optional[str]:
        """
        检测需求矛盾
        
        检查新提取的偏好是否与已有偏好矛盾。
        只有当用户明确表达了相反的偏好时才触发矛盾检测，
        避免在模糊查询（如"随便推荐"）时误报矛盾。
        
        Args:
            new_preferences: 新提取的偏好信息
            
        Returns:
            矛盾描述（如果有矛盾），否则返回None
        """
        contradictions = []
        
        current_prefs = self.user_preferences.get('preferences', {})
        new_prefs = new_preferences.get('preferences', {})
        
        # 只有当新偏好非空且包含明确的相反设置时才检测矛盾
        has_new_preferences = bool(new_prefs) or bool(new_preferences.get('dietary_goals'))
        
        if not has_new_preferences:
            return None
        
        # 素食 vs 非素食（只有用户明确提到要吃肉时才检测）
        if current_prefs.get('vegetarian') and 'vegetarian' in new_prefs and not new_prefs['vegetarian']:
            contradictions.append("您之前设置为素食，现在似乎想要非素食菜品？")
        if new_prefs.get('vegetarian') and not current_prefs.get('vegetarian'):
            if self.recommended_recipes:
                contradictions.append("您之前接受了含肉类的推荐，现在设置为素食，将重新为您推荐。")
        
        # 清淡 vs 重口味（只有用户明确提到要辣/油腻时才检测）
        spicy_keywords_in_new = any(k in new_preferences.get('excluded_ingredients', []) for k in ['不辣', '少辣'])
        if current_prefs.get('low_spicy') and not new_prefs.get('low_spicy') and not spicy_keywords_in_new:
            has_spicy_intent = any(k in new_preferences.get('excluded_ingredients', []) for k in ['辣'])
            if has_spicy_intent or new_preferences.get('cuisine_preference') == '川菜':
                contradictions.append("您之前要求少辣，现在似乎想要辣一点的菜品？")
        if current_prefs.get('low_fat') and not new_prefs.get('low_fat'):
            contradictions.append("您之前要求低脂，现在似乎想要口味重一些的菜品？")
        if current_prefs.get('low_sugar') and not new_prefs.get('low_sugar'):
            contradictions.append("您之前要求低糖，现在似乎想要甜一些的菜品？")
        
        # 膳食目标矛盾
        current_goals = set(self.user_preferences.get('dietary_goals', []))
        new_goals = set(new_preferences.get('dietary_goals', []))
        
        if '减肥' in current_goals and '增肌' in new_goals:
            contradictions.append("减肥和增肌的饮食需求有所冲突，建议选择其中一个主要目标。")
        if '增肌' in current_goals and '减肥' in new_goals:
            contradictions.append("增肌和减肥的饮食需求有所冲突，建议选择其中一个主要目标。")
        if '控糖' in current_goals and '增肌' in new_goals:
            contradictions.append("控糖和增肌的饮食需求有所冲突，需要平衡碳水摄入。")
        
        # 过敏信息矛盾
        current_allergies = set(self.user_preferences.get('allergies', []))
        new_allergies = set(new_preferences.get('allergies', []))
        if new_allergies - current_allergies:
            new_allergy_items = ', '.join(new_allergies - current_allergies)
            contradictions.append(f"您新添加了过敏信息：{new_allergy_items}，将重新为您推荐。")
        
        # 烹饪时间矛盾
        current_time = self.user_preferences.get('cooking_time_limit')
        new_time = new_preferences.get('cooking_time_limit')
        if current_time and new_time and new_time < current_time:
            contradictions.append(f"您之前要求{current_time}分钟内完成，现在要求{new_time}分钟，时间更紧迫了。")
        
        # 用餐人数矛盾
        current_servings = self.user_preferences.get('servings')
        new_servings = new_preferences.get('servings')
        if current_servings and new_servings and new_servings != current_servings:
            contradictions.append(f"您之前设置{current_servings}人用餐，现在改为{new_servings}人。")
        
        # 餐次矛盾
        current_meal = self.user_preferences.get('meal_type')
        new_meal = new_preferences.get('meal_type')
        if current_meal and new_meal and current_meal != new_meal:
            contradictions.append(f"您之前要求{current_meal}，现在改为{new_meal}。")
        
        # 排除食材与已推荐菜品矛盾
        new_excluded = set(new_preferences.get('excluded_ingredients', []))
        if new_excluded and self.recommended_recipes:
            excluded_in_recommended = []
            for recipe_name in self.recommended_recipes:
                excluded_in_recommended.append(recipe_name)
            if excluded_in_recommended:
                contradictions.append(f"您新排除的食材可能影响之前推荐的菜品，将重新为您推荐。")
        
        return '; '.join(contradictions) if contradictions else None
    
    def get_stable_recipes(self, excluded_recipes: List[str]) -> List[str]:
        """
        获取稳定菜品列表（最小化修改原则）
        
        当用户只否定部分菜品时，保留未被否定的菜品，只替换被否定的部分。
        
        Args:
            excluded_recipes: 需要排除的菜品名称列表
            
        Returns:
            可以保留的稳定菜品列表
        """
        excluded_set = set(excluded_recipes)
        return [r for r in self.recommended_recipes if r not in excluded_set]
    
    def reset_dialog(self):
        """
        重置对话状态
        
        清空对话历史和用户偏好，恢复初始状态。
        """
        self.history = []
        self.user_preferences = {
            'allergies': [],
            'preferences': {},
            'dietary_goals': [],
            'excluded_ingredients': [],
            'cooking_time_limit': None,
            'servings': None,
            'meal_type': None,
            'cuisine_preference': None,
            'difficulty': None
        }
        self.dialog_state = 'initial'
        self.turn_count = 0
        self.recommended_recipes = []
        self.rejected_recipes = []
        self.last_action = None
    
    def should_confirm(self) -> bool:
        """
        判断是否需要确认
        
        根据对话状态和轮数判断是否需要用户确认推荐。
        
        Returns:
            True: 需要确认
            False: 不需要确认
        """
        return self.turn_count >= 3 and self.dialog_state == 'initial'
    
    def set_dialog_state(self, state: str):
        """
        设置对话状态
        
        Args:
            state: 对话状态，可选值: 'initial', 'setting_preferences', 'confirming', 'completed'
        """
        self.dialog_state = state
    
    def get_context_summary(self) -> str:
        """
        获取对话上下文摘要
        
        将对话历史和用户偏好合并为一段文字摘要，用于传递给LLM。
        
        Returns:
            对话上下文摘要文本
        """
        parts = []
        
        # 用户偏好摘要
        prefs = self.user_preferences
        if prefs['allergies']:
            parts.append(f"过敏: {', '.join(prefs['allergies'])}")
        if prefs['excluded_ingredients']:
            parts.append(f"不吃: {', '.join(prefs['excluded_ingredients'])}")
        if prefs['dietary_goals']:
            parts.append(f"目标: {', '.join(prefs['dietary_goals'])}")
        if prefs['cooking_time_limit']:
            parts.append(f"时间限制: {prefs['cooking_time_limit']}分钟")
        if prefs['servings']:
            parts.append(f"人数: {prefs['servings']}人")
        if prefs['meal_type']:
            parts.append(f"餐次: {prefs['meal_type']}")
        if prefs['difficulty']:
            parts.append(f"难度: {prefs['difficulty']}")
        
        pref_str = prefs.get('preferences', {})
        if pref_str.get('low_spicy'):
            parts.append("口味: 少辣")
        if pref_str.get('low_fat') or pref_str.get('light'):
            parts.append("口味: 清淡")
        if pref_str.get('low_sugar'):
            parts.append("口味: 低糖")
        if pref_str.get('vegetarian'):
            parts.append("饮食: 素食")
        
        if self.rejected_recipes:
            parts.append(f"已否定: {', '.join(self.rejected_recipes)}")
        
        return '; '.join(parts) if parts else '无特殊约束'
    
    def start_onboarding(self):
        """开始新用户引导流程"""
        self.onboarding_step = 1
        self.is_onboarding = True
        return '嗨！我是方太膳食规划助手～在给你推荐菜之前，先简单了解一下你的口味和身体状况。\n\n你对什么食物过敏吗？比如花生、海鲜、牛奶这些，没有的话说「无」就行～'
    
    def get_onboarding_question(self) -> str:
        """根据当前引导阶段返回问题"""
        questions = {
            2: '口味方面呢？偏辣、偏甜、清淡、重口味……你更喜欢哪种？',
            3: '身体方面有什么需要特别注意的吗？比如在减肥、控糖、高血压之类的，没有就说「无」～',
        }
        return questions.get(self.onboarding_step, "")
    
    def process_onboarding(self, user_message: str) -> str:
        """处理用户的引导回复，返回下一步问题或完成提示"""
        msg = user_message.strip()
        
        if self.onboarding_step == 1:
            # 收集过敏信息
            if msg not in ('无', '没有', '没', '无过敏', '不过敏', '没有过敏'):
                self.onboarding_data['allergies'] = [a.strip() for a in msg.replace('过敏', '').replace('、', ',').replace('，', ',').split(',') if a.strip()]
                self.user_preferences['allergies'] = self.onboarding_data['allergies']
            self.onboarding_step = 2  # 下一步问口味
            
        elif self.onboarding_step == 2:
            # 收集口味偏好
            if msg not in ('无', '没有', '没'):
                self.onboarding_data['taste'] = msg
                prefs = self.user_preferences.get('preferences', {})
                if any(k in msg for k in ['辣', '麻辣', '香辣']):
                    prefs['low_spicy'] = False
                elif any(k in msg for k in ['清淡', '不辣', '少油', '少盐']):
                    prefs['low_fat'] = True
                elif any(k in msg for k in ['甜', '酸甜']):
                    prefs['low_sugar'] = False
                elif any(k in msg for k in ['素食', '吃素', '不吃肉']):
                    prefs['vegetarian'] = True
                self.user_preferences['preferences'] = prefs
            self.onboarding_step = 3  # 下一步问健康
            
        elif self.onboarding_step == 3:
            # 收集健康情况
            if msg not in ('无', '没有', '没'):
                self.onboarding_data['health'] = msg
                for goal in ['减肥', '减脂', '瘦身', '增肌', '控糖', '降血压', '降尿酸', '补钙', '补铁', '补气血']:
                    if goal in msg:
                        self.user_preferences['dietary_goals'].append(goal)
            self.onboarding_step = 4  # 引导完成
            
        if self.onboarding_step == 4:
            # 引导完成
            self.is_onboarding = False
            parts = []
            if self.onboarding_data['allergies']:
                parts.append(f"过敏: {', '.join(self.onboarding_data['allergies'])}")
            if self.onboarding_data['taste']:
                parts.append(f"口味: {self.onboarding_data['taste']}")
            if self.onboarding_data['health']:
                parts.append(f"健康: {self.onboarding_data['health']}")
            
            summary = '; '.join(parts) if parts else '无特殊要求'
            return f"收到！都记下来了（{summary}）。现在可以开始给你推荐了——想吃什么？比如晚餐、午餐、清淡的或者半小时能搞定的？"
        else:
            return self.get_onboarding_question()



    def detect_intent(self, user_message: str, llm_client=None) -> str:
        """检测用户意图（LLM驱动，关键词回退）"""
        if llm_client is not None:
            try:
                result = self._detect_intent_llm(user_message, llm_client)
                if result:
                    return result
            except Exception as e:
                print(f'[WARN] LLM意图识别失败，回退关键词: {e}')
        return self._detect_intent_keyword(user_message)

    def extract_preferences(self, user_message: str, llm_client=None) -> Dict:
        """提取偏好（LLM驱动，关键词回退）"""
        if llm_client is not None:
            try:
                result = self._detect_with_llm(user_message, llm_client)
                if result and result.get('preferences'):
                    self._update_preferences(result['preferences'])
                    return result['preferences']
            except Exception as e:
                print(f'[WARN] LLM偏好提取失败，回退关键词: {e}')
        return self._extract_preferences_keyword(user_message)

    def detect_with_llm(self, user_message: str, llm_client) -> Optional[Dict]:
        """一次LLM调用同时返回意图和偏好"""
        return self._detect_with_llm(user_message, llm_client)

    def _detect_with_llm(self, user_message: str, llm_client) -> Optional[Dict]:
        """LLM驱动的意图识别+偏好提取"""
        prompt = f"""你是膳食规划Agent的意图识别模块。分析用户消息，严格只返回JSON。

<用户消息>
{user_message}

<当前状态>
已推荐菜品: {self.recommended_recipes[-5:] if self.recommended_recipes else '无'}
已拒绝菜品: {self.rejected_recipes if self.rejected_recipes else '无'}
对话轮数: {self.turn_count}
已有偏好: {json.dumps(self.user_preferences, ensure_ascii=False)}

<意图分类>
- recommend: 请求推荐菜品
- set_preference: 设置/声明偏好（如"我是素食者"）
- add_constraint: 在已有推荐基础上追加约束（如"不要太辣"）
- reject: 否定当前推荐（如"不要这个"）
- replace: 请求替换某道菜（如"把红烧肉换成清蒸鱼"）
- ask_more: 请求更多推荐
- ask_detail: 询问菜品详情/做法/营养
- confirm: 确认推荐
- cancel: 取消/重置
- greet: 问候
- vague: 模糊查询（如"随便"）
- clarify: 用户消息信息不足，需要反问

<偏好提取>
从消息中提取用户偏好：
- allergies: 过敏原列表
- excluded_ingredients: 不吃/不要的食材列表
- dietary_goals: 健康目标列表
- meal_type: 餐次或null
- servings: 用餐人数或null
- cooking_time_limit: 烹饪时间限制或null
- cuisine_preference: 菜系偏好或null
- difficulty: 难度或null
- low_spicy: true/false
- low_fat: true/false
- low_sugar: true/false
- vegetarian: true/false

<输出格式>
严格只返回一行JSON，不要markdown代码块:
{{"intent":"xxx","preferences":{{}},"reasoning":"xxx"}}"""

        try:
            response = llm_client.chat([{'role': 'user', 'content': prompt}], temperature=0.1, max_tokens=300)
            response = response.strip()
            if response.startswith('```'):
                lines_resp = response.split('\n')
                response = '\n'.join(lines_resp[1:-1])
            result = json.loads(response)
            return result
        except Exception as e:
            print(f'[WARN] LLM响应解析失败: {e}')
            return None

    def _detect_intent_llm(self, user_message: str, llm_client) -> Optional[str]:
        """仅获取意图"""
        result = self._detect_with_llm(user_message, llm_client)
        if result and result.get('intent'):
            if result.get('preferences'):
                self._update_preferences(result['preferences'])
            return result['intent']
        return None

    def record_timing(self, timing_data: Dict):
        """记录单轮耗时数据"""
        self.timing_history.append(timing_data)
        if len(self.timing_history) > 100:
            self.timing_history = self.timing_history[-50:]

    def get_timing_stats(self) -> Dict:
        """返回会话级性能统计"""
        if not self.timing_history:
            return {'count': 0}
        def _avg(key):
            vals = [t[key] for t in self.timing_history if t.get(key) is not None]
            return round(sum(vals)/len(vals),2) if vals else 0
        def _p(key, pct):
            vals = sorted([t[key] for t in self.timing_history if t.get(key) is not None])
            if not vals: return 0
            return round(vals[min(int(len(vals)*pct/100), len(vals)-1)], 2)
        return {
            'count': len(self.timing_history),
            'avg_total_ms': _avg('t_total_ms'),
            'avg_llm_ms': _avg('t_llm_ms'),
            'avg_search_ms': _avg('t_search_ms'),
            'avg_filter_ms': _avg('t_filter_ms'),
            'avg_verify_ms': _avg('t_verify_ms'),
            'avg_first_token_ms': _avg('t_first_token_ms'),
            'p50_total_ms': _p('t_total_ms', 50),
            'p95_total_ms': _p('t_total_ms', 95),
            'p50_first_token_ms': _p('t_first_token_ms', 50),
            'p95_first_token_ms': _p('t_first_token_ms', 95),
        }

if __name__ == "__main__":
    print("=" * 60)
    print("对话管理器测试")
    print("=" * 60)
    
    # 创建对话管理器实例
    dm = DialogManager()
    
    print("\n1. 测试意图识别...")
    intents = [
        "推荐今晚吃什么",
        "我对花生过敏",
        "修改我的偏好",
        "这个菜有多少热量",
        "好的，就要这个"
    ]
    
    for msg in intents:
        intent = dm.detect_intent(msg)
        print(f"   消息: '{msg}' -> 意图: {intent}")
    
    print("\n2. 测试偏好提取...")
    pref = dm.extract_preferences("我是素食者，不吃辣，想要减肥")
    print(f"   提取的偏好: {pref}")
    
    print("\n3. 测试对话状态...")
    dm.add_message('user', '推荐今晚吃什么')
    dm.add_message('system', '好的，为您推荐...')
    dm.add_message('user', '我想要清淡一点的')
    print(f"   对话轮数: {dm.turn_count}")
    print(f"   用户档案: {dm.get_user_profile()}")
    print(f"   是否需要确认: {dm.should_confirm()}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)