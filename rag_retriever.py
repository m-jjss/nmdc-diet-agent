import json
import os
import hashlib
from typing import List, Dict, Optional

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from config import Config
from llm_client import get_llm_client


class RAGRetriever:
    """
    RAG检索器类
    
    基于FAISS向量数据库实现菜谱检索，支持两种检索方式：
    1. 语义检索：基于文本Embedding向量相似度匹配
    2. 关键词检索：基于标签和食材的精确匹配
    
    核心功能：
    - 构建FAISS索引：将菜谱文本转换为向量并存储
    - 向量检索：根据用户查询向量查找最相似的菜谱
    - 关键词过滤：根据用户需求（如"不要辣"）过滤菜谱
    - 混合排序：综合语义相似度和关键词匹配度排序
    
    优化策略：
    - 索引缓存：避免重复构建FAISS索引
    - 向量缓存：缓存已计算的文本向量
    - 批量处理：支持批量菜谱向量化
    """
    
    def __init__(self):
        """
        初始化RAG检索器
        
        加载菜谱数据和营养数据库，初始化FAISS索引（延迟加载）。
        """
        self.recipes = []  # 菜谱列表，格式: [{'name': '...', 'ingredients': [...], ...}, ...]
        self.nutrition_db = {}  # 营养数据库，格式: {食材名: {'calories': ..., ...}}
        self.index = None  # FAISS索引对象
        self.recipe_index_map = []  # 索引ID到菜谱的映射
        self.vector_cache = {}  # 向量缓存，格式: {hash_key: vector}
        self._load_data()
    
    def _load_data(self):
        """
        加载菜谱数据和营养数据库
        
        从配置的JSON文件路径加载数据，失败时返回空数据结构。
        """
        # 加载菜谱数据
        try:
            with open(Config.RECIPES_JSON_PATH, 'r', encoding='utf-8') as f:
                self.recipes = json.load(f)
            print(f"成功加载 {len(self.recipes)} 条菜谱")
        except Exception as e:
            print(f"加载菜谱数据失败: {e}")
            self.recipes = []
        
        # 加载营养数据库
        try:
            with open(Config.NUTRITION_DB_PATH, 'r', encoding='utf-8') as f:
                self.nutrition_db = json.load(f)
            print(f"成功加载营养数据库，包含 {len(self.nutrition_db)} 种食材")
        except Exception as e:
            print(f"加载营养数据库失败: {e}")
            self.nutrition_db = {}
    
    def _text_to_vector(self, text: str) -> List[float]:
        """
        将文本转换为向量表示
        
        使用LLM客户端的embed接口进行向量化，支持缓存机制。
        
        Args:
            text: 待向量化的文本
            
        Returns:
            文本的向量表示，维度由配置的VECTOR_DIMENSION决定
        """
        # 计算缓存键
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        # 优先从缓存获取
        if cache_key in self.vector_cache:
            return self.vector_cache[cache_key]
        
        # 使用LLM客户端进行向量化
        client = get_llm_client()
        vector = client.embed(text)
        
        # 缓存向量结果
        self.vector_cache[cache_key] = vector
        return vector
    
    def build_index(self):
        """
        构建FAISS向量索引
        
        将所有菜谱的文本描述转换为向量，并构建FAISS索引。
        已构建的索引会被跳过，避免重复计算。
        支持索引持久化，从磁盘加载已保存的索引。
        
        Returns:
            True: 索引构建成功
            False: 索引构建失败（如FAISS未安装）
        """
        # 如果索引已存在，直接返回
        if self.index is not None:
            return True
        
        # FAISS未安装时返回失败
        if faiss is None:
            print("FAISS未安装，无法构建向量索引")
            return False
        
        # 如果没有菜谱数据，返回失败
        if not self.recipes:
            print("没有菜谱数据，无法构建索引")
            return False
        
        # 尝试从磁盘加载已保存的索引
        if self._load_index_from_disk():
            return True
        
        # 构建新索引
        print(f"正在构建FAISS索引，共 {len(self.recipes)} 条菜谱...")
        
        try:
            # 批量生成菜谱向量
            vectors = []
            self.recipe_index_map = []
            
            for recipe in self.recipes:
                # 构建菜谱文本描述
                text = f"{recipe.get('name', '')} {recipe.get('description', '')} {' '.join(recipe.get('ingredients', []))}"
                
                # 向量化并添加到列表
                vector = self._text_to_vector(text)
                
                # 确保向量维度一致
                if len(vector) == Config.VECTOR_DIMENSION:
                    vectors.append(vector)
                    self.recipe_index_map.append(recipe)
            
            # 转换为numpy数组
            vectors_np = np.array(vectors, dtype=np.float32)
            
            # 构建FAISS索引
            self.index = faiss.IndexFlatL2(Config.VECTOR_DIMENSION)
            self.index.add(vectors_np)
            
            # 保存索引到磁盘
            self._save_index_to_disk()
            
            print(f"FAISS索引构建完成，共 {len(vectors)} 条向量")
            return True
        except Exception as e:
            print(f"构建FAISS索引失败: {e}")
            return False
    
    def _save_index_to_disk(self):
        """
        保存索引到磁盘
        
        将FAISS索引和菜谱映射关系保存到文件，以便下次快速加载。
        """
        if self.index is None:
            return
        
        try:
            # 确保索引目录存在
            os.makedirs(str(Config.FAISS_INDEX_PATH), exist_ok=True)
            
            # 保存FAISS索引
            index_path = str(Config.FAISS_INDEX_PATH / "recipe_index.faiss")
            faiss.write_index(self.index, index_path)
            print(f"FAISS索引已保存到: {index_path}")
            
            # 保存菜谱映射关系（只保存名称和索引位置）
            map_data = []
            for i, recipe in enumerate(self.recipe_index_map):
                map_data.append({
                    'index': i,
                    'name': recipe.get('name', ''),
                    'hash': hashlib.md5(recipe.get('name', '').encode()).hexdigest()[:8]
                })
            
            map_path = str(Config.FAISS_INDEX_PATH / "recipe_index_map.json")
            with open(map_path, 'w', encoding='utf-8') as f:
                json.dump(map_data, f, ensure_ascii=False, indent=2)
            print(f"菜谱映射已保存到: {map_path}")
            
        except Exception as e:
            print(f"保存索引失败: {e}")
    
    def _load_index_from_disk(self) -> bool:
        """
        从磁盘加载索引
        
        如果磁盘上有已保存的索引，尝试加载它。
        
        Returns:
            True: 加载成功
            False: 加载失败
        """
        try:
            index_path = str(Config.FAISS_INDEX_PATH / "recipe_index.faiss")
            map_path = str(Config.FAISS_INDEX_PATH / "recipe_index_map.json")
            
            if not os.path.exists(index_path) or not os.path.exists(map_path):
                return False
            
            print("正在从磁盘加载FAISS索引...")
            
            # 加载FAISS索引
            self.index = faiss.read_index(index_path)
            
            # 加载菜谱映射
            with open(map_path, 'r', encoding='utf-8') as f:
                map_data = json.load(f)
            
            # 重建菜谱映射关系
            self.recipe_index_map = []
            for item in map_data:
                # 根据名称查找菜谱
                for recipe in self.recipes:
                    if recipe.get('name', '') == item['name']:
                        self.recipe_index_map.append(recipe)
                        break
            
            print(f"FAISS索引加载完成，共 {len(self.recipe_index_map)} 条向量")
            return True
        
        except Exception as e:
            print(f"加载索引失败: {e}")
            return False
    
    def search(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """
        检索菜谱
        
        综合使用语义检索和关键词过滤，返回最匹配的菜谱列表。
        
        Args:
            query: 用户查询文本
            top_k: 返回结果数量，默认为10
            filters: 过滤条件，格式: {'exclude_ingredients': [...], 'tags': [...]}
            
        Returns:
            匹配的菜谱列表，按相似度降序排列
        """
        # 确保索引已构建
        if self.index is None and not self.build_index():
            # 索引构建失败时回退到简单的关键词匹配
            return self._simple_search(query, top_k, filters)
        
        # 将查询文本转换为向量
        query_vector = np.array([self._text_to_vector(query)], dtype=np.float32)
        
        # 使用FAISS进行向量检索
        distances, indices = self.index.search(query_vector, min(top_k, len(self.recipe_index_map)))
        
        # 获取检索结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.recipe_index_map):
                recipe = self.recipe_index_map[idx].copy()
                recipe['score'] = float(1 / (1 + distances[0][i]))  # 转换为相似度得分(0~1)
                results.append(recipe)
        
        # 应用关键词过滤
        if filters:
            results = self._apply_filters(results, filters)
        
        # 重新排序（综合相似度和关键词匹配度）
        results = self._rerank(results, query)
        
        return results[:top_k]
    
    def _simple_search(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """
        简单关键词检索（FAISS不可用时的回退方案）
        
        基于文本匹配度进行简单检索，返回匹配的菜谱列表。
        
        Args:
            query: 用户查询文本
            top_k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            匹配的菜谱列表
        """
        results = []
        
        for recipe in self.recipes:
            # 计算匹配度
            score = self._calculate_match_score(recipe, query)
            if score > 0:
                recipe_copy = recipe.copy()
                recipe_copy['score'] = score
                results.append(recipe_copy)
        
        # 按匹配度排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # 应用过滤条件
        if filters:
            results = self._apply_filters(results, filters)
        
        return results[:top_k]
    
    def _calculate_match_score(self, recipe: Dict, query: str) -> float:
        """
        计算菜谱与查询的匹配度
        
        基于菜名、食材、描述与查询的文本匹配程度计算得分。
        
        Args:
            recipe: 菜谱数据
            query: 查询文本
            
        Returns:
            匹配度得分(0~1)
        """
        score = 0
        query_lower = query.lower()
        
        # 菜名匹配（权重最高）
        name = recipe.get('name', '').lower()
        if query_lower in name:
            score += 0.5
        
        # 食材匹配
        ingredients = ' '.join([i.lower() for i in recipe.get('ingredients', [])])
        for word in query_lower.split():
            if word in ingredients:
                score += 0.1
        
        # 描述匹配
        description = recipe.get('description', '').lower()
        if query_lower in description:
            score += 0.2
        
        # 标签匹配
        tags = ' '.join([t.lower() for t in recipe.get('tags', [])])
        if query_lower in tags:
            score += 0.2
        
        return min(score, 1.0)
    
    def _apply_filters(self, recipes: List[Dict], filters: Dict) -> List[Dict]:
        """
        应用过滤条件
        
        根据用户指定的条件过滤菜谱列表，支持多维度过滤。
        
        Args:
            recipes: 待过滤的菜谱列表
            filters: 过滤条件，支持:
                - exclude_ingredients: 排除的食材列表
                - exclude_recipes: 排除的菜品名称列表
                - tags: 必须包含的标签列表
                - max_calories: 最大热量限制
                - max_cooking_time: 最大烹饪时间（分钟）
                - servings: 用餐人数
                - meal_type: 餐次
                - low_fat: 低脂
                - low_spicy: 少辣
                - low_sugar: 低糖
                - vegetarian: 素食
                
        Returns:
            过滤后的菜谱列表
        """
        filtered = recipes
        
        # 排除指定食材（在食材、描述、做法中检查）
        exclude_ingredients = filters.get('exclude_ingredients', [])
        if exclude_ingredients:
            exclude_set = set([str(i).lower() for i in exclude_ingredients])
            filtered = [
                r for r in filtered
                if not exclude_set.intersection(
                    set([str(i).lower() for i in r.get('ingredients', [])]) |
                    set(r.get('description', '').lower().split()) |
                    set(r.get('method', '').lower().split())
                ) and not any(
                    ex in str(r.get('description', '')).lower() or
                    ex in str(r.get('method', '')).lower()
                    for ex in exclude_set
                )
            ]
        
        # 排除指定菜品
        exclude_recipes = filters.get('exclude_recipes', [])
        if exclude_recipes:
            exclude_set = set([str(r).lower() for r in exclude_recipes])
            filtered = [
                r for r in filtered
                if str(r.get('name', '')).lower() not in exclude_set
            ]
        
        # 必须包含指定标签
        required_tags = filters.get('tags', [])
        if required_tags:
            tag_set = set([str(t).lower() for t in required_tags])
            filtered = [
                r for r in filtered
                if tag_set.intersection([str(t).lower() for t in r.get('tags', [])])
            ]
        
        # 热量限制
        max_calories = filters.get('max_calories')
        if max_calories:
            filtered = [
                r for r in filtered
                if self._calculate_calories(r) <= max_calories
            ]
        
        # 烹饪时间限制（未知烹饪时间=999的菜谱保留，不排除）
        max_cooking_time = filters.get('max_cooking_time')
        if max_cooking_time:
            filtered = [
                r for r in filtered
                if self._extract_cooking_time(r) <= max_cooking_time or self._extract_cooking_time(r) == 999
            ]
        
        # 口味偏好过滤
        if filters.get('low_fat'):
            high_fat_methods = ['油炸', '油煎', '红烧', '油焖', '爆炒']
            filtered = [
                r for r in filtered
                if not any(m in str(r.get('description', '')).lower() or
                          m in str(r.get('method', '')).lower() or
                          m in ' '.join(r.get('tags', [])).lower()
                          for m in high_fat_methods)
            ]
        
        if filters.get('low_spicy'):
            spicy_keywords = ['辣椒', '花椒', '辣', '麻辣', '香辣']
            filtered = [
                r for r in filtered
                if not any(s in str(r.get('description', '')).lower() or
                          s in str(r.get('method', '')).lower() or
                          s in ' '.join(r.get('tags', [])).lower() or
                          s in ' '.join([str(i).lower() for i in r.get('ingredients', [])])
                          for s in spicy_keywords)
            ]
        
        if filters.get('low_sugar'):
            sweet_keywords = ['糖', '蜂蜜', '冰糖', '白糖', '甜点']
            filtered = [
                r for r in filtered
                if not any(s in ' '.join([str(i).lower() for i in r.get('ingredients', [])])
                          for s in sweet_keywords)
            ]
        
        if filters.get('vegetarian'):
            meat_keywords = ['猪肉', '牛肉', '羊肉', '鸡肉', '鸭肉', '鱼肉', '虾', '蟹', '海鲜']
            filtered = [
                r for r in filtered
                if not any(m in ' '.join([str(i).lower() for i in r.get('ingredients', [])])
                          for m in meat_keywords)
            ]
        
        return filtered
    
    def _extract_cooking_time(self, recipe: Dict) -> int:
        """
        从菜谱中提取烹饪时间
        
        Args:
            recipe: 菜谱数据
            
        Returns:
            烹饪时间（分钟），无法提取时返回999
        """
        import re
        
        # 检查tags中的时间信息
        for tag in recipe.get('tags', []):
            match = re.search(r'(\d+)\s*分钟', str(tag))
            if match:
                return int(match.group(1))
        
        # 检查description中的时间信息
        description = str(recipe.get('description', ''))
        match = re.search(r'(\d+)\s*分钟', description)
        if match:
            return int(match.group(1))
        
        # 检查method中的时间信息
        method = str(recipe.get('method', ''))
        match = re.search(r'(\d+)\s*分钟', method)
        if match:
            return int(match.group(1))
        
        # 无法提取，返回较大值表示无限制
        return 999
    
    def _calculate_calories(self, recipe: Dict) -> float:
        """
        计算菜谱的热量
        
        根据营养数据库估算菜谱的总热量。
        
        Args:
            recipe: 菜谱数据
            
        Returns:
            估算的热量值（千卡）
        """
        total_calories = 0
        
        for ingredient in recipe.get('ingredients', []):
            # 在营养数据库中查找食材热量
            if ingredient in self.nutrition_db:
                total_calories += self.nutrition_db[ingredient].get('calories', 0)
            else:
                # 未知食材按平均热量估算
                total_calories += 50
        
        return total_calories
    
    def _rerank(self, recipes: List[Dict], query: str) -> List[Dict]:
        """
        重新排序检索结果（含多样性优化）
        
        综合考虑语义相似度得分、关键词匹配度和多样性进行排序。
        增加餐次感知：晚餐/午餐优先推荐主菜，早餐优先推荐主食/饮品。
        烹饪方式多样性：避免全是同一烹饪方式。
        
        Args:
            recipes: 待排序的菜谱列表
            query: 用户查询文本
            
        Returns:
            重新排序后的菜谱列表
        """
        query_lower = query.lower()
        
        # 判断餐次
        is_dinner = any(kw in query_lower for kw in ['晚餐', '晚饭', '今晚', '晚上'])
        is_lunch = any(kw in query_lower for kw in ['午餐', '午饭', '中午'])
        is_breakfast = any(kw in query_lower for kw in ['早餐', '早饭', '早上', '早点'])
        is_meal_query = is_dinner or is_lunch or is_breakfast
        
        # 主菜关键词（正餐应优先）
        main_dish_tags = ['主菜', '热菜', '家常菜', '炒菜', '荤菜', '素菜']
        # 点心/零食关键词（正餐应降权）
        snack_tags = ['点心', '零食', '小吃', '甜品', '饼干', '烘焙', '蛋糕', '零食小吃']
        
        meat_keywords = ['猪肉', '牛肉', '羊肉', '鸡肉', '鸭肉', '鱼肉', '虾', '蟹', '排骨', '五花肉', '鸡', '鱼', '肉', '虾仁']
        cooking_methods_all = ['蒸', '煮', '炒', '炖', '烤', '凉拌', '煎', '炸', '煲', '烧', '焖']
        
        for recipe in recipes:
            # 获取基础相似度得分（来自FAISS检索）
            base_score = recipe.get('score', 0)
            
            # 计算关键词匹配得分
            keyword_score = self._calculate_match_score(recipe, query)
            
            # 综合得分：语义相似度占70%，关键词匹配占30%
            final_score = base_score * 0.7 + keyword_score * 0.3
            
            # 餐次感知加权
            if is_meal_query:
                name = str(recipe.get('name', ''))
                ings = str(recipe.get('ingredients', ''))
                tags = [str(t) for t in recipe.get('tags', [])]
                
                # 正餐（午餐/晚餐）降权甜点、零食、饼干
                if is_dinner or is_lunch:
                    if any(st in name or st in ' '.join(tags) for st in snack_tags):
                        final_score *= 0.5  # 大幅降权
                
                # 正餐（午餐/晚餐）提权含主料的热菜
                if is_dinner or is_lunch:
                    has_meat = any(m in ings + name for m in meat_keywords)
                    is_main = any(md in ' '.join(tags) for md in main_dish_tags)
                    if has_meat or is_main:
                        final_score *= 1.15  # 小幅提权
                
                # 早餐提权粥、饼、饮品
                if is_breakfast:
                    breakfast_boost = ['粥', '饼', '豆浆', '牛奶', '面', '馒头', '包子', '饺子']
                    if any(b in name for b in breakfast_boost):
                        final_score *= 1.2
            
            recipe['final_score'] = final_score
        
        # 按综合得分排序
        recipes.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        # 多样性重排序：避免连续推荐相同烹饪方式
        diverse = []
        used_methods = set()
        remaining = list(recipes)
        
        while remaining:
            # 从剩余中选择得分最高且烹饪方式不在已用集合中的
            best_idx = None
            for i, r in enumerate(remaining):
                method = self._get_cooking_method(r)
                if method not in used_methods or len(used_methods) >= len(cooking_methods_all):
                    best_idx = i
                    break
            
            if best_idx is None:
                best_idx = 0  # 回退：选得分最高的
            
            chosen = remaining.pop(best_idx)
            method = self._get_cooking_method(chosen)
            used_methods.add(method)
            diverse.append(chosen)
        
        return diverse
    
    def _get_cooking_method(self, recipe: Dict) -> str:
        """从菜谱中提取主要烹饪方式"""
        name = str(recipe.get('name', ''))
        ings = str(recipe.get('ingredients', ''))
        tags = [str(t) for t in recipe.get('tags', [])]
        desc = str(recipe.get('description', ''))
        method_text = name + ings + ' '.join(tags) + desc
        
        # 按优先级匹配烹饪方式
        for m in ['凉拌', '蒸', '煮', '炒', '炖', '烤', '煎', '炸', '煲', '烧', '焖', '烩']:
            if m in method_text:
                return m
        return '其他'
    
    def get_recipe_by_name(self, name: str) -> Optional[Dict]:
        """
        根据菜名获取菜谱详情
        
        Args:
            name: 菜名
            
        Returns:
            菜谱数据，如果未找到返回None
        """
        for recipe in self.recipes:
            if recipe.get('name') == name:
                return recipe
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("RAG检索器测试")
    print("=" * 60)
    
    # 创建检索器实例
    retriever = RAGRetriever()
    
    print("\n1. 构建索引...")
    retriever.build_index()
    
    print("\n2. 测试检索（'今晚吃什么'）...")
    results = retriever.search("今晚吃什么", top_k=5)
    for i, r in enumerate(results):
        print(f"   {i+1}. {r['name']} (得分: {r.get('score', 0):.4f})")
    
    print("\n3. 测试过滤检索（'不要辣的'）...")
    results = retriever.search("推荐菜品", top_k=5, filters={'exclude_ingredients': ['辣椒', '辣']})
    for i, r in enumerate(results):
        print(f"   {i+1}. {r['name']}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)