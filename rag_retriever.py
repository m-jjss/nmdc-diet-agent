import json
import os
import re
import hashlib
from typing import List, Dict, Optional

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from config import Config
from llm_client import get_llm_client


def _faiss_index_dir() -> str:
    """返回 FAISS 索引实际落盘目录。

    faiss 的磁盘写用 C 层 fopen，在 Windows 上无法打开含非 ANSI（中文）字符的路径
    （但 Python/pathlib 层可以）。因此当配置的索引目录含非 ASCII 字符时，回退到系统
    临时目录并加项目标识（ASCII 可写路径），保证本机调试与演示环境下索引也能持久化。
    保存与加载统一走本函数，保证读写路径一致。
    """
    path = str(Config.FAISS_INDEX_PATH)
    if path.isascii():
        return path
    import tempfile
    pid = hashlib.md5(path.encode('utf-8')).hexdigest()[:8]
    return os.path.join(tempfile.gettempdir(), "nmdc_faiss_" + pid)


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
        self._cross_encoder = None  # 交叉编码器精排模型（bge-reranker，懒加载）
        self._ce_state = 'unloaded'  # unloaded / ready / unavailable
        self._ce_score_cache = {}  # (query, recipe_name) -> score，避免重复打分
        self._load_data()
    
    # 非菜品条目：方太料理机的功能操作/预处理指令，不是一道菜，检索时必须排除
    NON_DISH_NAMES = {
        '68℃慢煮', '59℃慢煮', '高温快煮', '停刀烧煮', '蛋白打发',
        '奶油打发', '绞肉', '果蔬清洗', '酸奶发酵', '乳化', '颠勺',
    }

    def _is_non_dish(self, recipe: Dict) -> bool:
        """
        判断条目是否为非菜品（料理机操作/预处理指令）
        
        判定依据：
        1. 名称命中已知功能操作黑名单（如"68℃慢煮""奶油打发"）
        2. 配料以"主料：自定义"开头（无具体食材，只是料理机程序占位）
        
        Args:
            recipe: 菜谱条目
            
        Returns:
            True: 是非菜品条目，应过滤
        """
        name = str(recipe.get('name', '')).strip()
        if name in self.NON_DISH_NAMES:
            return True
        ingredients = str(recipe.get('ingredients', ''))
        if ingredients.startswith('主料：自定义'):
            return True
        return False

    def _safe_ingredients(self, recipe: Dict) -> List[str]:
        """
        解析菜谱食材为食材名列表，用于匹配/过滤。
        兼容 ingredients 为字符串（"主料：梨肉1000g；红枣肉20g"）或列表两种格式。
        注意：只用于检索匹配，不修改 recipe 数据（营养计算依赖原始字符串中的克数）。

        Args:
            recipe: 菜谱条目

        Returns:
            食材名列表（小写）
        """
        ings = recipe.get('ingredients', [])
        if isinstance(ings, list):
            return [str(i).lower() for i in ings if str(i).strip()]
        if isinstance(ings, str):
            ingredient_list = []
            for part in ings.split('；'):
                part = part.strip()
                if '：' in part:
                    part = part.split('：', 1)[1].strip()
                if not part:
                    continue
                # 去数字单位
                text = re.sub(r'\d+(\.\d+)?\s*[g克ml毫升l升kg千克]', '', part)
                # 按常见分隔符切分
                for sep in ['，', ',', '、', ';']:
                    text = text.replace(sep, '，')
                for sub in text.split('，'):
                    sub = sub.strip()
                    if sub:
                        ingredient_list.append(sub.lower())
            return ingredient_list
        return []

    def _load_data(self):
        """
        加载菜谱数据和营养数据库
        
        从配置的JSON文件路径加载数据，失败时返回空数据结构。
        加载时过滤掉非菜品条目（料理机操作/预处理指令），保证检索结果只含真实菜品。
        """
        # 加载菜谱数据
        try:
            with open(Config.RECIPES_JSON_PATH, 'r', encoding='utf-8') as f:
                all_recipes = json.load(f)
            # 过滤非菜品条目（料理机操作类数据）
            self.recipes = [r for r in all_recipes if not self._is_non_dish(r)]
            filtered_count = len(all_recipes) - len(self.recipes)
            if filtered_count:
                print(f"已过滤 {filtered_count} 条非菜品条目（料理机操作/预处理）")
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
                text = f"{recipe.get('name', '')} {recipe.get('description', '')} {' '.join(self._safe_ingredients(recipe))}"
                
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
            # 确保索引目录存在（含非 ASCII 路径时自动回退到 ASCII 临时目录）
            _dir = _faiss_index_dir()
            os.makedirs(_dir, exist_ok=True)

            # 保存FAISS索引（serialize + Python bytes 写盘，天然支持中文路径，
            # 避免 faiss C fopen 在 Windows 打开非 ANSI 路径失败）
            index_path = os.path.join(_dir, "recipe_index.faiss")
            with open(index_path, 'wb') as f:
                f.write(faiss.serialize_index(self.index))
            print(f"FAISS索引已保存到: {index_path}")

            # 保存菜谱映射关系（只保存名称和索引位置）
            map_data = []
            for i, recipe in enumerate(self.recipe_index_map):
                map_data.append({
                    'index': i,
                    'name': recipe.get('name', ''),
                    'hash': hashlib.md5(recipe.get('name', '').encode()).hexdigest()[:8]
                })

            map_path = os.path.join(_dir, "recipe_index_map.json")
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
            _dir = _faiss_index_dir()
            index_path = os.path.join(_dir, "recipe_index.faiss")
            map_path = os.path.join(_dir, "recipe_index_map.json")
            
            if not os.path.exists(index_path) or not os.path.exists(map_path):
                return False
            
            print("正在从磁盘加载FAISS索引...")

            # 加载FAISS索引（Python bytes 读盘 + deserialize，兼容中文路径）
            with open(index_path, 'rb') as f:
                self.index = faiss.deserialize_index(
                    np.frombuffer(f.read(), dtype=np.uint8))

            # 维度校验：配置的 embedding 模型变更（如 384维MiniLM → 512维bge）后，
            # 旧索引与查询向量维度不一致会导致检索错乱，此时应丢弃旧索引重建。
            if getattr(self.index, 'd', 0) != Config.VECTOR_DIMENSION:
                print(f"FAISS索引维度不匹配（索引 {self.index.d} 维 vs 配置 {Config.VECTOR_DIMENSION} 维），将重建索引")
                self.index = None
                return False
            
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
    
    # ── 交叉编码器精排（bge-reranker）─────────────────────────
    # 双塔 embedding 只能"粗排"（查询与文档各自编码后比相似度），
    # 交叉编码器把 query 和菜谱文本拼在一起过一遍 BERT，能捕捉细粒度相关性，
    # 对"下饭菜/清淡的汤"这类模糊查询的前几名排序提升明显。
    # 模型不可用（未下载/离线）时自动跳过，不影响原有检索流程。

    _CE_MODEL_NAME = 'BAAI/bge-reranker-base'
    _CE_MAX_PAIRS = 16       # 每次精排最多打分的候选数（控制CPU耗时）
    _CE_SCORE_MIN = 0.0      # 交叉编码器输出经sigmoid后低于该值视为弱相关

    def _init_cross_encoder(self):
        """懒加载交叉编码器；不可用时置为 unavailable，只尝试一次。"""
        if self._ce_state == 'ready':
            return self._cross_encoder
        if self._ce_state == 'unavailable':
            return None
        try:
            import os as _os
            _os.environ.setdefault('HF_HUB_OFFLINE', '1')  # 竞赛演示环境离线优先，避免网络阻塞
            try:
                import torch as _torch
                _torch.set_num_threads(4)  # 限制推理线程，防止高频评测下 OMP 线程耗尽崩溃
            except ImportError:
                pass
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(self._CE_MODEL_NAME, max_length=128)
            self._ce_state = 'ready'
            print(f"[OK] 交叉编码器精排已加载: {self._CE_MODEL_NAME}")
        except Exception as e:
            self._ce_state = 'unavailable'
            self._cross_encoder = None
            print(f"[INFO] 交叉编码器不可用({str(e)[:80]})，跳过精排，使用原排序")
        return self._cross_encoder

    def _cross_encoder_rerank(self, candidates: List[Dict], query: str, top_k: int) -> List[Dict]:
        """用交叉编码器对候选菜精排，返回按相关性降序的前 top_k 个。
        打分文本 = 菜名 + 主要食材，与用户查询拼接后逐一打分；
        分数缓存避免同一轮多路检索重复计算。"""
        ce = self._init_cross_encoder()
        if ce is None or not candidates:
            return candidates[:top_k]
        pool = candidates[:self._CE_MAX_PAIRS]
        pairs, keys = [], []
        for r in pool:
            name = r.get('name', '')
            ings = ' '.join(self._safe_ingredients(r))[:60]
            keys.append((query, name))
            pairs.append((query, f"{name} {ings}"))
        to_pred, pred_idx = [], []
        for i, k in enumerate(keys):
            if k in self._ce_score_cache:
                continue
            to_pred.append(pairs[i])
            pred_idx.append(i)
        if to_pred:
            try:
                scores = ce.predict(to_pred)
                import math
                for j, i in enumerate(pred_idx):
                    s = float(scores[j])
                    # bge-reranker 输出 logits，sigmoid 归一到 0~1
                    self._ce_score_cache[keys[i]] = 1 / (1 + math.exp(-s)) if -60 < s < 60 else (1.0 if s > 0 else 0.0)
            except Exception as e:
                print(f"[WARN] 精排打分失败: {str(e)[:60]}")
                return candidates[:top_k]
        scored = [(self._ce_score_cache.get(k, 0.0), r) for k, r in zip(keys, pool)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

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
        
        # 使用FAISS进行向量检索（扩大召回，给关键词匹配留出合并空间）
        distances, indices = self.index.search(query_vector, min(top_k * 3, len(self.recipe_index_map)))
        
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
        
        # 关键词精确匹配候选：确保真正相关的菜（"虾"）优先进来精确命中
        kw_results = self._simple_search(query, top_k=top_k * 2, filters=filters)

        # 合并策略（语义 embedding 已启用后）
        # 1) 关键词命中优先（相关性最强的精确匹配，如"虾"）
        # 2) FAISS 语义近邻其次（能理解同义/口语表达，如"想吃点下饭的""清淡一点的晚餐"）
        # 3) 仅当两者都不足时才退化到确定性规则兜底（"今晚吃什么"这类无指向查询）
        #    —— 规则兜底不再提前用，避免语义有效的查询被固定菜单淹没
        # 合并后若交叉编码器可用，用 bge-reranker 对候选池做细粒度精排再截断 top_k
        pool_size = max(top_k * 2, 12)
        merged, seen = [], set()
        for r in kw_results:
            name = r.get('name', '')
            if not name or name in seen:
                continue
            seen.add(name)
            merged.append(r)
            if len(merged) >= pool_size:
                break

        if len(merged) < pool_size:
            for r in results:
                name = r.get('name', '')
                if name and name not in seen:
                    seen.add(name)
                    merged.append(r)
                if len(merged) >= pool_size:
                    break

        if len(merged) < pool_size:
            for r in self._fallback_selection(filters, pool_size):
                name = r.get('name', '')
                if name and name not in seen:
                    seen.add(name)
                    merged.append(r)
                if len(merged) >= pool_size:
                    break

        # 交叉编码器精排：对合并池按"查询-菜谱"真实相关性重排，取前 top_k
        return self._cross_encoder_rerank(merged, query, top_k)

    def _fallback_selection(self, filters: Optional[Dict] = None, top_k: int = 5) -> List[Dict]:
        """
        模糊查询兜底：当关键词/语义检索均无法给出相关结果（如"今晚吃什么""随便推荐"）时，
        从全库按规则挑选一餐：荤素搭配 + 烹饪方式多样 + 遵循用户偏好过滤。

        Args:
            filters: 检索过滤条件
            top_k: 返回数量

        Returns:
            规则化挑选的菜谱列表
        """
        pool = self._apply_filters(self.recipes, filters or {})
        if not pool:
            pool = self.recipes

        meat_kws = ['肉', '鸡', '鸭', '鱼', '虾', '蟹', '贝', '牛', '猪', '羊', '排骨',
                    '腊', '肠', '腿', '翅', '肝', '鲍', '参', '蛤', '鱿', '鳗', '蚝']
        meat_dishes = [r for r in pool
                       if any(k in str(r.get('name', '')) + ' '.join(self._safe_ingredients(r))
                              for k in meat_kws)]
        veg_dishes = [r for r in pool if r not in meat_dishes]

        methods = ['蒸', '炒', '煮', '炖', '烤', '凉拌', '煎', '煲', '烧', '焖', '炸']
        used_names = set()
        result = []

        def _pick(source):
            # 优先挑选不同烹饪方式的主料菜
            for m in methods:
                for r in source:
                    if r.get('name') in used_names:
                        continue
                    if self._get_cooking_method(r) == m:
                        used_names.add(r.get('name'))
                        return r
            for r in source:
                if r.get('name') not in used_names:
                    used_names.add(r.get('name'))
                    return r
            return None

        # 荤素交替选择，保证一餐搭配均衡
        sources = [meat_dishes, veg_dishes]
        idx = 0
        for _ in range(top_k):
            src = sources[idx % 2]
            idx += 1
            if not src:
                src = sources[(idx) % 2]
            chosen = _pick(src) if src else None
            if chosen:
                result.append(chosen)
            else:
                break

        # 仍不足时从全部池补充
        if len(result) < top_k:
            for r in pool:
                if r.get('name') and r.get('name') not in used_names:
                    result.append(r)
                    used_names.add(r.get('name'))
                if len(result) >= top_k:
                    break

        return result[:top_k]
    
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
        # 查询按空格拆分（如"辣 麻辣 香辣 辣椒 剁椒 川菜 湘菜 辛辣"），任一词命中即计分
        query_tokens = [t for t in query.lower().split() if t] or [query.lower()]

        name = str(recipe.get('name', '')).lower()
        ingredients = ' '.join(self._safe_ingredients(recipe))
        description = str(recipe.get('description', '')).lower()
        tags = ' '.join(str(t).lower() for t in recipe.get('tags', []))
        label = str(recipe.get('label', '')).lower()

        for token in query_tokens:
            token = token.strip()
            if not token:
                continue
            t_score = 0
            # 菜名匹配（权重最高）
            if token in name:
                t_score += 0.5
            # 食材匹配
            if token in ingredients:
                t_score += 0.1
            # 描述匹配
            if token in description:
                t_score += 0.2
            # 标签匹配
            if token in tags or token in label:
                t_score += 0.2
            score = max(score, t_score)

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
        
        # 排除指定食材（在菜名、食材、描述、做法中做子串检查）
        exclude_ingredients = filters.get('exclude_ingredients', [])
        if exclude_ingredients:
            # 非食材虚词兜底：即使上游（LLM/关键词）漏过滤了"重复/太多"等程度副词，
            # 这里也不能把它们当字面关键词过滤——否则会把候选全部清空（如"做法不要重复"）。
            _NON_FOOD_FILTER = ('重复', '太多', '太多油', '太重', '太重口', '太油', '太咸', '太甜',
                                '太辣', '一样', '相同', '类似', '复杂', '麻烦')
            exclude_ingredients = [e for e in exclude_ingredients
                                   if str(e).strip() and str(e).strip() not in _NON_FOOD_FILTER]
            if not exclude_ingredients:
                exclude_ingredients = []
            if exclude_ingredients:
                # 类别词展开：如"海鲜"要能排除鱼虾蟹贝等具体菜品，而不是只匹配字面"海鲜"
                category_expansion = {
                '海鲜': ['海鲜', '鱼', '虾', '蟹', '贝', '蚝', '蛤', '鱿', '鳗', '鲍', '牡蛎', '海参', '海带', '紫菜', '鱼片', '鱼柳', '鱼头', '鱼丸', '鱼蛋', '虾仁', '虾滑', '虾米', '蟹肉', '蟹黄', '带子', '扇贝', '蛏子', '花甲', '蛤蜊', '三文鱼', '鳕鱼', '鲈鱼', '黄鱼', '鲳鱼', '银鱼', '带鱼'],
                '鱼': ['鱼', '鱼片', '鱼柳', '鱼块', '鱼头', '鱼丸', '鱼蛋', '三文鱼', '鳕鱼', '鲈鱼', '黄鱼', '鲳鱼', '鲷鱼', '带鱼', '银鱼', '罗非鱼', '龙利鱼', '巴沙鱼', '金枪鱼'],
                '虾': ['虾', '虾仁', '虾滑', '虾米', '明虾', '基围虾', '小龙虾'],
                '蟹': ['蟹', '蟹肉', '蟹黄', '蟹柳', '大闸蟹'],
                '贝': ['贝', '蛤蜊', '扇贝', '带子', '蛏子', '花甲', '青口', '牡蛎', '生蚝', '鲍鱼'],
                # 肉类类别词展开：用户说"不吃羊肉"要能排除"烤羊排/羊腿/羊蝎子"等次生词，
                # 而不是只匹配字面"羊肉"。注意刻意不放"牛油"（会误伤牛油果）、"鸡蛋"（不吃鸡不等于忌蛋）。
                '羊': ['羊', '羊肉', '羊排', '羊腿', '羊蝎子', '羊腩', '羊肚', '羊杂', '羊肉卷', '肥羊', '烤羊排', '孜然羊肉', '羊肉串'],
                '羊肉': ['羊', '羊肉', '羊排', '羊腿', '羊蝎子', '羊腩', '羊肚', '羊杂', '羊肉卷', '肥羊', '烤羊排', '孜然羊肉', '羊肉串'],
                '牛': ['牛', '牛肉', '牛腩', '牛柳', '牛排', '牛腱', '肥牛', '牛肚', '牛百叶', '牛肉丸', '酱牛肉', '卤牛肉'],
                '牛肉': ['牛', '牛肉', '牛腩', '牛柳', '牛排', '牛腱', '肥牛', '牛肚', '牛百叶', '牛肉丸', '酱牛肉', '卤牛肉'],
                '猪': ['猪', '猪肉', '五花肉', '猪排', '猪蹄', '排骨', '里脊', '肉末', '肉丝', '瘦肉', '回锅肉', '红烧肉'],
                '猪肉': ['猪', '猪肉', '五花肉', '猪排', '猪蹄', '排骨', '里脊', '肉末', '肉丝', '瘦肉', '回锅肉', '红烧肉'],
                '鸡': ['鸡', '鸡肉', '鸡腿', '鸡翅', '鸡爪', '鸡胸', '鸡肝', '乌鸡', '白切鸡', '盐焗鸡', '烤鸡'],
                '鸡肉': ['鸡', '鸡肉', '鸡腿', '鸡翅', '鸡爪', '鸡胸', '鸡肝', '乌鸡', '白切鸡', '盐焗鸡', '烤鸡'],
                '鸭': ['鸭', '鸭肉', '鸭腿', '烤鸭', '烧鸭', '酱鸭', '盐水鸭', '鸭胗'],
                '鸭肉': ['鸭', '鸭肉', '鸭腿', '烤鸭', '烧鸭', '酱鸭', '盐水鸭', '鸭胗'],
                '鹅': ['鹅', '鹅肉', '烧鹅'],
                '鹅肉': ['鹅', '鹅肉', '烧鹅'],
            }
            exclude_set = [str(i).lower() for i in exclude_ingredients]
            expanded = []
            for ex in exclude_set:
                expanded.append(ex)
                expanded.extend(category_expansion.get(ex, []))
            exclude_set = list(dict.fromkeys(expanded))  # 去重保序

            def _excluded(r):
                name_l = str(r.get('name', '')).lower()
                _ing = r.get('ingredients', '')
                if isinstance(_ing, str):
                    # ingredients 是字符串（如"主料：…香菜…"）：必须整体匹配。
                    # 若用 [str(i) for i in 字符串] 会逐字符遍历，把"香菜"拆成单字，
                    # 导致"我不吃香菜"这类排除约束全部失效（曾漏推含香菜菜）。
                    ing_l = _ing.lower()
                else:
                    ing_l = ' '.join(str(i).lower() for i in _ing)
                desc_l = str(r.get('description', '')).lower()
                meth_l = str(r.get('method', '')).lower()
                return any(ex in name_l or ex in ing_l or ex in desc_l or ex in meth_l
                           for ex in exclude_set)

            filtered = [r for r in filtered if not _excluded(r)]
        
        # 排除指定菜品
        exclude_recipes = filters.get('exclude_recipes', [])
        if exclude_recipes:
            exclude_set = set([str(r).lower() for r in exclude_recipes])
            filtered = [
                r for r in filtered
                if str(r.get('name', '')).lower() not in exclude_set
            ]
        
        # 偏好烹饪方式过滤（"有没有炸的/来点烤的"）：保留用其中任一方式烹饪的菜。
        # cooking_method 可能是列表（"有炒的有蒸的有炖的"表示集合，任一命中即可），
        # 也可能是单个字符串；把它展开为方法集合做"任一命中"过滤，
        # 避免把列表整体当单个关键词导致不匹配任何菜（曾导致推荐被全部清空）。
        preferred_method = filters.get('preferred_method')
        if preferred_method:
            if isinstance(preferred_method, (list, tuple, set)):
                _pm_set = {str(m).lower() for m in preferred_method if m}
            else:
                _pm_set = {str(preferred_method).lower()}
            filtered = [
                r for r in filtered
                if any(pm in str(self._get_cooking_method(r)).lower()
                       or pm in str(r.get('method', '')).lower()
                       or pm in str(r.get('name', '')).lower()
                       for pm in _pm_set)
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
            # 高脂做法：含"油焖/油炸/红烧/干锅/水煮/麻辣香锅/辣子"等重油菜
            high_fat_methods = ['油炸', '油煎', '红烧', '油焖', '爆炒', '干煸', '回锅',
                                '干锅', '水煮肉', '水煮鱼', '麻辣香锅', '辣子鸡', '辣子',
                                '红烧肉', '铁板', '干炸', '酥炸', '糖醋里脊',
                                '烤鱼', '石锅', '焗饭', '铁板烧', '香锅', '酱爆', '油爆',
                                '炸鸡', '炸猪排', '天妇罗', '炙烤肥牛', '烤羊排', '烧肉']
            # 高脂食材（含腊味/肥肉/油炸类），清淡/低脂/减肥时排除
            high_fat_ings = ['五花肉', '肥肉', '肥牛', '猪油', '黄油', '奶油', '培根',
                             '腊肉', '腊肠', '肥肠', '猪蹄', '蹄髈', '羊油', '牛油', '酥肉',
                             '猪皮', '鸡皮', '油渣']
            filtered = [
                r for r in filtered
                # 高脂做法（油炸/红烧/油焖/干锅/麻辣香锅等）查菜名+描述+做法+标签，
                # 避免"油焖小龙虾""家常麻辣香锅"这类菜名体现重油但描述未提及而漏网
                if not any(m in str(r.get('name', '')).lower() or
                          m in str(r.get('description', '')).lower() or
                          m in str(r.get('method', '')).lower() or
                          m in ' '.join(str(t) for t in r.get('tags', [])).lower()
                          for m in high_fat_methods)
                and not any(f in ' '.join(self._safe_ingredients(r)) or
                            f in str(r.get('name', '')).lower()
                            for f in high_fat_ings)
            ]
        
        if filters.get('low_spicy'):
            spicy_keywords = ['辣椒', '花椒', '辣', '麻辣', '香辣']
            filtered = [
                r for r in filtered
                if not any(s in str(r.get('name', '')).lower() or
                          s in str(r.get('description', '')).lower() or
                          s in str(r.get('method', '')).lower() or
                          s in ' '.join(str(t) for t in r.get('tags', [])).lower() or
                          s in ' '.join(self._safe_ingredients(r))
                          for s in spicy_keywords)
            ]
        
        if filters.get('low_sugar'):
            # 甜味来源：添加糖（糖/蜂蜜/冰糖/白糖/甜点）或食材本身带甜（紫薯/芋泥/红薯/山药/香蕉/芒果）
            sweet_addeds = ['糖', '蜂蜜', '冰糖', '白糖', '甜点']
            sweet_ints = ['紫薯', '芋泥', '红薯', '地瓜', '山药', '南瓜', '香蕉', '芒果', '枣泥', '红豆沙']
            filtered = [
                r for r in filtered
                # 用 _safe_ingredients（兼容字符串/列表）判断，避免 ingredients 为字符串
                # 时逐字符遍历导致"糖/蜂蜜"等甜味词检测失效
                if not any(s in ' '.join(self._safe_ingredients(r))
                          for s in sweet_addeds)
                # 名称/标签/描述含"甜/甜品"或配料含高甜食材 → 排除（如"紫薯芋泥切糕"标"甜"）
                if not (any(f in str(r.get('name', '')).lower()
                           or f in ' '.join([str(t).lower() for t in r.get('tags', [])])
                           or f in str(r.get('description', '')).lower()
                           for f in ['甜', '甜点', '甜品', '糖']) or
                        any(f in ' '.join(self._safe_ingredients(r))
                            for f in sweet_ints))
            ]
        
        if filters.get('vegetarian'):
            # 荤菜判定：覆盖具体肉种（猪肉/牛肉/...）+ 通用"鱼/肉"及常见海鲜水产关键词
            meat_keywords = ['猪', '牛', '羊', '鸡', '鸭', '鹅', '鱼', '虾', '蟹', '海鲜',
                             '肉', '培根', '腊肉', '腊肠', '烤肠', '热狗', '香肠', '火腿',
                             '排骨', '鸡翅', '鸡腿', '肉末', '肉沫', '肥牛', '肥羊', '羊排',
                             '乳鸽', '鸽', '牛蛙', '蛙', '蛇', '鳗', '章鱼', '鱿鱼', '墨鱼',
                             '扇贝', '蛤蜊', '花蛤', '蛤', '蛏', '生蚝', '蚝', '贝', '海螺',
                             '熏肉', '叉烧', '午餐肉',
                             '带鱼', '三文鱼', '鳕鱼', '金枪鱼']
            filtered = [
                r for r in filtered
                if not any(m in ' '.join(self._safe_ingredients(r))
                          for m in meat_keywords)
                # 名称含明显荤菜字眼（如"咸肉蒸蛋""广式腊肠""天麻炖乳鸽"）也排除
                and not re.search(r'(肉|鱼|虾|蟹|排骨|鸡|鸭|猪|牛|羊|火腿|香肠|腊肠|烤肠|热狗|鸽|蛙|鳗)',
                                  str(r.get('name', '')))
            ]
        
        # 高蛋白目标：只保留含优质蛋白食材的菜品
        if filters.get('high_protein'):
            protein_kws = ['鸡胸', '鸡', '牛肉', '牛', '鱼', '虾', '蟹', '蛋', '瘦肉',
                           '三文鱼', '鳕鱼', '鲈鱼', '豆腐', '豆', '虾仁', '龙利鱼']
            filtered = [
                r for r in filtered
                if any(p in ' '.join(self._safe_ingredients(r)) or
                       p in str(r.get('name', ''))
                       for p in protein_kws)
            ]
        
        # 想吃荤/肉偏好：只保留含肉类的菜品（鱼等已在上游 exclude_ingredients 单独排除）
        # 注意：用两字荤词，避免单字"鸡/鸭/鹅/鱼"误匹配"鸡蛋/鱼香茄子"等纯素菜
        if filters.get('meat'):
            meat_kws = ['猪肉', '牛肉', '羊肉', '鸡肉', '鸭肉', '鹅肉', '鱼肉',
                        '虾', '蟹', '贝', '螺', '生蚝', '花蛤', '蛤蜊', '蛏', '扇贝',
                        '排骨', '培根', '腊肉', '腊肠', '香肠', '火腿', '腊鸭',
                        '鸡腿', '鸡翅', '鸡胸', '鸡胗', '鸡爪', '猪蹄', '猪肚', '猪肝',
                        '牛腩', '牛百叶', '羊排', '五花肉', '肉末', '肉沫', '肉丝', '肉片',
                        '红烧肉', '卤肉', '牛筋', '狮子头',
                        '乳鸽', '牛蛙', '蛇', '鳗', '鱿鱼', '章鱼', '墨鱼',
                        '带鱼', '鲈鱼', '鲫鱼', '三文鱼', '鳕鱼', '金枪鱼',
                        '鸭血', '鹅肝', '牛', '猪', '肉']
            filtered = [
                r for r in filtered
                if any(m in ' '.join(self._safe_ingredients(r)) or
                       re.search(m, str(r.get('name', '')))
                       for m in meat_kws)
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
        
        for ingredient in self._safe_ingredients(recipe):
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
                ings = ' '.join(self._safe_ingredients(recipe))
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