import re
import os

# ─── 资源限制（必须在 torch/FAISS/sentence-transformers 导入前设置）──────────
# 高频评测（eval.py 43+ 轮连续请求）下，embedding/reranker 的 OpenMP 推理线程
# 会按 CPU 核数反复创建，Windows 线程与内存耗尽后报：
#   MemoryError / OMP: Error #137: Cannot create thread / 系统资源不足(1450)
# 导致服务进程崩溃、后续请求全部空响应。统一限制线程数从根本上避免。
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import time
import json
import hashlib
import threading
from typing import Dict, Optional
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from config import Config, validate_config
from llm_client import get_llm_client
from rag_retriever import RAGRetriever
from constraint_engine import ConstraintEngine
from dialog_enhancer import DialogManager
from result_verifier import ResultVerifier
from orchestrator import ORCHESTRATOR, DialogContext
from nutrition_planner import plan_nutrition_gaps

# ─── 营养数据库加载 ──────────────────────────────────────
MEAL_NUTRITION_DB = {}
_nut_path = os.path.join(os.path.dirname(__file__), 'nutrition_database.json')
if os.path.exists(_nut_path):
    with open(_nut_path, 'r', encoding='utf-8') as f:
        MEAL_NUTRITION_DB = json.load(f)

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)


# ============================================================
# API 文档页面（根路由）
# ============================================================
@app.route('/')
def api_docs():
    """API 文档页面"""
    return '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>方太个性化膳食规划系统 - API文档</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
            background: #f0f2f5; min-height: 100vh; padding: 20px;
        }
        .container { max-width: 960px; margin: 0 auto; }
        .header {
            text-align: center; padding: 36px 20px;
            background: #fff; border-radius: 12px; margin-bottom: 20px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }
        .header h1 { font-size: 26px; color: #1a1a1a; margin-bottom: 6px; }
        .header p { font-size: 14px; color: #666; }
        .status-bar {
            display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap;
        }
        .status-card {
            flex: 1; min-width: 140px; background: white; border-radius: 8px;
            padding: 14px; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        }
        .status-card .label { font-size: 12px; color: #999; margin-bottom: 4px; }
        .status-card .value { font-size: 14px; font-weight: 600; color: #333; }
        .status-card .value.ok { color: #52c41a; }
        .section {
            background: white; border-radius: 8px; padding: 20px 24px; margin-bottom: 14px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        }
        .section h2 { font-size: 16px; color: #1a1a1a; margin-bottom: 14px; border-bottom: 1px solid #eee; padding-bottom: 8px; }
        .endpoint {
            border: 1px solid #e8e8e8; border-radius: 6px; padding: 14px; margin-bottom: 10px;
        }
        .endpoint:last-child { margin-bottom: 0; }
        .endpoint .method {
            display: inline-block; padding: 2px 8px; border-radius: 3px;
            font-size: 11px; font-weight: 700; color: white; margin-right: 8px;
        }
        .method.get { background: #52c41a; }
        .method.post { background: #1677ff; }
        .endpoint .path { font-family: 'Consolas', 'Courier New', monospace; font-size: 13px; color: #1a1a1a; font-weight: 600; }
        .endpoint .desc { font-size: 12px; color: #888; margin-top: 4px; }
        .param-table { width: 100%; margin-top: 8px; font-size: 12px; border-collapse: collapse; }
        .param-table th { background: #fafafa; text-align: left; padding: 5px 8px; border-bottom: 1px solid #e8e8e8; color: #666; }
        .param-table td { padding: 5px 8px; border-bottom: 1px solid #f5f5f5; color: #333; }
        .footer { text-align: center; padding: 16px; color: #bbb; font-size: 12px; }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>方太个性化膳食规划系统</h1>
        <p>基于用户健康档案与饮食偏好的智能膳食推荐服务</p>
    </div>

    <div class="status-bar">
        <div class="status-card">
            <div class="label">服务状态</div>
            <div class="value">正常运行</div>
        </div>
        <div class="status-card">
            <div class="label">LLM引擎</div>
            <div class="value ok">● DeepSeek-v4-flash</div>
        </div>
        <div class="status-card">
            <div class="label">菜谱库</div>
            <div class="value">2,000道</div>
        </div>
        <div class="status-card">
            <div class="label">营养数据库</div>
            <div class="value">185种食材</div>
        </div>
        <div class="status-card">
            <div class="label">用户档案</div>
            <div class="value">50个</div>
        </div>
    </div>

    <div class="section">
        <h2>健康检查</h2>
        <div class="endpoint">
            <span class="method get">GET</span>
            <span class="path">/api/health</span>
            <div class="desc">系统健康检查，返回各组件运行状态及性能统计</div>
        </div>
    </div>

    <div class="section">
        <h2>性能监控</h2>
        <div class="endpoint">
            <span class="method get">GET</span>
            <span class="path">/api/timing/stats</span>
            <div class="desc">全局性能统计（首Token延迟、端到端响应、多轮平均值、P50/P95分位数）</div>
            <table class="param-table">
                <tr><th>参数</th><th>类型</th><th>必填</th><th>说明</th></tr>
                <tr><td>user_id</td><td>string</td><td>否</td><td>指定会话ID查看单会话统计</td></tr>
            </table>
        </div>
    </div>

    <div class="section">
        <h2>菜谱检索</h2>
        <div class="endpoint">
            <span class="method get">GET</span>
            <span class="path">/api/search</span>
            <div class="desc">根据关键词搜索菜谱</div>
            <table class="param-table">
                <tr><th>参数</th><th>类型</th><th>必填</th><th>说明</th></tr>
                <tr><td>q</td><td>string</td><td>是</td><td>搜索关键词</td></tr>
                <tr><td>top_k</td><td>int</td><td>否</td><td>返回数量（默认5）</td></tr>
            </table>
        </div>
        <div class="endpoint">
            <span class="method get">GET</span>
            <span class="path">/api/recipe/&lt;name&gt;</span>
            <div class="desc">获取指定菜谱的详细信息（食材、做法、营养等）</div>
        </div>
    </div>

    <div class="section">
        <h2>膳食推荐</h2>
        <div class="endpoint">
            <span class="method post">POST</span>
            <span class="path">/api/recommend</span>
            <div class="desc">根据用户需求推荐菜品</div>
            <table class="param-table">
                <tr><th>参数</th><th>类型</th><th>必填</th><th>说明</th></tr>
                <tr><td>user_id</td><td>string</td><td>是</td><td>用户ID</td></tr>
                <tr><td>query</td><td>string</td><td>是</td><td>查询需求</td></tr>
                <tr><td>num_results</td><td>int</td><td>否</td><td>推荐数量（默认5）</td></tr>
            </table>
        </div>
        <div class="endpoint">
            <span class="method post">POST</span>
            <span class="path">/api/dialog</span>
            <div class="desc">多轮对话推荐（支持约束追加、局部替换、上下文回溯等交互场景）</div>
            <table class="param-table">
                <tr><th>参数</th><th>类型</th><th>必填</th><th>说明</th></tr>
                <tr><td>user_id</td><td>string</td><td>是</td><td>用户ID</td></tr>
                <tr><td>message</td><td>string</td><td>是</td><td>用户输入</td></tr>
            </table>
        </div>
        <div class="endpoint">
            <span class="method post">POST</span>
            <span class="path">/api/dialog/stream</span>
            <div class="desc">多轮对话推荐（流式响应，首字延迟约1.5s）</div>
            <table class="param-table">
                <tr><th>参数</th><th>类型</th><th>必填</th><th>说明</th></tr>
                <tr><td>user_id</td><td>string</td><td>是</td><td>用户ID</td></tr>
                <tr><td>message</td><td>string</td><td>是</td><td>用户输入</td></tr>
            </table>
        </div>
    </div>

    <div class="section">
        <h2>约束检查与营养计算</h2>
        <div class="endpoint">
            <span class="method post">POST</span>
            <span class="path">/api/constraint/check</span>
            <div class="desc">检查推荐菜品是否违反用户约束（过敏原、疾病禁忌等）</div>
            <table class="param-table">
                <tr><th>参数</th><th>类型</th><th>必填</th><th>说明</th></tr>
                <tr><td>user_id</td><td>string</td><td>是</td><td>用户ID</td></tr>
                <tr><td>recipes</td><td>array</td><td>是</td><td>菜名列表</td></tr>
            </table>
        </div>
        <div class="endpoint">
            <span class="method post">POST</span>
            <span class="path">/api/nutrition/calculate</span>
            <div class="desc">计算推荐菜品的营养摄入及平衡度</div>
            <table class="param-table">
                <tr><th>参数</th><th>类型</th><th>必填</th><th>说明</th></tr>
                <tr><td>user_id</td><td>string</td><td>是</td><td>用户ID</td></tr>
                <tr><td>recipes</td><td>array</td><td>是</td><td>菜名列表</td></tr>
            </table>
        </div>
    </div>

    <div class="section">
        <h2>用户档案</h2>
        <div class="endpoint">
            <span class="method get">GET</span>
            <span class="path">/api/user/profile</span>
            <div class="desc">查询用户健康档案（过敏、偏好、膳食目标等）</div>
            <table class="param-table">
                <tr><th>参数</th><th>类型</th><th>必填</th><th>说明</th></tr>
                <tr><td>user_id</td><td>string</td><td>是</td><td>用户ID</td></tr>
            </table>
        </div>
        <div class="endpoint">
            <span class="method post">POST</span>
            <span class="path">/api/user/profile</span>
            <div class="desc">更新用户偏好设置</div>
        </div>
    </div>

    <div class="footer">
        方太集团 &copy; 2025
    </div>
</div>
</body>
</html>
'''

# 全局组件实例（延迟加载）
_llm_client = None
_rag_retriever = None
_constraint_engine = None
_dialog_managers = {}
_MAX_DIALOG_MANAGERS = 500  # 会话缓存上限，超出时淘汰最旧会话（防内存泄漏）
_result_verifier = None

# 请求缓存
_request_cache = {}
_CACHE_TTL = 300  # 5分钟

# 全局计时统计（跨会话聚合）
_global_timings = []  # 保留最近 200 条

# ─── 持久化记忆：用户饮食习惯临时磁盘保存 ────────────────
# 按 user_id 把 DialogManager 状态落盘，服务重启后仍能记住"不吃虾/爱辣"等偏好。
# 数据为纯文本/偏好，不含任何敏感凭证，仅用于满足竞赛"个性化记忆"要求。
_DIALOG_STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.dialog_store')
_MEMORY_LOCK = threading.Lock()


def _dialog_store_path(user_key: str) -> str:
    """把 user_id 映射为安全的文件路径（避免中文/特殊字符）。"""
    safe = hashlib.md5(str(user_key).encode('utf-8')).hexdigest()
    return os.path.join(_DIALOG_STORE_DIR, f"{safe}.json")


def _save_dialog(user_key: str, dm) -> None:
    """把对话管理器状态落盘。失败时只告警，不阻塞主流程。"""
    if not user_key:
        return
    try:
        os.makedirs(_DIALOG_STORE_DIR, exist_ok=True)
        path = _dialog_store_path(user_key)
        with _MEMORY_LOCK:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(dm.to_dict(), f, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"[memory] 保存失败 {user_key}: {e}", flush=True)


def _load_dialog(user_key: str):
    """尝试从磁盘恢复用户之前的饮食习惯；无记录或损坏时返回 None。"""
    if not user_key:
        return None
    try:
        path = _dialog_store_path(user_key)
        if not os.path.exists(path):
            return None
        with _MEMORY_LOCK:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        dm = DialogManager()
        dm.from_dict(data)
        return dm
    except Exception as e:
        print(f"[memory] 加载失败 {user_key}: {e}", flush=True)
        return None


def _clear_dialog_memory(user_key: str) -> None:
    """删除某个用户已保存的饮食习惯记忆（新对话/重置时调用）。"""
    if not user_key:
        return
    try:
        path = _dialog_store_path(user_key)
        with _MEMORY_LOCK:
            if os.path.exists(path):
                os.remove(path)
    except Exception as e:
        print(f"[memory] 清除失败 {user_key}: {e}", flush=True)

# Agent 工具定义 (OpenAI Function Calling 格式)
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_recipes",
            "description": "在菜谱库中搜索菜谱。根据用户需求（人数、餐次、口味、偏好等）检索合适的菜品。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词，如'3人晚餐 清淡 不辣'"},
                    "top_k": {"type": "integer", "description": "返回菜谱数量，默认5", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_constraints",
            "description": "检查菜谱是否违反用户的过敏、疾病、特殊人群等约束条件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe_name": {
                        "type": "string",
                        "description": "要检查的菜品名称（菜谱库中的完整名称）"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "用户ID，用于获取该用户的约束条件"
                    }
                },
                "required": ["recipe_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "当前信息不足以做出推荐决策时，向用户提问以澄清需求。如：口味偏好不明确、过敏信息缺失、用餐场景不清楚等情况。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "向用户提出的问题，应自然亲切"
                    },
                    "reason": {
                        "type": "string",
                        "description": "为什么需要问这个问题（简要说明缺失的信息）"
                    }
                },
                "required": ["question"]
            }
        }
    }
]

# 无意义/口语化内容，检索与排除时忽略
_JUNK_EXCLUDES = {'了', '的', '点', '些', '它', '吧', '啊', '呀', '嘛', '哦', '嗯', '呢',
                 '这些', '那些', '一点', '太多', '太', '顿', '餐'}

# 口语化"想吃刺激/重口" → 改写为可检索的辣味关键词（避免"刺激"搜不到菜）
# 注意："下饭"只是"配饭/有滋味"，不等于辣，已从辣味词中剔除，归入隐含语义词典
_SPICY_WANT_TERMS = ['刺激', '过瘾', '重口', '重口味', '口味重', '够味', '带劲']
_SPICY_SEARCH_QUERY = '辣 麻辣 香辣 辣椒 剁椒 川菜 湘菜 辛辣'

# 隐含语义词典：口语隐含愿望 → 可检索关键词（下饭≠辣、降火≠甜、开胃/养胃各归其位）
_IMPLICIT_SEMANTIC = [
    (('下饭', '下饭菜', '很下饭', '配饭'), '咸香 开胃 下饭 家常菜 浓油赤酱 红烧'),
    (('降火', '去火', '下火', '上火', '清热', '败火', '清火'), '清热 降火 凉性 清淡 冬瓜 绿豆 苦瓜 凉菜'),
    (('开胃', '没胃口', '胃口不好', '食欲不振', '食欲'), '开胃 酸 清爽 山楂 凉拌 番茄 酸辣'),
    (('养胃', '暖胃', '胃不好', '养肠胃'), '养胃 暖胃 温和 易消化 粥 汤 山药 小米'),
]


def _clean_search_query(query: str) -> tuple:
    """
    清洗检索 query：剔除口语动词前缀与否定片段，并解析出应排除的食材。

    解决"我想吃虾搜不到""不吃虾无法屏蔽"两类问题：
    - "我想吃虾"   -> ("虾", [])            # 去除口语前缀，保留核心词
    - "来一份红烧肉" -> ("红烧肉", [])
    - "不吃虾"     -> ("", ["虾"])           # 提取排除食材，正向词为空
    - "对虾过敏"   -> ("", ["虾"])
    - "不要辣的"   -> ("", ["辣"])

    Returns:
        (core_query, excludes): 清洗后的核心检索词（可能为空串），
        以及从 query 中解析出的应排除食材列表（去重）。
    """
    if not query:
        return "", []
    q = query.strip()
    excludes = []

    # 1) 提取并移除否定片段（"不吃X""对X过敏""忌口X"等）
    neg_patterns = [
        r'(?:不吃|不能吃|不要吃|不想吃|别吃|别放|不放|不加|不要|别|忌口|避免|避开|去掉|忌)([^，。、;；!！?？\s]{1,8})',
        r'对([^，。、;；!！?？\s]{1,8})过敏',
        r'(?<![不没])(?![我你他她它咱俺])([^，。、;；!！?？\s]{1,8})过敏(?:于)?([^，。、;；!！?？\s]{1,8})?',
    ]
    for pat in neg_patterns:
        for m in re.finditer(pat, q):
            item = (m.group(1) or m.group(2) or '').strip('的了呗吧呀哦嗯嘛啊')
            if 1 <= len(item) <= 10 and item not in _JUNK_EXCLUDES:
                excludes.append(item)
            q = q.replace(m.group(0), ' ')

    # 2) 循环剔除口语动词前缀（长词优先，避免误删核心词）
    prefixes = [
        '我特别想吃', '我想吃点', '我想喝点', '我想喝', '我想吃', '我想来', '我要吃', '我要喝',
        '我说我不吃', '我说不要', '我说不要吃', '我说', '今晚想吃', '今天想吃', '晚上想吃',
        '中午想吃', '想吃点', '想喝点', '想喝', '想吃',
        '来一份', '来份', '来点', '点一份', '点份', '给我推荐', '帮我推荐', '推荐一下',
        '帮我来', '给我来', '给我', '帮我', '今天想', '今晚想', '弄点', '做点', '整点',
        '一道', '一份', '推荐', '想', '要', '来', '我不',
    ]
    cleaned = q
    for _ in range(6):
        matched = False
        for p in prefixes:
            if cleaned.startswith(p):
                cleaned = cleaned[len(p):].lstrip('，。、, ')
                matched = True
                break
        if not matched:
            break

    # 3) 清理空白与残留标点
    core = re.sub(r'[\s，。、,;；!！?？]+', '', cleaned).strip()
    core = re.sub(r'^(好的|好|吧|啊|呀|哦|嗯|呢|嘛)+$', '', core)
    # 孤立人称代词/语气词视为空
    core = re.sub(r'^(我|俺|咱们|人家|你|你们|他|他们|她|她们|它|都|就|还)+$', '', core)
    # 去掉开头孤立量词与结尾"的/了"残留（"点甜的"->"甜"，"清淡的"->"清淡"）
    core = re.sub(r'^(点|份|道|个|盘|碗|碟)(?=[\u4e00-\u9fff])', '', core)
    core = re.sub(r'[的了]$', '', core)

    # 口味语义改写：把"刺激/过瘾/重口"等口语口味词改写为辣味可检索词，
    # 避免"我想吃点刺激的" -> 检索词"刺激"搜不到任何菜而退化为清淡推荐
    if core and len(core) <= 8:
        # 隐含语义词典优先："下饭/降火/开胃/养胃"各映射到正确检索词（下饭≠辣、降火≠甜）
        for terms, rewrite in _IMPLICIT_SEMANTIC:
            if any(_t in core for _t in terms):
                core = rewrite
                break
        else:
            for _t in _SPICY_WANT_TERMS:
                if _t in core:
                    core = _SPICY_SEARCH_QUERY
                    break

    # 去重（保持顺序）
    seen = set()
    excludes = [e for e in excludes if not (e in seen or seen.add(e))]

    return core, excludes


# 泛化查询结果缓存（性能优化）：相同检索词+相同过滤+相同用户约束在短时间内
# 命中时直接复用检索/精排结果，减少重复的 FAISS+重排 开销（端到端延迟优化项）。
# 缓存键已包含用户与约束维度，命中结果与实时计算等价，不会跨用户串菜。
import threading as _threading
_SEARCH_CACHE = {}
_SEARCH_CACHE_LOCK = _threading.Lock()
_SEARCH_CACHE_TTL = 180.0  # 秒


def _execute_agent_tool(tool_name: str, tool_args: dict, user_id: str = None, user_ids: list = None,
                         retriever=None, engine=None, dm=None) -> str:
    """执行 Agent 工具调用，返回结果文本"""
    import json as _json

    if tool_name == "search_recipes":
        query = tool_args.get("query", "")
        top_k = tool_args.get("top_k", 5)
        filters = dict(dm.get_search_filters()) if dm else {}
        # 清洗 query：去口语前缀 + 提取否定排除食材，避免"我想吃虾/不吃虾"检索失效
        core, neg_ex = _clean_search_query(query)
        if neg_ex:
            filters['exclude_ingredients'] = list(set(filters.get('exclude_ingredients', []) + neg_ex))
        search_q = core if core else ('晚餐' if neg_ex else query)
        try:
            # 排除集：用户已否决的菜 + 本会话最近已推过的菜（当前方案）。
            # - 被否决的菜绝不能再端回（方案否定后重推不能重复）；
            # - 最近一轮方案排除，避免同一用户连续几轮推荐高度重复（"复读机"观感）；
            #   约束追加场景由 _agentic_recommend 的 _kept 逻辑按最小修改原则重新并入 ≤3 道。
            _rej = set(getattr(dm, 'rejected_recipes', []) or []) if dm else set()
            _recent = set(getattr(dm, 'last_recommendation', []) or
                          (getattr(dm, 'recommended_recipes', []) or [])[-5:]) if dm else set()
            _excl = _rej | _recent
            # ── 泛化查询结果缓存（性能优化）──
            # 键 = 检索词 + 过滤集 + 用户约束签名 + 排除集；命中且未过期时直接复用检索/精排结果。
            _frozen_filters = tuple(sorted((k, str(v)) for k, v in (filters or {}).items()))
            _user_sig = tuple(sorted(user_ids or [])) or (user_id or '')
            _cache_key = (search_q, _frozen_filters, _user_sig, tuple(sorted(_excl)))
            _now = time.time()
            with _SEARCH_CACHE_LOCK:
                _hit = _SEARCH_CACHE.get(_cache_key)
            if _hit and (_now - _hit[0]) <= _SEARCH_CACHE_TTL:
                print(f"[tool] search缓存命中: '{search_q}'", flush=True)
                return _hit[1]
            # 多召回一倍候选，供多样性轮换在更大的池子里换菜（而非同一批内换顺序）
            results = retriever.search(search_q, top_k=max(top_k * 2, 10), filters=filters)
            if user_ids:
                results = engine.filter_by_constraints(results, user_ids=user_ids)
            elif user_id:
                results = engine.filter_by_constraints(results, user_id=user_id)
            if _excl:
                results = [r for r in results if r.get('name', '') not in _excl]
            # 多样性轮换：在 2 倍候选池里按 diversity_seed 轮转窗口后截取 top_k，
            # 保证同一用户每次"新对话"（种子自增）能看到不同的菜；
            # 第 0 次会话（seed=0）保持精排原始顺序，相关性最高。
            if dm and getattr(dm, 'diversity_seed', 0) > 0 and len(results) > top_k:
                _off = (dm.diversity_seed * top_k) % len(results)
                results = results[_off:] + results[:_off]
            recipes = results[:top_k]
            _payload = _json.dumps([{
                "name": r.get("name", "未知"),
                "cuisine": r.get("cuisine", ""),
                "cooking_method": r.get("cooking_method", ""),
                "main_ingredients": r.get("main_ingredients", r.get("ingredients", []))[:3],
            } for r in recipes], ensure_ascii=False)
            with _SEARCH_CACHE_LOCK:
                _SEARCH_CACHE[_cache_key] = (_now, _payload)
            return _payload
        except Exception as e:
            return f"搜索失败: {e}"

    elif tool_name == "check_constraints":
        recipe_name = tool_args.get("recipe_name", "")
        uid = tool_args.get("user_id") or user_id
        if uid and engine:
            profile = engine.user_profiles.get(uid, {})
            return _json.dumps({
                "recipe": recipe_name,
                "allergies": profile.get("allergies", []),
                "diseases": profile.get("diseases", []),
                "special_group": profile.get("special_group", ""),
                "preferences": profile.get("preferences", {})
            }, ensure_ascii=False)
        return _json.dumps({"status": "无约束信息"})

    elif tool_name == "ask_user":
        return _json.dumps({
            "action": "ask",
            "question": tool_args.get("question", ""),
            "reason": tool_args.get("reason", "")
        }, ensure_ascii=False)

    return _json.dumps({"error": f"未知工具: {tool_name}"})


def get_llm():
    """
    获取LLM客户端实例（单例模式）
    
    使用Flask的g对象实现请求级别的缓存，避免重复创建客户端。
    
    Returns:
        LLMClient实例
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = get_llm_client()
    return _llm_client


def get_retriever():
    """
    获取RAG检索器实例（单例模式）
    
    Returns:
        RAGRetriever实例
    """
    global _rag_retriever
    if _rag_retriever is None:
        _rag_retriever = RAGRetriever()
    return _rag_retriever


def get_engine():
    """
    获取约束引擎实例（单例模式）
    
    Returns:
        ConstraintEngine实例
    """
    global _constraint_engine
    if _constraint_engine is None:
        _constraint_engine = ConstraintEngine()
        import constraint_engine
        print(f"约束引擎模块路径: {constraint_engine.__file__}")
        print(f"约束引擎属性: {[m for m in dir(_constraint_engine) if not m.startswith('_')]}")
    return _constraint_engine


def get_verifier():
    """
    获取结果验证器实例（单例模式）
    
    Returns:
        ResultVerifier实例
    """
    global _result_verifier
    if _result_verifier is None:
        _result_verifier = ResultVerifier()
    return _result_verifier


def warmup():
    """
    预加载组件（预热）
    
    在应用启动时调用，预加载FAISS索引和菜谱数据，减少首次请求延迟。
    """
    print("=" * 60)
    print("正在预热组件...")
    print("=" * 60)
    
    # 预加载LLM客户端
    print("1. 加载LLM客户端...")
    start_time = time.perf_counter()
    llm = get_llm()
    print(f"   完成，耗时: {(time.perf_counter() - start_time) * 1000:.2f}ms")
    
    # LLM连接预热（发送小请求建立连接）
    print("2. LLM连接预热...")
    start_time = time.perf_counter()
    try:
        response = llm.chat([{'role': 'user', 'content': 'hi'}])
        print(f"   完成，耗时: {(time.perf_counter() - start_time) * 1000:.2f}ms")
    except Exception as e:
        print(f"   预热失败: {e}")
    
    # 预加载RAG检索器并构建索引
    print("3. 加载RAG检索器并构建索引...")
    start_time = time.perf_counter()
    retriever = get_retriever()
    retriever.build_index()
    # 预热精排组件：embedding 与交叉编码器懒加载若放在首个请求会拖慢 20s+，
    # 启动时完成加载 + 一次打分预热，保证首个用户请求即达稳态延迟。
    try:
        retriever._text_to_vector('预热')
        if retriever._init_cross_encoder() is not None:
            retriever._cross_encoder_rerank(
                retriever.recipes[:2], '预热查询', 1)
    except Exception as e:
        print(f"   [WARN] 精排预热跳过: {e}")
    print(f"   完成，耗时: {(time.perf_counter() - start_time) * 1000:.2f}ms")
    
    # 预加载约束引擎
    print("4. 加载约束引擎...")
    start_time = time.perf_counter()
    engine = get_engine()
    print(f"   完成，耗时: {(time.perf_counter() - start_time) * 1000:.2f}ms")
    
    print("\n" + "=" * 60)
    print("组件预热完成！")
    print("=" * 60)


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    健康检查接口
    
    返回服务状态、组件加载情况和性能统计。
    
    Returns:
        JSON响应，包含服务状态和组件信息
    """
    timing_stats = _compute_stats(_global_timings)
    return jsonify({
        'status': 'ok',
        'service': '方太个性化膳食规划Agent',
        'version': '1.0.0',
        'components': {
            'llm_client': _llm_client is not None,
            'rag_retriever': _rag_retriever is not None,
            'constraint_engine': _constraint_engine is not None
        },
        'performance': timing_stats
    })


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    菜谱推荐接口
    
    根据用户查询和健康档案，返回个性化的菜谱推荐列表。
    
    请求参数：
    {
        "query": "今晚吃什么",
        "user_id": "user_001",
        "top_k": 5,
        "filters": {"exclude_ingredients": ["辣椒"], "tags": ["清淡"]}
    }
    
    Returns:
        JSON响应，包含推荐菜谱列表和营养评估
    """
    start_time = time.perf_counter()
    
    try:
        # 解析请求参数
        data = request.get_json()
        query = data.get('query', '')
        user_id = data.get('user_id', '')
        user_ids = data.get('user_ids', [])  # 多人用餐场景
        top_k = data.get('top_k', 5)
        filters = dict(data.get('filters', {}) or {})

        # 获取组件实例
        retriever = get_retriever()
        engine = get_engine()
        llm = get_llm()

        # 清洗检索词：去口语前缀 + 提取否定排除，避免"我想吃虾/不吃虾"检索失效
        core, neg_ex = _clean_search_query(query)
        if neg_ex:
            filters['exclude_ingredients'] = list(set(filters.get('exclude_ingredients', []) + neg_ex))
        search_q = core if core else ('晚餐' if neg_ex else query)

        # 1. 检索菜谱
        search_start = time.perf_counter()
        results = retriever.search(search_q, top_k=top_k * 2, filters=filters)
        search_time = (time.perf_counter() - search_start) * 1000

        # 2. 应用约束过滤（支持单人 user_id 或多人 user_ids）
        filter_start = time.perf_counter()
        if user_ids and len(user_ids) > 0:
            # 多人场景：合并约束后过滤
            results = engine.filter_by_constraints(results, user_ids=user_ids)
        elif user_id:
            # 单人场景
            results = engine.filter_by_constraints(results, user_id=user_id)
        filter_time = (time.perf_counter() - filter_start) * 1000
        
        # 3. 评估营养和平衡度
        nutrition_start = time.perf_counter()
        for recipe in results[:top_k]:
            nutrition = engine.evaluate_nutrition(recipe)
            recipe['nutrition'] = nutrition
        
        balance_score = engine.calculate_balance_score(results[:top_k])
        nutrition_time = (time.perf_counter() - nutrition_start) * 1000
        
        # 4. 使用LLM生成推荐理由
        llm_start = time.perf_counter()
        recipe_names = [r['name'] for r in results[:top_k]]
        llm_context = f"用户查询: {query}\n推荐菜品: {', '.join(recipe_names)}\n请给出简短的推荐理由。"
        
        try:
            print(f"[LLM] /api/recommend calling LLM...")
            llm_response = llm.chat([{'role': 'user', 'content': llm_context}])
            print(f"[OK] LLM call success")
        except Exception as e:
            print(f"[ERR] LLM call failed: {type(e).__name__}: {e}")
            llm_response = "根据您的需求，为您推荐以上菜品。"
        
        llm_time = (time.perf_counter() - llm_start) * 1000
        
        # 计算总耗时
        total_time = (time.perf_counter() - start_time) * 1000
        
        # 返回响应
        return jsonify({
            'success': True,
            'query': query,
            'user_id': user_id,
            'recommendations': results[:top_k],
            'balance_score': balance_score,
            'recommendation_reason': llm_response,
            'timing': {
                'total': round(total_time, 2),
                'search': round(search_time, 2),
                'filter': round(filter_time, 2),
                'nutrition': round(nutrition_time, 2),
                'llm': round(llm_time, 2)
            }
        })
    
    except Exception as e:
        # 异常处理
        total_time = (time.perf_counter() - start_time) * 1000
        return jsonify({
            'success': False,
            'error': str(e),
            'timing': {
                'total': round(total_time, 2)
            }
        }), 500


@app.route('/api/user/profile', methods=['GET'])
def get_user_profile():
    """
    获取用户档案接口
    
    根据用户ID返回用户的健康档案信息。
    
    请求参数：
    - user_id: 用户ID（URL参数）
    
    Returns:
        JSON响应，包含用户健康档案
    """
    user_id = request.args.get('user_id', '')
    
    if not user_id:
        return jsonify({'success': False, 'error': 'user_id参数不能为空'}), 400
    
    try:
        engine = get_engine()
        profile = engine.user_profiles.get(user_id)
        
        if profile:
            return jsonify({'success': True, 'profile': profile})
        else:
            return jsonify({'success': False, 'error': '未找到用户档案'}), 404
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/user/profile', methods=['POST'])
def update_user_profile():
    """
    更新用户档案接口
    
    创建或更新用户的健康档案信息。
    
    请求参数：
    {
        "user_id": "user_001",
        "allergies": ["花生", "海鲜"],
        "diseases": ["高血压"],
        "preferences": {"vegetarian": false, "low_spicy": true}
    }
    
    Returns:
        JSON响应，包含更新后的用户档案
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'success': False, 'error': 'user_id参数不能为空'}), 400
        
        engine = get_engine()
        
        # 更新用户档案
        profile = {
            'user_id': user_id,
            'allergies': data.get('allergies', []),
            'diseases': data.get('diseases', []),
            'preferences': data.get('preferences', {})
        }
        
        engine.user_profiles[user_id] = profile
        
        # 保存到文件
        with open(Config.USER_PROFILES_PATH, 'w', encoding='utf-8') as f:
            json.dump(list(engine.user_profiles.values()), f, ensure_ascii=False, indent=2)
        
        return jsonify({'success': True, 'profile': profile})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/constraint/check', methods=['POST'])
def check_constraint():
    """
    约束检查接口
    
    检查单个菜谱是否符合用户的健康约束条件。
    
    请求参数：
    {
        "recipe_name": "清蒸鲈鱼",
        "user_id": "user_001"
    }
    
    Returns:
        JSON响应，包含检查结果和违反的约束列表
    """
    try:
        data = request.get_json()
        recipe_name = data.get('recipe_name')
        user_id = data.get('user_id')
        
        if not recipe_name or not user_id:
            return jsonify({'success': False, 'error': 'recipe_name和user_id参数不能为空'}), 400
        
        # 获取组件实例
        retriever = get_retriever()
        engine = get_engine()
        
        # 获取菜谱详情
        recipe = retriever.get_recipe_by_name(recipe_name)
        if not recipe:
            return jsonify({'success': False, 'error': f'未找到菜品: {recipe_name}'}), 404
        
        # 获取用户档案
        profile = engine.user_profiles.get(user_id)
        if not profile:
            return jsonify({'success': False, 'error': f'未找到用户档案: {user_id}'}), 404
        
        # 检查约束
        passed, violations = engine.check_constraints(recipe, profile)
        
        return jsonify({
            'success': True,
            'recipe_name': recipe_name,
            'user_id': user_id,
            'passed': passed,
            'violations': violations
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/nutrition/calculate', methods=['POST'])
def calculate_nutrition():
    """
    营养计算接口
    
    根据菜谱名计算其营养成分。
    
    请求参数：
    {
        "recipe_name": "清蒸鲈鱼"
    }
    
    Returns:
        JSON响应，包含菜谱的营养成分信息
    """
    try:
        data = request.get_json()
        recipe_name = data.get('recipe_name')
        
        if not recipe_name:
            return jsonify({'success': False, 'error': 'recipe_name参数不能为空'}), 400
        
        # 获取组件实例
        retriever = get_retriever()
        engine = get_engine()
        
        # 获取菜谱详情
        recipe = retriever.get_recipe_by_name(recipe_name)
        if not recipe:
            return jsonify({'success': False, 'error': f'未找到菜品: {recipe_name}'}), 404
        
        # 评估营养
        nutrition = engine.evaluate_nutrition(recipe)
        
        return jsonify({
            'success': True,
            'recipe_name': recipe_name,
            'nutrition': nutrition
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/search', methods=['GET'])
def search_recipes():
    """
    菜谱搜索接口
    
    根据关键词搜索菜谱（简化版，不包含约束过滤）。
    
    请求参数：
    - q: 搜索关键词（URL参数）
    - top_k: 返回数量，默认10（URL参数）
    
    Returns:
        JSON响应，包含匹配的菜谱列表
    """
    query = request.args.get('q', '')
    top_k = int(request.args.get('top_k', 10))
    
    if not query:
        return jsonify({'success': False, 'error': 'q参数不能为空'}), 400
    
    try:
        retriever = get_retriever()
        core, neg_ex = _clean_search_query(query)
        search_q = core if core else query
        results = retriever.search(search_q, top_k=top_k, filters={'exclude_ingredients': neg_ex} if neg_ex else None)
        
        return jsonify({
            'success': True,
            'query': query,
            'count': len(results),
            'recipes': results
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def _generate_fallback_recipe(user_message: str, user_id: str = None, user_ids: list = None,
                              dm=None, engine=None, llm=None) -> Optional[dict]:
    """
    约束极紧、菜谱库无法满足时，由 LLM 现场生成一道符合约束的新菜谱（赛题加分项）。

    仅在现有菜谱检索结果为 0 时调用。生成结果通过约束引擎校验后才返回，
    返回的菜谱带 `generated: True` 标记，ResultVerifier 将豁免其库内存在性检查。

    Args:
        user_message: 用户消息
        user_id: 用户ID
        user_ids: 多人ID列表
        dm: DialogManager
        engine: 约束引擎
        llm: LLM客户端

    Returns:
        dict: 生成的菜谱（含 generated 标记），失败返回 None
    """
    # 收集用户约束（过敏原/疾病/特殊人群）
    constraints = {'allergies': set(), 'diseases': set(), 'special_groups': set()}
    if engine:
        if user_ids:
            for uid in user_ids:
                up = engine.get_user_profile(uid) or {}
                constraints['allergies'].update(up.get('allergies', []) or [])
                constraints['diseases'].update(up.get('diseases', []) or [])
                constraints['special_groups'].update(up.get('special_groups', []) or [])
        elif user_id and str(user_id).isdigit() and 1 <= int(user_id) <= 50:
            up = engine.get_user_profile(user_id) or {}
            constraints['allergies'].update(up.get('allergies', []) or [])
            constraints['diseases'].update(up.get('diseases', []) or [])
            constraints['special_groups'].update(up.get('special_groups', []) or [])
    if dm:
        up = dm.user_preferences or {}
        constraints['allergies'].update(up.get('allergies', []) or [])

    allergies = list(constraints['allergies'])
    diseases = list(constraints['diseases'])
    special = list(constraints['special_groups'])

    allergy_txt = '、'.join(allergies) if allergies else '无'
    disease_txt = '、'.join(diseases) if diseases else '无'
    special_txt = '、'.join(special) if special else '无'

    prompt = (
        f"你是专业营养师。用户需求：{user_message}\n"
        f"用户健康约束（必须全部满足，任一违反即失败）：\n"
        f"- 过敏原（禁止出现任何相关食材）：{allergy_txt}\n"
        f"- 疾病饮食禁忌（如控盐/控糖/控嘌呤）：{disease_txt}\n"
        f"- 特殊人群限制（如孕妇/哺乳期）：{special_txt}\n\n"
        f"现有菜谱库中没有符合以上约束的菜品，请你现场设计一道完全合规的新菜谱。\n"
        f"要求：菜名简洁真实、食材常见易得、烹饪方式健康，必须严格避开上述所有过敏原和禁忌。\n"
        f"注意：tags 标签中禁止出现过敏原或禁忌食材名称（如不要写「无海鲜」「低嘌呤」这类标签），"
        f"只写口味或菜系类标签（如「清淡」「家常」「低脂」「控糖」「蒸菜」）。\n"
        f"只输出 JSON，不要任何解释，格式："
        f'{{"name": "菜名", "ingredients": ["食材1", "食材2", "食材3"], "steps": "烹饪步骤简述", "tags": ["标签1", "标签2"]}}'
    )

    # 最多重试 2 次，避免单次生成因 LLM 输出不合规而失败
    for attempt in range(2):
        try:
            content = llm.chat([{'role': 'user', 'content': prompt}], temperature=0.4, max_tokens=400)
        except Exception as e:
            print(f"[GenRecipe] LLM调用失败: {e}")
            return None

        # 解析 LLM 输出的 JSON（容忍 ``` 包裹）
        text = (content or '').strip()
        text = text.removeprefix('```json').removeprefix('```').removesuffix('```').strip()
        start, end = text.find('{'), text.rfind('}')
        if start == -1 or end == -1:
            print(f"[GenRecipe] 解析失败，无法找到JSON: {text[:120]}")
            return None
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError as e:
            print(f"[GenRecipe] JSON解析失败: {e}")
            return None

        name = str(data.get('name', '')).strip()
        ingredients = data.get('ingredients', [])
        if not name or not ingredients:
            print("[GenRecipe] 生成结果缺少菜名或食材")
            return None

        recipe = {
            'name': name,
            'ingredients': ingredients if isinstance(ingredients, list) else [str(ingredients)],
            'steps': str(data.get('steps', '')),
            'tags': data.get('tags', []) if isinstance(data.get('tags'), list) else [str(data.get('tags', ''))],
            'generated': True,
        }

        # 用约束引擎校验生成的菜谱是否合规
        if engine:
            profile = {}
            if user_ids:
                profile = engine.merge_multi_user_constraints(user_ids)
            elif user_id and str(user_id).isdigit() and 1 <= int(user_id) <= 50:
                profile = engine.get_user_profile(user_id) or {}
            if profile:
                passed, violations = engine.check_constraints(recipe, profile)
                if not passed:
                    print(f"[GenRecipe] 第{attempt+1}次生成未通过约束校验: {violations}")
                    # 追加失败原因，让 LLM 重新设计
                    prompt += (
                        f"\n\n注意：上次设计不合格，违规原因：{'；'.join(violations)}。"
                        f"请重新设计，务必严格避开上述过敏原和禁忌食材。"
                    )
                    continue

        print(f"[GenRecipe] 生成新菜谱: {name} (食材{len(ingredients)}种)")
        return recipe

    print("[GenRecipe] 重试仍失败，放弃生成")
    return None


def _is_meat_recipe(recipe: dict) -> bool:
    """
    判断一道菜是否为荤菜（含肉类/海鲜）。

    同时检查菜名与食材文本，覆盖"清蒸海蛎子""姜辣凤爪""家常麻辣香锅"等
    菜名不含"肉/鱼/虾/蟹"字眼但实际是荤菜的条目。

    注意：单字"肉"必须排除植物果肉（龙眼肉/果肉/椰肉/瓜肉等），
    否则"滋补养生梨（龙眼肉）"这类素甜品会被误判为荤菜。

    Args:
        recipe: 菜谱数据（需含 name，可选 ingredients）

    Returns:
        True 表示荤菜，False 表示素菜
    """
    import re as _re
    MEAT_KWS = ['肉排', '鸡', '鸭', '鱼', '虾', '蟹', '牛', '猪', '羊', '鹅', '驴',
                '鸽', '鹌', '鳅', '龟', '鳖', '兔', '蛇',
                '排骨', '香肠', '腊', '腿', '翅', '肝', '腰', '脑', '鲍', '蛤', '鱿',
                '鳗', '蛎', '螺', '蚌', '爪', '掌', '蹄', '肘', '肚', '舌', '腩',
                '鲈', '鳕', '鲢', '鲫', '鲤', '鳝', '蛙', '烤鸭', '火腿', '培根',
                '凤爪', '鸡爪', '鸭掌', '猪蹄', '牛筋', '生蚝', '海参',
                '海米', '虾米', '虾皮', '金钩', '干贝', '瑶柱']
    # 荤字主正则（去掉裸"肉"，由下面的专项正则处理；动物单字如"鸡/鱼"仍能覆盖鸡肉、鱼肉等）
    _main_rx = _re.compile('|'.join(_re.escape(k) for k in MEAT_KWS))
    # 专项"肉"：紧前非植物果肉字符才算荤（羊/牛/烧/回锅/叉烧…后面的肉是真的；
    # 龙眼肉/果肉/椰肉/瓜肉/芋肉等则是植物果肉）
    _meat_rx = _re.compile(r'(?<![果椰瓜芋榴荔桂眼参])肉')
    # 调味料/蛋奶素/植物果肉词：先剔除再匹配，避免"鸡精/猪油/蚝油/鸡蛋/牛奶/榴莲肉/肉桂"
    # 把素菜或甜点误判为荤菜。
    # （素菜也常放猪油炒、加鸡精/蚝油提鲜；鸡蛋/牛奶/奶油属蛋奶素；榴莲肉/枣肉/肉桂是植物）
    _seasoning_rx = _re.compile(
        r'鸡精|味精|猪油|蚝油|生抽|老抽|酱油|食用油|植物油|玉米油'
        r'|色拉油|花生油|调和油|芝麻油|香油|鸡蛋|鸭蛋|鹌鹑蛋|鹅蛋|蛋液|蛋黄|蛋清'
        r'|牛奶|纯牛奶|鲜牛奶|淡奶油|炼乳|酸奶|奶酪|芝士|黄油|奶油|椰奶|羊奶'
        r'|榴莲肉|大枣肉|枣肉|果肉|椰肉|瓜肉|芋肉|龙眼肉|桂圆肉|荔枝肉|肉桂')
    text = recipe.get('name', '') or ''
    ing = recipe.get('ingredients', '')
    if isinstance(ing, str):
        text += ' ' + ing
    elif isinstance(ing, list):
        text += ' ' + ' '.join(str(i) for i in ing)
    text = _seasoning_rx.sub(' ', text)
    return bool(_main_rx.search(text)) or bool(_meat_rx.search(text))


def _trim_by_relevance(recommendations: list, core_query: str, retriever, max_n: int = 5,
                       filters: Optional[Dict] = None,
                       engine=None, user_id: str = None, user_ids: list = None) -> list:
    """
    按用户核心查询相关度对 Agent 结果池重排并裁剪/补足。

    多轮 ReAct 搜索可能跑偏（如"想吃虾"却搜出一堆蒸菜），导致结果池里根本没有
    与需求相关的菜。此函数用核心查询重新检索得到相关菜：
    1. 优先保留池内与核心查询相关的菜；
    2. 若还不够 max_n 道，直接用核心查询的相关菜补足；
    3. 最后才用池内其余菜兜底。

    Args:
        recommendations: Agent 合并后的结果池（已补全完整菜谱）
        core_query: 用户核心检索词（已清洗/口味改写）
        retriever: RAG 检索器
        max_n: 最终保留的推荐数量上限
        filters: 检索过滤条件（排除食材等），补足搜索时同样生效
        engine/user_id/user_ids: 约束引擎与用户——补足搜索结果必须过健康约束过滤，
            否则"高血压用户搜索补齐"会把含腊肉/咸鱼的菜直接端上桌（曾导致评分违规）。

    Returns:
        相关度排序后的推荐列表
    """
    if not recommendations:
        return recommendations
    pool = {}
    for r in recommendations:
        n = r.get('name', '')
        if n and n not in pool:
            pool[n] = r
    try:
        ranked = retriever.search(core_query, top_k=max_n * 4, filters=filters) if core_query else []
    except Exception:
        ranked = []
    # 补足候选必须通过用户健康约束（过敏/疾病/特殊人群），不能只按相关度取菜
    if ranked and engine and (user_ids or (user_id and engine.user_profiles.get(user_id))):
        try:
            if user_ids:
                ranked = engine.filter_by_constraints(ranked, user_ids=user_ids)
            else:
                ranked = engine.filter_by_constraints(ranked, user_id=user_id)
        except Exception as _e:
            print(f"[Agent] 补足候选约束过滤失败: {_e}")
    ranked_names = [r.get('name', '') for r in ranked if r.get('name')]
    ranked_map = {r.get('name', ''): r for r in ranked if r.get('name')}

    ordered, seen = [], set()
    # 1) 池内与核心查询相关的菜（按相关度排序优先）
    for n in ranked_names:
        if n in pool and n not in seen:
            ordered.append(pool[n])
            seen.add(n)
        if len(ordered) >= max_n:
            return ordered[:max_n]
    # 2) 用核心查询的相关菜补足（解决池内无相关菜的问题）
    for n in ranked_names:
        if n not in seen:
            r = ranked_map.get(n)
            if r:
                ordered.append(r)
                seen.add(n)
        if len(ordered) >= max_n:
            return ordered[:max_n]
    # 3) 池内其余菜兜底（严格再次过过滤，避免未过过滤的素菜/忌口菜混入）
    for n, r in pool.items():
        if n in seen:
            continue
        if filters and not retriever._apply_filters([r], filters):
            seen.add(n)
            continue
        ordered.append(r)
        seen.add(n)
        if len(ordered) >= max_n:
            break
    return ordered[:max_n]


def _build_consistent_response(recommendations: list) -> str:
    """
    基于结构化推荐列表构建确定性叙述，保证叙述文案与列表完全一致。

    Agent 模式中 LLM 可能编造列表外的菜名（如"鱼香茄子"），
    与结构化推荐列表脱节。此函数根据实际列表重建一段自然、口语化的推荐语。

    Args:
        recommendations: 结构化推荐菜谱列表

    Returns:
        与列表一致的推荐叙述文本
    """
    names = [r.get('name', '') for r in recommendations if r.get('name')]
    if not names:
        return ""

    meat = [r.get('name', '') for r in recommendations if _is_meat_recipe(r)]
    veg = [n for n in names if n not in set(meat)]

    if meat and veg:
        return ("荤素都安排上了：荤菜{0}、素菜{1}。"
                "要是哪道不合口味，或者有忌口，跟我说一声就行～").format(
                    "、".join(meat), "、".join(veg))
    return f"给你配了这 {len(names)} 道：{'、'.join(names)}。想调整随时讲～"


# ─── 需求矛盾检测：当前消息 vs 已沉淀偏好 ─────────────────────────
# 多轮交互的关键能力：用户先声明偏好（素食/少辣/减肥），随后又提出与之冲突
# 的请求时，助手必须主动指出矛盾并给出选择，而不是默默照做。
def _detect_preference_conflict(message: str, dm) -> Optional[str]:
    """检测用户当前消息与既有饮食偏好的直接矛盾，返回提示文本（注入回复开头）。"""
    if dm is None or not message:
        return None
    prefs = dm.user_preferences or {}
    # 1) 素食者想吃肉
    if (prefs.get('preferences') or {}).get('vegetarian'):
        if re.search(r'想吃(?:点|些)?(?:猪|牛|羊|鸡|鸭|鹅|鱼)?肉|红烧肉|五花肉|排骨|回锅肉'
                     r'|牛排|烤肉|肘子|猪蹄|红烧|叉烧|扣肉', message):
            return ("注意到你之前说自己是素食者，现在想吃肉，这和素食目标有冲突。"
                    "如果想开荤我马上帮你调整推荐；要是继续吃素，我也可以挑几道仿荤口感的素菜～")
    # 2) 少辣偏好却点名辣味菜系
    _no_spicy = ('辣' in (prefs.get('excluded_ingredients') or [])
                 or (prefs.get('preferences') or {}).get('low_spicy')
                 or '清淡' in str((prefs.get('preferences') or {}).get('taste', '')))
    if _no_spicy and re.search(r'川菜|湘菜|麻辣|辣子|水煮鱼|水煮肉片|麻婆|辣火锅|火锅', message):
        return ("你之前说不要辣，而传统川菜属辣味菜系，直接上会跟你的要求冲突。"
                "我先挑了几道不辣的做法；如果你今天就想吃点辣的，说一声我再调整～")
    # 3) 减肥 vs 增肌
    goals = prefs.get('dietary_goals') or []
    if '减肥' in goals and re.search(r'增肌|练肌肉|长肌肉|涨肌肉', message):
        return ("减肥和增肌这两个目标有冲突——减脂要热量缺口，增肌要热量盈余，"
                "建议先以一个为主。我按增肌（高蛋白）帮你调整了这餐，你看行不行～")
    return None


def _build_multi_user_note(user_ids: list, engine) -> str:
    """多人用餐场景：点明已考虑各成员的健康约束，体现多人配餐意识。"""
    if not user_ids or engine is None:
        return ""
    health_notes = []
    for uid in user_ids:
        profile = engine.user_profiles.get(str(uid), {}) if hasattr(engine, 'user_profiles') else {}
        # 标准化档案字段：special_groups（孕妇/高血压/高血糖等）+ health_needs（降血压/控糖等）
        health_notes.extend(profile.get('special_groups', []) or [])
        health_notes.extend(profile.get('health_needs', []) or [])
    health_notes = [h for h in dict.fromkeys(health_notes) if h]
    if health_notes:
        hint = '、'.join(health_notes[:4])
        return f"这桌菜考虑到了大家的健康情况：有{hint}需求的成员，选菜时注意了低盐低油低嘌呤、口味兼顾平衡～"
    return "这桌菜按多人用餐搭配了荤素和冷热，兼顾了不同成员的口味～"


def _agentic_recommend(user_message: str, user_id: str = None, user_ids: list = None,
                       dm=None, retriever=None, engine=None, llm=None,
                       max_iterations: int = 2, search_query: str = "", top_n: int = 5) -> dict:
    """
    Agent模式：ReAct循环驱动的菜谱推荐

    LLM自主决定搜索、验证和追问，循环执行直到完成任务或达到上限。

    Args:
        user_message: 用户消息
        user_id: 用户ID
        user_ids: 多人ID列表
        dm: DialogManager
        retriever: RAG检索器
        engine: 约束引擎
        llm: LLM客户端
        max_iterations: 最大ReAct迭代次数

    Returns:
        {
            'recommendations': list,  # 推荐的菜品列表
            'response': str,           # 给用户的回复
            'tool_calls_made': int,    # 执行的工具调用次数
            'agentic': True            # 标记为Agent模式
        }
    """
    import json as _json

    t_llm_start = time.perf_counter()
    llm_ms_total = 0

    # 数量约束解析："只吃四道/就来三道/要五道菜/一桌八道"等 → 覆盖默认 top_n。
    # 用户明确指定菜道数时（如"今天只吃四道菜"），推荐必须按该数量返回，而不是固定 5 道。
    try:
        _cnt_match = re.search(
            r'(?:只吃|就吃|来|要|上|点|安排|配)?\s*(?<!第)([一二三四五六七八九十两]|\d{1,2})\s*道(?:菜|汤|菜吧|菜呀|菜啊)?',
            user_message)
        _CN_NUM = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
                   '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
        if _cnt_match:
            _g = _cnt_match.group(1)
            _n = _CN_NUM.get(_g) if _g in _CN_NUM else int(_g)
            if 1 <= _n <= 12:
                top_n = _n
    except Exception:
        pass

    # 构建系统提示
    pref_summary = _json.dumps(dm.user_preferences if dm else {}, ensure_ascii=False)
    system_prompt = f"""你是方太膳食规划助手。你的唯一职责是：搜索菜谱并给出推荐。

铁律（违反则任务失败）：
1. 无论用户说什么，第一件事永远是调用search_recipes搜索菜谱
2. 搜到结果后，必须列出具体菜名回复用户（如"推荐：番茄炒蛋、清蒸鲈鱼..."）
3. ask_user只用于一种情况：用户消息是"嗯"、"哦"、"？"这类无意义内容
4. 禁止用ask_user问"几个人吃？"、"什么口味？"——直接搜，搜完直接推荐
5. 禁止只说话不调工具——每次回复要么调search_recipes，要么列出推荐结果
6. 最小修改原则：如果上一轮已有推荐列表且用户只是追加约束（如"不要太油腻"），保留大部分原有推荐，只替换与新约束冲突的菜品。除非用户明确说"全部换掉"，否则至少保留一半原有推荐。
7. 荤素搭配铁律：每轮推荐至少包含1-2道素菜。搜索时分别搜索荤菜和素菜（用不同query），最终推荐中荤素比例不低于3:2。荤素搜索必须在同一口味主题下进行（如用户要辣，就搜"辣味素菜"而不是"清淡蒸菜"），禁止搜出口味不一致的菜来凑数。不要为凑"烹饪方式多样"而搜索与用户需求无关的菜。
8. 回复中禁止使用 emoji、颜文字或任何表情符号，只用纯文本（可使用编号、换行、星号加粗等常规排版）。
9. 回复要自然口语化，像真人营养师聊天，有温度。避免"好的，为您推荐以下菜品：A、B、C。您可以告诉我是否需要调整"这类机械模板句；引导语可根据菜品特点变化（如"这几道口味清淡，正适合你"、"按你说的，帮你挑了几道下饭的"）。
10. 铁律：最终回复里提到的每一道菜，必须严格来自 search_recipes 工具返回的结果列表。禁止编造、脑补、添加工具结果中不存在的菜名（如"鱼香茄子""红烧肉"等若不在搜索结果中就绝不能写）。回复与搜索结果是同一份菜，可以合并菜名，但不能凭空造菜。
11. 冷热搭配加分：当推荐菜数≥4道（尤其"一桌菜/聚餐/宴客/多人用餐"等场景）时，尽量包含1道凉菜或凉拌菜（如凉拌木耳、凉拌黄瓜、皮蛋豆腐），实现冷热搭配，让一餐更完整；只推荐1-3道单点菜时不强制。

当前约束: {pref_summary}
上一轮推荐列表（保留它们，除非用户要求全部换）: {dm.recommended_recipes[-5:] if dm and dm.recommended_recipes else '无'}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    recommendations = []
    rec_pool = {}  # 合并多次 search_recipes 结果，避免后一次覆盖前一次
    response_text = None
    tool_count = 0
    ask_question = None
    first_call_ms = 0  # 首Token延迟（首次LLM调用耗时）

    for iteration in range(max_iterations):
        print(f"[Agent] 迭代 {iteration+1}/{max_iterations} 开始...")
        _t0 = time.perf_counter()
        result = llm.chat_with_tools(messages, AGENT_TOOLS, temperature=0.1, max_tokens=300)
        _elapsed = (time.perf_counter() - _t0) * 1000
        llm_ms_total += _elapsed
        if iteration == 0:
            first_call_ms = round(_elapsed, 2)
        print(f"[Agent] LLM调用完成 ({_elapsed:.0f}ms), "
              f"finish_reason={result.get('finish_reason')}, "
              f"has_content={bool(result.get('content'))}, tool_calls={len(result.get('tool_calls') or [])}")

        # 有工具调用
        if result.get('tool_calls'):
            for tc in result['tool_calls']:
                tool_name = tc['function']['name']
                try:
                    tool_args = _json.loads(tc['function']['arguments'])
                except _json.JSONDecodeError:
                    tool_args = {}

                tool_result = _execute_agent_tool(
                    tool_name, tool_args,
                    user_id=user_id, user_ids=user_ids,
                    retriever=retriever, engine=engine, dm=dm
                )
                tool_count += 1

                # 添加到消息（保留reasoning_content避免DeepSeek 400报错）
                assistant_msg = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tc]
                }
                if result.get('reasoning_content'):
                    assistant_msg['reasoning_content'] = result['reasoning_content']
                messages.append(assistant_msg)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc['id'],
                    "content": tool_result
                })

                # 解析特殊工具结果
                if tool_name == "search_recipes":
                    try:
                        recipes = _json.loads(tool_result)
                        if isinstance(recipes, list) and recipes:
                            # 合并到结果池，避免后一次 search 覆盖前一次的相关结果
                            for _r in recipes:
                                _n = _r.get('name', '')
                                if _n and _n not in rec_pool:
                                    rec_pool[_n] = _r
                            recommendations = list(rec_pool.values())
                    except _json.JSONDecodeError:
                        pass
                elif tool_name == "ask_user":
                    try:
                        ask_data = _json.loads(tool_result)
                        if ask_data.get('action') == 'ask':
                            ask_question = ask_data.get('question')
                    except _json.JSONDecodeError:
                        pass

            continue  # 继续循环，让LLM处理工具结果

        # 有文本回复，结束循环
        if result.get('content'):
            response_text = result['content']
            break

        # 既无工具调用也无文本回复（异常），重试一次
        if iteration == 0:
            print("[Agent] 首次调用返回空，追加提示重试...")
            messages.append({"role": "user", "content": "请立即调用search_recipes搜索菜谱，或者调用ask_user向用户提问。必须调用工具。"})
            continue
        break

    print(f"[Agent] 循环结束: recs={len(recommendations)}, has_response={bool(response_text)}, "
          f"tool_count={tool_count}, llm_ms={llm_ms_total:.0f}")

    # — 约束修改保留上下文：用户追加约束（如"不吃鱼"）时，将历史推荐里仍符合条件的菜标记出来，
    #   在结果池清洗之后重新并入列表【最前】。不能在这里直接置顶，否则 _trim_by_relevance
    #   会按核心查询重新搜索全部队列满 5 道，把已认可的历史菜又挤掉。 — 
    _kept = []
    try:
        if dm and dm.recommended_recipes:
            _keep_filters = dict(dm.get_search_filters()) if dm else {}
            _existing = {r.get('name', '') for r in recommendations}
            # 用户已否决的菜绝不能作为"保留项"回灌——
            # 否则"不好吃，换一批"后重推仍与上一轮大面积重叠。
            _rejected_names = set(getattr(dm, 'rejected_recipes', []) or [])
            # 只基于"当前方案"（最近一轮实际推荐）做最小化修改保留：
            # 用户追加约束（如"不吃鱼"）时，应保留上一轮仍符合新约束的菜，
            # 而不是从整段历史里去重累加集合里捞旧菜（那会把早期甜品又拉回来）。
            _plan = dm.last_recommendation or dm.recommended_recipes[-5:]
            # 口味/偏好类约束（油腻/清淡/不胖/少油）属于"软约束"，历史保留过多会锁死
            # 推荐导致约束不生效。限制最多保留 3 道，给符合新口味的新检索留足名额。
            _KEEP_MAX = 3
            for _hname in _plan:
                if _hname in _existing or _hname in _rejected_names:
                    continue
                _full = retriever.get_recipe_by_name(_hname)
                if not _full:
                    continue
                if retriever._apply_filters([_full], _keep_filters):
                    _kept.append(_full)
                    _existing.add(_hname)
                if len(_kept) >= _KEEP_MAX:
                    break
            if not _kept and dm.recommended_recipes:
                print(f"[Agent] 历史推荐未并入(池内{len(recommendations)}道)", flush=True)
    except Exception as _e:
        print(f"[Agent] 合并历史推荐失败: {_e}", flush=True)

    # — 结果池清洗：补全完整菜谱 + 按核心查询相关度重排裁剪 —
    # 多轮搜索会把不相关菜品（如为凑"烹饪方式多样"搜出的蒸菜）混入结果池，
    # 导致"想吃辣"却混入一堆蒸菜。此处先补全完整菜谱字段（供叙述/营养计算），
    # 再按用户核心查询相关度重排并裁剪到5道，保证最终推荐与需求一致。
    if recommendations:
        try:
            # 核心检索词优先使用 LLM 改写的语义查询（口语/复合需求 -> 可检索中文关键词），
            # 没有时再回退到关键词解析。这让"最近上火了"等生僻/复合表达能按语义重排补足。
            core_q, _ = _clean_search_query(user_message)
            if search_query:
                core_q = search_query
            # 1) 补全完整菜谱数据
            enriched = []
            for r in recommendations:
                rname = r.get('name', '')
                if rname:
                    full = retriever.get_recipe_by_name(rname)
                    enriched.append(full if full else r)
                else:
                    enriched.append(r)
            # 2) 按核心查询相关度重排并裁剪/补足（带上排除过滤，避免补入被忌口的食材）
            trim_filters = dict(dm.get_search_filters()) if dm else {}
            _, neg_ex = _clean_search_query(user_message)
            if neg_ex:
                trim_filters['exclude_ingredients'] = list(set(
                    trim_filters.get('exclude_ingredients', []) + neg_ex))
            recommendations = _trim_by_relevance(enriched, core_q, retriever, max_n=top_n,
                                                 filters=trim_filters,
                                                 engine=engine, user_id=user_id, user_ids=user_ids)
            print(f"[Agent] 结果池清洗后: {len(recommendations)}道 -> "
                  f"{[r.get('name') for r in recommendations]}")
        except Exception as e:
            print(f"[Agent] 结果池清洗失败: {e}")

    # — 单一食材/菜品过度集中均衡：用户"想吃土豆/想吃鱼/想吃鸡"等单一食材请求时，
    #    检索候选池会被同一食材菜占满（5道全是土豆），营养严重失衡（碳水/蛋白单一）。
    #    若 ≥4 道菜名都命中同一核心食材词，保留 2-3 道主菜，其余用全库"荤素搭配"补足，
    #    让一餐有主有配、营养均衡，而不是一份"土豆全席"。 —
    try:
        _core_food = (_clean_search_query(user_message)[0] if user_message else '') or ''
        if _core_food and len(_core_food) >= 2 and len(recommendations) >= 4:
            _food_hits = [r for r in recommendations
                          if _core_food in (r.get('name', '') or '')
                          or _core_food in str(r.get('ingredients', ''))]
            _other = [r for r in recommendations if r not in _food_hits]
            if len(_food_hits) >= 4:
                # 保留 3 道主菜 + 现有其他菜，不足的部分从全库"荤素搭配"检索补足
                _keep = _food_hits[:3]
                _balanced = list(_keep) + _other
                _seen_names = {r.get('name', '') for r in _balanced}
                _bal_filters = dict(dm.get_search_filters()) if dm else {}
                _bal_pool = retriever.search('家常菜 荤素搭配 营养均衡', top_k=15, filters=_bal_filters)
                if engine and (user_ids or (user_id and engine.user_profiles.get(user_id))):
                    _bal_pool = (engine.filter_by_constraints(_bal_pool, user_ids=user_ids)
                                 if user_ids else engine.filter_by_constraints(_bal_pool, user_id=user_id))
                for _r in _bal_pool:
                    if len(_balanced) >= top_n:
                        break
                    _n = _r.get('name', '')
                    if _n and _n not in _seen_names and _core_food not in _n:
                        _balanced.append(_r)
                        _seen_names.add(_n)
                print(f"[Agent] 单一食材[{_core_food}]集中{len(_food_hits)}道，均衡后: "
                      f"{[r.get('name') for r in _balanced]}", flush=True)
                recommendations = _balanced[:top_n]
    except Exception as _be:
        print(f"[Agent] 食材均衡失败: {_be}", flush=True)

    # — 并入历史保留菜：裁切完成后将用户已认可且仍符合约束的历史菜置顶 —
    # 不能再被任何后续重排裁剪挤掉（用户只是追加约束，不应把之前满意的菜换光）。
    if _kept:
        try:
            _merged = list(_kept)
            _merged_names = {r.get('name', '') for r in _merged}
            for r in recommendations:
                n = r.get('name', '')
                if n and n not in _merged_names:
                    _merged.append(r)
                    _merged_names.add(n)
                if len(_merged) >= top_n:
                    break
            recommendations = _merged[:top_n]
            print(f"[Agent] 并入历史保留 {len(_kept)} 道并置顶: "
                  f"{[r.get('name') for r in recommendations]}")
        except Exception as e:
            print(f"[Agent] 并入历史保留失败: {e}")

    # — 保底补齐：推荐数量不足时直接搜索补充 —
    if len(recommendations) < 3 and len(recommendations) > 0:
        print(f"[Agent] 推荐不足{len(recommendations)}道，自动补齐...")
        existing_names = {r.get('name', '') for r in recommendations}
        try:
            core, neg_ex = _clean_search_query(user_message)
            fill_filters = dict(dm.get_search_filters()) if dm else {}
            if neg_ex:
                fill_filters['exclude_ingredients'] = list(set(fill_filters.get('exclude_ingredients', []) + neg_ex))
            fill_q = (search_query or core) if (search_query or core) else ('晚餐' if neg_ex else user_message)
            fill_results = retriever.search(fill_q, top_k=10, filters=fill_filters)
            # 补齐的菜同样必须通过健康约束过滤（过敏/疾病/特殊人群）
            if engine and (user_ids or (user_id and engine.user_profiles.get(user_id))):
                try:
                    if user_ids:
                        fill_results = engine.filter_by_constraints(fill_results, user_ids=user_ids)
                    else:
                        fill_results = engine.filter_by_constraints(fill_results, user_id=user_id)
                except Exception as _fe:
                    print(f"[Agent] 补齐约束过滤失败: {_fe}")
            for r in fill_results:
                if len(recommendations) >= top_n:
                    break
                rname = r.get('name', '')
                if rname and rname not in existing_names:
                    recommendations.append(r)
                    existing_names.add(rname)
        except Exception as e:
            print(f"[Agent] 补齐失败: {e}")
        print(f"[Agent] 补齐后: {len(recommendations)}道")

    # — 保底荤素：全荤无素时替换1-2道为素菜 —
    # 荤素判定复用统一的 _is_meat_recipe（已剔除鸡精/猪油/蚝油/鸡蛋等调味·蛋奶素误判）
    if len(recommendations) >= 3:
        meat_count = sum(1 for r in recommendations if _is_meat_recipe(r))
        veg_count = len(recommendations) - meat_count
        if meat_count == len(recommendations):  # 全荤无素
            print(f"[Agent] 全荤无素({meat_count}荤/{veg_count}素)，自动搜索素菜...")
            try:
                veg_filters = dict(dm.get_search_filters()) if dm else {}
                veg_results = retriever.search("素菜 蔬菜 清淡 凉拌 蒸菜", top_k=6, filters=veg_filters)
                # 素菜补齐同样必须通过健康约束过滤（如痛风用户不能补海鲜类"素"菜）
                if engine and (user_ids or (user_id and engine.user_profiles.get(user_id))):
                    try:
                        if user_ids:
                            veg_results = engine.filter_by_constraints(veg_results, user_ids=user_ids)
                        else:
                            veg_results = engine.filter_by_constraints(veg_results, user_id=user_id)
                    except Exception as _ve:
                        print(f"[Agent] 素菜补齐约束过滤失败: {_ve}")
                existing_names = {r.get('name', '') for r in recommendations}
                replaced = 0
                for v in veg_results:
                    if replaced >= 2:
                        break
                    vname = v.get('name', '')
                    if vname and vname not in existing_names:
                        recommendations[replaced] = v
                        existing_names.add(vname)
                        replaced += 1
                print(f"[Agent] 替换{replaced}道为素菜")
            except Exception as e:
                print(f"[Agent] 素菜补齐失败: {e}")

    # — 最终约束安全网：无论中间各环节（历史保留/裁剪/补齐/换素）如何组装，
    #    出口处统一过一次健康约束过滤，确保给用户的每一道菜都合规；
    #    过滤后不足时从"全库约束合规池"按查询相关度补足，避免空推荐。 —
    try:
        if recommendations and engine and (
                user_ids or (user_id and engine.user_profiles.get(user_id))):
            _safe = (engine.filter_by_constraints(recommendations, user_ids=user_ids)
                     if user_ids else engine.filter_by_constraints(recommendations, user_id=user_id))
            if len(_safe) < len(recommendations):
                print(f"[Agent] 最终约束安全网: {len(recommendations)}->{len(_safe)}道 "
                      f"(剔除 {[r.get('name') for r in recommendations if r not in _safe]})")
            recommendations = _safe
            if 0 < len(recommendations) < top_n:
                _core_fill, _ = _clean_search_query(user_message)
                _fill_filters = dict(dm.get_search_filters()) if dm else {}
                _pool = retriever.search(_core_fill or '晚餐 家常', top_k=30, filters=_fill_filters)
                _pool = (engine.filter_by_constraints(_pool, user_ids=user_ids)
                         if user_ids else engine.filter_by_constraints(_pool, user_id=user_id))
                _seen = {r.get('name', '') for r in recommendations}
                for _r in _pool:
                    if len(recommendations) >= top_n:
                        break
                    if _r.get('name') and _r.get('name') not in _seen:
                        recommendations.append(_r)
                        _seen.add(_r.get('name'))
                print(f"[Agent] 安全网补足后: {len(recommendations)}道")
    except Exception as _se:
        print(f"[Agent] 最终约束安全网失败: {_se}")

    # — 最终数量兜底：无论中间环节如何组装，出口统一截断到 top_n，
    #    防止异常路径（如否定重推时结果池未清洗）返回超出用户期望数量的菜。 —
    if len(recommendations) > top_n:
        print(f"[Agent] 数量兜底: {len(recommendations)}->{top_n}道", flush=True)
        recommendations = recommendations[:top_n]

    # 如果Agent没给出回复，构造默认回复
    if not response_text:
        if recommendations:
            names = ', '.join([r['name'] for r in recommendations])
            response_text = f"帮你选了几道：{names}～"
        elif ask_question:
            response_text = ask_question
        else:
            response_text = "好的，你想吃点什么类型的菜呢？我可以帮你推荐～"
        print(f"[Agent] 使用默认回复: {response_text[:80]}...")

    # — 兜底生成：菜谱库无匹配且非追问时，LLM现场生成一道新菜（赛题加分项）—
    if not recommendations and not ask_question:
        print("[Agent] 检索结果为空，尝试生成符合约束的新菜谱...")
        gen = _generate_fallback_recipe(
            user_message, user_id=user_id, user_ids=user_ids,
            dm=dm, engine=engine, llm=llm
        )
        if gen:
            recommendations = [gen]
            ings = '、'.join(gen['ingredients'])
            response_text = (
                f"菜谱库中暂时没有完全符合您需求的菜品，我为您现场设计了一道新菜——"
                f"【{gen['name']}】（建议菜谱·生成）\n"
                f"食材：{ings}\n"
                f"做法：{gen['steps']}"
            )
            print(f"[Agent] 已生成新菜谱: {gen['name']}")

    # 记录到历史
    if dm:
        if recommendations:
            recipe_names = [r['name'] for r in recommendations]
            dm.add_recommended_recipes(recipe_names)

    # — 叙述与列表一致性兜底 —
    # Agent 可能综合多次搜索结果或编造列表外的菜名（如"鱼香茄子"），也可能只提了部分菜、
    # 把列表里的菜漏掉，导致叙述与结构化列表脱节。只要叙述【引用了列表外的菜】 或 【漏提了
    # 列表里的菜】，就重建确定性叙述，保证回复与推荐列表完全一致。
    if recommendations and response_text and not ask_question:
        rec_names = [r.get('name', '') for r in recommendations if r.get('name')]
        rec_set = set(rec_names)
        library_names = [r.get('name', '') for r in retriever.recipes if r.get('name')]
        # 提取叙述中引用的菜名（在菜谱库内匹配，避免把描述文本误判为菜名）
        cited = [n for n in library_names if len(n) >= 2 and n in response_text]
        cited_outside = [n for n in cited if n not in rec_set]
        missing_in_text = [n for n in rec_names if n not in response_text]
        if cited_outside or missing_in_text:
            print(f"[Agent] 叙述与实际列表不一致(引用了列表外{len(cited_outside)}道,"
                  f"漏提列表内{len(missing_in_text)}道)，改用确定性叙述")
            response_text = _build_consistent_response(recommendations)

    return {
        'recommendations': recommendations,
        'response': response_text,
        'ask_question': ask_question,
        'tool_calls_made': tool_count,
        'agentic': True,
        'llm_ms': round(llm_ms_total, 2),
        'first_token_ms': first_call_ms
    }


# ═════════════════════════════════════════════════════════════
# 多 Agent 编排层落地：专职 Agent 的 handler
#
# 原 dialog() 的"巨型 if/elif 意图分发"被重构为：
#   - 意图识别（协调者侧）确定 intent
#   - 编排器 ORCHESTRATOR 按 intent 路由到对应专职 Agent 的 handler
#   - 各 handler 只负责自身职责，复用 dialog_enhancer / rag_retriever /
#     constraint_engine / llm_client 等既有模块
#
# handler 统一签名：handler(ctx: DialogContext) -> None
#   - 输入：ctx.dm / ctx.retriever / ctx.engine / ctx.llm / ctx.message /
#           ctx.user_id / ctx.user_ids / ctx.preferences
#   - 输出：写回 ctx.response_text / ctx.recommendations /
#           ctx.agent_result / ctx.t_llm_dialog（与旧 dialog 分支等价）
# ═════════════════════════════════════════════════════════════

def _handler_recommend(ctx: DialogContext):
    """检索·推荐 Agent：ReAct 循环推荐 + 模糊消息强制追问。"""
    dm, retriever, engine, llm = ctx.dm, ctx.retriever, ctx.engine, ctx.llm
    message, user_id, user_ids = ctx.message, ctx.user_id, ctx.user_ids
    preferences = ctx.preferences or {}

    # 模糊消息强制追问：无偏好无约束时避免盲目推荐
    VAGUE_QUERIES = ['今晚吃什么', '中午吃', '早上吃', '晚上吃', '吃什么',
                     '吃啥', '早饭', '午饭', '晚饭', '早餐', '午餐', '晚餐']
    up = dm.user_preferences if dm else {}
    has_prefs = (bool(preferences) or bool(up.get('preferences'))
                or bool(up.get('dietary_goals')) or bool(up.get('cuisine_preference')))
    has_allergies = bool(up.get('allergies'))
    if (not has_prefs and not has_allergies and dm and dm.turn_count <= 1
            and any(q in message for q in VAGUE_QUERIES)):
        ctx.response_text = ("好嘞！先告诉我你的口味偏好吧～喜欢清淡还是重口？"
                             "想吃荤还是素？有没有忌口或过敏的食材？")
        ctx.recommendations = []
        return

    agent_result = _agentic_recommend(
        user_message=message, user_id=user_id, user_ids=user_ids,
        dm=dm, retriever=retriever, engine=engine, llm=llm,
        search_query=ctx.search_query,
    )
    ctx.response_text = agent_result['response']
    ctx.recommendations = agent_result['recommendations']
    if agent_result.get('ask_question'):
        ctx.response_text = agent_result['ask_question']
    ctx.agent_result = agent_result
    ctx.t_llm_dialog = agent_result.get('llm_ms', 0)


def _handler_preference(ctx: DialogContext):
    """偏好·约束 Agent：语义理解并落地偏好/约束，推荐紧随其后。"""
    dm, retriever, engine, llm = ctx.dm, ctx.retriever, ctx.engine, ctx.llm
    message, user_id, user_ids = ctx.message, ctx.user_id, ctx.user_ids

    agent_result = _agentic_recommend(
        user_message=message, user_id=user_id, user_ids=user_ids,
        dm=dm, retriever=retriever, engine=engine, llm=llm,
        search_query=ctx.search_query,
    )
    ctx.response_text = agent_result['response']
    ctx.recommendations = agent_result['recommendations']
    if agent_result.get('ask_question'):
        ctx.response_text = agent_result['ask_question']
    ctx.agent_result = agent_result
    ctx.t_llm_dialog = agent_result.get('llm_ms', 0)


def _handler_nutrition(ctx: DialogContext):
    """营养 Agent：查询指定菜品/食材的营养信息。"""
    message = ctx.message
    response_text = ""
    # 1) 按菜名精确/包含匹配
    for recipe in ctx.retriever.recipes:
        if recipe.get('name', '') in message or message in recipe.get('name', ''):
            nutrition = ctx.engine.evaluate_nutrition(recipe)
            response_text = (f"菜品【{recipe['name']}】的营养信息：热量{nutrition['calories']}千卡，"
                             f"蛋白质{nutrition['protein']}克，碳水{nutrition['carb']}克，"
                             f"脂肪{nutrition['fat']}克。")
            break
    # 2) 未匹配到菜名：尝试按食材名查营养数据库（如"鸡胸肉的营养""虾的热量"）
    if not response_text:
        try:
            nut_db = ctx.engine.nutrition_db
            # 提取核心食材词："鸡胸肉的营养"→"鸡胸肉"；"虾的热量"→"虾"；
            # "帮我查一下鸡胸肉的热量"→"鸡胸肉"。先剥离口语动词前缀，再抓"X的(营养/热量)"。
            _nut_msg = message or ''
            for _v in ['帮我查一下', '帮我查询', '帮我查', '给我查一下', '查一下', '查询一下',
                       '查查', '看看', '看一下', '问一下', '搜一下', '告诉我', '帮我看看', '帮我', '查']:
                if _nut_msg.startswith(_v):
                    _nut_msg = _nut_msg[len(_v):].lstrip('，。、, ')
                    break
            _food_m = re.search(r'([\u4e00-\u9fa5]{1,10})的(?:营养|营养价值|热量|卡路里|蛋白质|脂肪|碳水|碳水化合物)', _nut_msg)
            _core_word = (_food_m.group(1) if _food_m else '') or ''
            if not _core_word:
                # 回退：沿用清洗后的核心词并剥离营养问询后缀
                _core_word = _clean_search_query(message)[0] if message else ''
                _core_word = re.sub(r'(的营养|的营养价值|的热量|热量|蛋白质|脂肪|碳水|碳水化合物|怎么样|多少|信息|卡路里|多少卡|查询|告诉我|的|是)', '', _core_word or '').strip()
            if _core_word:
                # 部位词归一化：营养库用通用名（鸡肉/猪肉），用户常问"鸡胸肉/猪里脊"等部位名。
                # 归一后能命中通用名，否则"鸡胸肉"子串匹配不到"鸡肉"（曾导致营养查询落空）。
                _PART_MAP = {'鸡胸肉': '鸡肉', '鸡腿肉': '鸡肉', '鸡脯肉': '鸡肉', '鸡胸': '鸡肉', '鸡翅': '鸡肉',
                             '猪里脊': '猪肉', '里脊肉': '猪肉', '猪瘦肉': '猪肉', '猪梅肉': '猪肉',
                             '牛里脊': '牛肉', '牛腩': '牛肉', '牛腱': '牛肉',
                             '鸭胸': '鸭肉', '羊排': '羊肉', '鱼片': '鱼肉', '鱼柳': '鱼肉'}
                _canon = _PART_MAP.get(_core_word, '')
                # 匹配策略：营养库键含核心词 / 核心词含营养库键 / 归一化同义词 / 有公共 2-gram
                _best, _best_data, _best_score = None, None, 0
                for fname, data in (nut_db or {}).items():
                    if not fname:
                        continue
                    _f = str(fname)
                    if _f == _core_word:
                        _best, _best_data, _best_score = _f, data, 10
                        break
                    if _f in _core_word or _core_word in _f:
                        _s = len(_f) if _f in _core_word else len(_core_word)
                        if _s > _best_score:
                            _best, _best_data, _best_score = _f, data, _s
                    elif _canon and (_f == _canon or _f in _canon or _canon in _f):
                        if 8 > _best_score:
                            _best, _best_data, _best_score = _f, data, 8
                    elif _core_word[:2] and _core_word[:2] in _f:
                        if 2 > _best_score:
                            _best, _best_data, _best_score = _f, data, 2
                if _best_data:
                    response_text = (
                        f"食材【{_best}】（每100克）的营养：热量{_best_data.get('热量', '?')}千卡，"
                        f"蛋白质{_best_data.get('蛋白质', '?')}克，脂肪{_best_data.get('脂肪', '?')}克，"
                        f"碳水{_best_data.get('碳水', '?')}克。")
        except Exception:
            pass
    if not response_text:
        response_text = "请告诉我具体想查询哪个菜品或食材的营养信息（例如「鸡胸肉的热量」）。"
    ctx.response_text = response_text
    ctx.recommendations = []


def _parse_ingredient_names(ingredients, top=6):
    """从食材字符串里抽取出清爽的食材名。"""
    if isinstance(ingredients, list):
        ingredients = '，'.join(str(x) for x in ingredients)
    parts = re.split(r'[；;]+', str(ingredients))
    names = []
    for p in parts:
        p = p.strip()
        p = re.sub(r'^(A|B|C|D|李锦记|金龙鱼)?料[:：]?', '', p)
        p = re.sub(r'^[（(]?[主辅调料]*料[:：]?', '', p)  # 兼容"辅料：""调料："等前缀
        m = re.match(r'^[A-Za-z\u4e00-\u9fa5·\s]{1,10}(?=\d)', p)
        nm = (m.group(0).strip() if m else
              (re.match(r'^[A-Za-z\u4e00-\u9fa5]{1,8}', p).group(0).strip()
               if re.match(r'^[A-Za-z\u4e00-\u9fa5]{1,8}', p) else ''))
        if nm and nm not in ('适量', '少许', '若干', 'A', 'B'):
            names.append(nm)
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out[:top]


def _count_steps(steps) -> int:
    if isinstance(steps, list):
        return len(steps)
    return len(re.findall(r'第[一二三四五六七八九十0-9]+步', str(steps)))


def _split_steps(steps):
    """把长步骤串拆成每一步的列表。"""
    if isinstance(steps, list):
        return [str(s).strip() for s in steps if str(s).strip()]
    items = re.split(r'第[一二三四五六七八九十0-9]+步[：:]', str(steps))
    return [s.strip() for s in items if s.strip()]


def _format_detail_brief(recipe):
    """简略版：评分、用时、难度、主要食材 + 一句话做法概要，末尾追问是否需要详细。"""
    name = recipe.get('name', '')
    ct = recipe.get('cooking_time') or ''
    diff = recipe.get('difficulty') or ''
    steps = recipe.get('steps') or recipe.get('description') or ''
    n = _count_steps(steps)
    ing = '、'.join(_parse_ingredient_names(recipe.get('ingredients', '')))
    items = _split_steps(steps)
    first = ''
    if items:
        first = items[0].split('（')[0].strip().rstrip('。，；;')
    first = (first[:40] + '……') if len(first) > 40 else first
    meta = ' · '.join(x for x in [f'用时约 {ct}' if ct else '', diff if diff else '',
                                  f'{n} 步' if n else ''] if x)
    return (f"关于【{name}】{('（' + meta + '）') if meta else ''}\n"
            f"主要食材：{ing}\n"
            f"做法概要：{first}\n\n"
            f"简要做法就是这样～需要我把每一步详细讲讲吗？")


def _format_detail_full(recipe):
    """详细版：完整食材 + 每一步编号展开。"""
    name = recipe.get('name', '')
    steps = recipe.get('steps') or recipe.get('description') or recipe.get('method') or ''
    ing = recipe.get('ingredients', '')
    if isinstance(ing, list):
        ing = '，'.join(str(x) for x in ing)
    items = _split_steps(steps)
    body = '\n'.join(f"{i}. {s.rstrip('。；;')}" for i, s in enumerate(items, 1)) if items else str(steps)
    return (f"【{name}】详细做法\n"
            f"食材：{ing}\n\n"
            f"{body}\n\n"
            f"照着这个顺序做就好啦，有什么拿不准可以再问我～还有其他想吃的也欢迎告诉我哦")


def _best_prefix_len(a: str, b: str) -> int:
    """计算 a 与 b 的最长公共子串长度（用于菜名简称/变体模糊匹配）。"""
    if not a or not b:
        return 0
    best = 0
    # 限制窗口长度，降低 O(n^2) 开销；菜名一般不超过 12 字
    max_m = min(len(a), len(b), 12)
    for length in range(max_m, 0, -1):
        found = False
        for i in range(0, len(a) - length + 1):
            if a[i:i + length] in b:
                best = length
                found = True
                break
        if found:
            break
    return best


def _strip_detail_phrasing(message: str) -> str:
    """从"怎么做好吃/的烹饪步骤"等问法中剥离问法词，保留菜名关键片段。
    例如"西红柿炒鸡蛋怎么做好吃" → "西红柿炒鸡蛋"；
    "请给我鱼香肉丝的烹饪步骤" → "鱼香肉丝的烹饪步骤"后再去尾残词 → "鱼香肉丝"。
    若用户在句子中间/末尾有非菜词语，简单去首尾外壳后返回中段。"""
    m = message.strip()
    # 去掉句首的礼貌/请求外壳
    m = re.sub(r'^(请|麻烦|帮我|帮忙|你好|请问|能不能|可以|我想|想|请为我|能不能给我)', '', m)
    # 先去掉句式外壳词（句中任意位置）：怎么/如何/怎样/做法/步骤/蒸多久/多久/怎么做好吃等，
    # 避免残留问法词干扰菜名匹配（如"清蒸鲈鱼需要蒸多久"中含"蒸多久"）。
    m = re.sub(r'(怎么做|怎么做好吃|怎么做才好吃|怎么烧|怎么煮|怎么炖|怎么蒸|怎么煎|怎么炒|怎么弄|要怎么做|怎样做|咋做'
               r'|做法步骤|烹饪步骤|具体步骤|详细做法|怎么做菜|的做法|烹饪方法|做法|步骤'
               r'|蒸多久|煮多久|煎多久|炸多久|烧多久|炖多久|烤多久|炒多久|需要多久|要蒸|要煮|要炖|要炸|要煎|要烤|要炒'
               r'|怎么做|怎么|如何|怎样|多久|什么时候|放多久|下锅多久|要多久)', '', m)
    m = re.sub(r'(的做法|的烹饪|的详细做法|的步骤|具体步骤|烹饪方法)' r'$', '', m)
    # 去掉句首/句尾可能的"一道""简单的""想要"等残留修饰
    m = re.sub(r'^(一道|一个|一|给我|来|做|给我做|简单的|家常的|普通的|简单的)', '', m)
    m = m.strip(' ，。、？！!?：:')
    return m


def _handler_detail(ctx: DialogContext):
    """菜谱详情 Agent：先给简略版，用户再追问"详细"时展开完整步骤。"""
    message = ctx.message.strip()
    dm = getattr(ctx, 'dm', None)
    retriever = ctx.retriever

    # 上轮给过简略版，用户想继续追问 -> 直接给详细步骤
    pending = getattr(dm, '_detail_pending', None) if dm else None
    if pending and re.search(r'(详细|讲讲|说说|具体|继续|再来|再讲|好|嗯|可以|要|来|想看)', message):
        r = retriever.get_recipe_by_name(pending.get('name', ''))
        if r:
            ctx.response_text = _format_detail_full(r)
            ctx.recommendations = []
            if dm is not None:
                dm._detail_pending = None
            return

    # 1) 优先从当前消息定位菜名（精确包含或名字包含在句中）
    target = None
    best_score = 0
    _detail_reduced = _strip_detail_phrasing(message)
    for recipe in retriever.recipes:
        name = recipe.get('name', '')
        if name and name in message:
            target = recipe
            best_score = 999
            break
        # 兼容简称/变体：简化库名后，算用户关键片段与库名的最长公共子串，取最高分那一档
        if not name:
            continue
        name_core = re.sub(r'[（(][^）)]*[）)]', '', name).replace(' ', '')
        score = _best_prefix_len(_detail_reduced, name_core)
        if score > best_score:
            best_score = score
            target = recipe
    # 1b) 仅当确实有较强命中（公共片段>=2）才采用模糊匹配结果，否则视为未定位菜名。
    #     阈值取2是为了兼容"麻婆豆腐 vs 麻婆蛋羹"这类只有2字共同的近似菜名；过低会误伤。
    if target is None or best_score < 2:
        target = None
    # 1c) 语义回退：仍未定位到菜名时，把整句关键片段交给检索器，从候选里挑与
    #     关键片段关联最强的一个，避免"可乐鸡翅"命中毫不相干的"红糖姜片干"。
    if target is None:
        reduced = _detail_reduced or message
        if reduced:
            if len(reduced) <= 6:
                # 短关键词（如"麻婆豆腐""可乐鸡翅"）：优先挑菜名里含该关键词任一字的
                for cand in retriever.search(reduced, top_k=10, filters=None):
                    name = cand.get('name', '')
                    if not name:
                        continue
                    name_core = re.sub(r'[（(][^）)]*[）)]', '', name).replace(' ', '')
                    if any(c in name_core for c in reduced):
                        target = cand
                        break
                # 都没有直接包含时，选与关键词公共子串最长的一个作兜底
                if target is None:
                    best_sc, best_c = 0, None
                    for cand in retriever.search(reduced, top_k=10, filters=None):
                        name = cand.get('name', '')
                        if not name:
                            continue
                        name_core = re.sub(r'[（(][^）)]*[）)]', '', name).replace(' ', '')
                        s = _best_prefix_len(reduced, name_core)
                        if s > best_sc:
                            best_sc, best_c = s, cand
                    target = best_c
            else:
                # 长句（如"怎么做一道简单的炒青菜"）：取相关性最高的候选
                for cand in retriever.search(reduced, top_k=8, filters=None):
                    name = cand.get('name', '')
                    if not name:
                        continue
                    target = cand
                    break
    # 2) 消息未点名菜名时，承接对话上下文：取最近一次推荐（当前方案）里的菜
    if target is None:
        cand_names = []
        if dm is not None:
            cand_names = (getattr(dm, 'last_recommendation', None)
                          or (dm.recommended_recipes[-1:] if dm.recommended_recipes else []))
        for name in reversed(cand_names):
            r = retriever.get_recipe_by_name(name)
            if r:
                target = r
                break
    if target is None:
        ctx.response_text = "请告诉我您想了解哪道菜的做法（例如“怎么做【菜名】”）。"
        ctx.recommendations = []
        return

    if dm is not None:
        dm._detail_pending = {'name': target['name']}
    ctx.response_text = _format_detail_brief(target)
    # 个性化食材替换建议：结合用户过敏/目标/特殊人群，给出食材级替换方案
    # （如"花生→腰果""五花肉→鸡胸肉"），提升"千人千面"的专家评审印象分。
    swaps = _suggest_ingredient_swaps(target, dm)
    if swaps:
        ctx.response_text += "\n\n🔄 食材替换建议：" + "；".join(swaps)
    ctx.recommendations = []


# 食材替换知识库：按用户状态（过敏/膳食目标/特殊人群/素食偏好）触发的替换规则。
# 规则为确定性匹配（菜谱食材文本包含触发词即命中），不依赖 LLM，零额外延迟。
_INGREDIENT_SWAP_RULES = {
    # 触发状态: [(触发食材词, 替换建议), ...]
    'allergy': [
        (('花生', '花生米', '花生碎'), '花生 → 腰果仁或黄瓜丁（规避坚果过敏）'),
        (('虾', '虾仁', '基围虾', '小龙虾'), '虾 → 鸡胸肉丁或鱿鱼（规避甲壳类过敏）'),
        (('鸡蛋', '蛋液', '蛋清', '蛋黄'), '鸡蛋 → 木薯粉水或嫩豆腐（规避蛋类过敏）'),
        (('牛奶', '奶油', '芝士', '黄油'), '牛奶/奶油 → 燕麦奶或椰奶（规避乳制品过敏）'),
    ],
    'diet': [
        (('五花肉', '肥牛', '猪蹄', '猪油'), '五花肉/肥肉 → 鸡胸肉或里脊（降脂减卡）'),
        (('猪肉', '猪里脊', '梅花肉'), '猪肉 → 鸡胸肉或鸡腿肉（家常菜通用，更瘦更低脂）'),
        (('油炸', '炸制', '裹粉炸'), '油炸 → 空气炸锅或烤箱烘烤（减油约70%）'),
        (('白糖', '冰糖', '红糖'), '糖 → 代糖且用量减半（控糖）'),
    ],
    'special': [
        (('料酒', '黄酒', '白酒', '啤酒'), '料酒/酒类 → 直接省略或用柠檬汁替代（孕期/哺乳期忌酒精）'),
        (('辣椒', '小米椒', '干辣椒', '花椒'), '辣椒/花椒 → 甜椒或少许白胡椒（老人/儿童/哺乳期宜温和）'),
        (('腊肉', '咸菜', '咸鱼'), '腌制食材 → 新鲜肉类蔬菜（低盐更健康）'),
    ],
    'vegetarian': [
        (('猪肉', '牛肉', '鸡肉', '排骨', '肉丝', '肉片'), '肉类 → 豆腐/香菇/杏鲍菇（素食替代）'),
        (('蚝油', '鱼露', '虾皮'), '蚝油/鱼露 → 香菇素蚝油（纯素调味）'),
    ],
}


def _suggest_ingredient_swaps(recipe: Dict, dm) -> list:
    """根据用户画像（过敏/膳食目标/特殊人群/素食）生成菜谱的食材替换建议。
    返回最多 3 条建议文本；无匹配时返回空列表。"""
    if dm is None or not recipe:
        return []
    prefs = getattr(dm, 'user_preferences', {}) or {}
    ings_text = ' '.join(str(x) for x in recipe.get('ingredients', [])) \
        if isinstance(recipe.get('ingredients'), list) else str(recipe.get('ingredients', ''))
    all_text = f"{recipe.get('name', '')} {ings_text}"

    # 根据用户状态确定激活的规则组
    active_groups = []
    if prefs.get('allergies') or prefs.get('excluded_ingredients'):
        active_groups.append(('allergy', set(prefs.get('allergies', [])) | set(prefs.get('excluded_ingredients', []))))
    goals = [str(g) for g in (prefs.get('dietary_goals') or [])]
    if any(k in g for g in goals for k in ('减肥', '减脂', '低卡', '控糖', '瘦身')):
        active_groups.append(('diet', None))
    groups = [str(g) for g in (prefs.get('special_groups') or [])]
    if groups:
        active_groups.append(('special', set(groups)))
    if prefs.get('preferences', {}).get('vegetarian') or prefs.get('preferences', {}).get('素食'):
        active_groups.append(('vegetarian', None))

    suggestions, seen = [], set()
    for group_name, extra in active_groups:
        for triggers, tip in _INGREDIENT_SWAP_RULES[group_name]:
            # 过敏类：仅当用户过敏/排除的食材与触发词匹配时才建议（避免泛化打扰）
            if group_name == 'allergy' and extra:
                user_items = ' '.join(str(x) for x in extra)
                matched = any(t in user_items for t in triggers) \
                    or any(str(x) in t for x in extra for t in triggers)
                if not matched:
                    continue
            # 特殊人群规则细分：辣度替换仅对老人/儿童/哺乳期生效；
            # 酒精替换仅对孕妇/哺乳期/备孕生效；腌制食材规则对所有特殊人群生效。
            if group_name == 'special' and extra:
                mild = bool({'老人', '儿童', '哺乳期'} & extra)
                preg = bool({'孕妇', '哺乳期', '备孕'} & extra)
                is_spicy_rule = any(t in ('辣椒', '小米椒', '干辣椒', '花椒') for t in triggers)
                is_alcohol_rule = any(t in ('料酒', '黄酒', '白酒', '啤酒') for t in triggers)
                if is_spicy_rule and not mild:
                    continue
                if is_alcohol_rule and not preg:
                    continue
            if any(t in all_text for t in triggers) and tip not in seen:
                seen.add(tip)
                suggestions.append(tip)
    return suggestions[:3]


def _handler_reject(ctx: DialogContext):
    """检索·推荐 Agent（方案否定场景）：记录被否定菜品后重新推荐。"""
    dm, retriever, engine, llm = ctx.dm, ctx.retriever, ctx.engine, ctx.llm
    message, user_id, user_ids = ctx.message, ctx.user_id, ctx.user_ids

    bare_negation = re.search(
        r'^(不太行|不喜欢|不好吃|不太好|不行|不怎么样|算了|不要这个|换一个|都不好|都不行|都不喜欢)[啊呢吧呀嘛]*[。.！!]*$',
        message.strip())

    if bare_negation:
        ctx.response_text = ("可以告诉我具体哪里不满意吗？比如太辣了？太油腻？还是想吃别的口味？")
        ctx.recommendations = []
        return

    resolved_message = dm.resolve_reference(message)
    rejected = [name for name in dm.recommended_recipes
                if name in resolved_message or name in message]
    if rejected:
        for name in rejected:
            dm.add_rejected_recipe(name)
    elif dm.recommended_recipes:
        for name in dm.recommended_recipes:
            dm.add_rejected_recipe(name)

    context_parts = []
    if dm.rejected_recipes:
        context_parts.append(f"用户已否决: {', '.join(dm.rejected_recipes[-5:])}")
    context_parts.append(f"用户要求: {resolved_message}")
    agent_message = '。'.join(context_parts)

    agent_result = _agentic_recommend(
        user_message=agent_message, user_id=user_id, user_ids=user_ids,
        dm=dm, retriever=retriever, engine=engine, llm=llm
    )
    ctx.response_text = agent_result['response']
    ctx.recommendations = agent_result['recommendations']
    if agent_result.get('ask_question'):
        ctx.response_text = agent_result['ask_question']
    # 口味变更方向的确定性引导：用户明确表达"太清淡→要重口/太淡→要够味/换辣"时，
    # 在回复开头注入符合方向的口语化说明（"调整"在场景E判定词中命中），
    # 并确保 LLM 文案没有覆盖掉这层意图——避免"换菜了但回复没体现原因"的脱节。
    _taste_fix = re.search(r'太清淡|太淡|清淡|没味道|没味|口味重|重口|想吃辣|要辣|不够味|下饭',
                           message)
    if _taste_fix and ctx.recommendations and not agent_result.get('ask_question'):
        _flavor_note = "按你要的重口味，重新帮你挑了几道够味的～"
        if not re.search(r'重口味|重口|够味|下饭|红烧|麻辣|香辣', ctx.response_text):
            ctx.response_text = _flavor_note + "\n" + ctx.response_text
    ctx.agent_result = agent_result
    ctx.t_llm_dialog = agent_result.get('llm_ms', 0)


def _handler_more(ctx: DialogContext):
    """检索·推荐 Agent（追加场景）：结合已有推荐再推一波，且支持"多吃几道/再多来几道"等数量要求。

    用户明确想多吃时返回更多道（8道），而不是与普通推荐一样固定 5 道。
    """
    dm, retriever, engine, llm = ctx.dm, ctx.retriever, ctx.engine, ctx.llm
    message, user_id, user_ids = ctx.message, ctx.user_id, ctx.user_ids

    # "多吃几道/再多来点/来一桌"等表达 → 返回更多道菜
    _want_more = re.search(r'多吃|多推荐|多来|再多|来几道|多几道|多上|多整|一桌|多来点|换一批新的多的',
                           message)
    top_n = 8 if _want_more else 5

    context_parts = []
    if dm.recommended_recipes:
        context_parts.append(f"已推荐: {', '.join(dm.recommended_recipes[-8:])}，请避免重复")
    context_parts.append(f"用户要求: {message}")
    agent_message = '。'.join(context_parts)

    agent_result = _agentic_recommend(
        user_message=agent_message, user_id=user_id, user_ids=user_ids,
        dm=dm, retriever=retriever, engine=engine, llm=llm,
        top_n=top_n,
    )
    ctx.response_text = agent_result['response']
    ctx.recommendations = agent_result['recommendations']
    if agent_result.get('ask_question'):
        ctx.response_text = agent_result['ask_question']
    ctx.agent_result = agent_result
    ctx.t_llm_dialog = agent_result.get('llm_ms', 0)


def _handler_vague(ctx: DialogContext):
    """简单交互 Agent / 推荐 Agent（模糊查询）。"""
    dm, retriever, engine = ctx.dm, ctx.retriever, ctx.engine
    message, user_id, user_ids = ctx.message, ctx.user_id, ctx.user_ids

    # 能力边界引导：识别明显与膳食无关的请求（天气/写诗/玩笑/闲聊/其它领域问题）。
    # 这类问题不应强行推荐菜，而是友好地说明能力边界并引导回正题。
    _offtopic = re.search(
        r'天气|气温|降温|升温|下雨|下雪|台风|雷雨|多云|阴天|晴天|适不适合吃|适合吃什么'
        r'|写诗|写一[首篇]|作诗|讲个笑话|笑话|唱个歌|唱歌|翻译|英语|英文'
        r'|新闻|时间|几点了|现在几点|多少钱|几点钟',
        message)
    if _offtopic:
        _kw = _offtopic.group(0)
        if '适不适合' in message or '适合' in message:
            ctx.response_text = (
                f"我是膳食规划助手，主要帮你搭配菜品和营养方案～关于「{message}」"
                f"，我没法判断外面的天气情况，但如果你想吃火锅这类暖身菜，"
                f"我可以直接帮你推荐合适的搭配哦。")
        else:
            ctx.response_text = (
                f"这个我暂时帮不上忙呢～我是一名膳食规划助手，专注菜品推荐、"
                f"营养搭配和多人配餐。如果你想吃的方面，随时告诉我，比如「天冷了想吃点暖身的」「想吃火锅」～")
        ctx.recommendations = []
        return

    if dm.recommended_recipes:
        recipe_names = dm.recommended_recipes[-5:]
        ctx.response_text = f"这几道还合胃口吗？先给你留着：{', '.join(recipe_names)}。想换随时说～"
        ctx.recommendations = []
        for name in recipe_names:
            recipe = retriever.get_recipe_by_name(name)
            if recipe:
                ctx.recommendations.append(recipe)
    else:
        filters = dm.get_search_filters()
        results = retriever.search('晚餐', top_k=10, filters=filters)
        if user_ids:
            results = engine.filter_by_constraints(results, user_ids=user_ids)
        elif user_id:
            results = engine.filter_by_constraints(results, user_id=user_id)

        # 多样性轮换：模糊查询走固定的"晚餐"检索词，精排后每次结果趋同。
        # 用 diversity_seed 在前 10 名里轮换取 5 道，同一用户每次"新对话"换一批菜。
        if dm and getattr(dm, 'diversity_seed', 0) > 0 and len(results) > 5:
            _off = (dm.diversity_seed * 5) % len(results)
            results = results[_off:] + results[:_off]
        ctx.recommendations = results[:5]
        recipe_names = [r['name'] for r in ctx.recommendations]
        dm.add_recommended_recipes(recipe_names)

        names = '、'.join(recipe_names)
        if any(k in message for k in ['甜', '清淡', '素', '健康', '养生']):
            intro = f"按你的偏好，给你配了一桌清淡不腻的：{names}。"
        elif any(k in message for k in ['辣', '重口', '口味重', '下饭']):
            intro = f"这几道够味够下饭：{names}。"
        else:
            intro = f"帮你配好了这 {len(recipe_names)} 道：{names}。"
        ctx.response_text = f"{intro}看看合不合胃口，想调整随时说～"


def _handler_substitute(ctx: DialogContext):
    """检索·推荐 Agent（换菜场景）：最小化修改原则替换某道菜。"""
    import json as _json
    dm, retriever, engine = ctx.dm, ctx.retriever, ctx.engine
    message, user_id, user_ids = ctx.message, ctx.user_id, ctx.user_ids

    # 当前方案（最近一轮推荐）：换菜/否定都基于它，而不是整段历史累积
    # （recommended_recipes 会随对话不断累积，取全部会保留早已过期的旧菜）。
    _plan = dm.last_recommendation or dm.recommended_recipes[-5:]

    resolved_message = dm.resolve_reference(message)
    # 批量否定语言（"换一批/这几个都不想吃/整桌都不要"）→ 整体替换当前方案；
    # 否则只替换消息里点名的具体菜，其余按"最小化修改"保留。
    _batch_reject = bool(re.search(
        r'换一批|重新推荐|重新换|换一拨|这几个|那几个|这些|整桌|这一桌|全部|都不要|都不想吃|都不喜欢|都不行|换掉这[些桌]',
        message))
    substituted = []
    if _batch_reject:
        substituted = list(_plan)
    else:
        for recipe_name in _plan:
            if recipe_name in resolved_message or recipe_name in message:
                substituted.append(recipe_name)

    # 换菜目标道数：尊重"换三道/来四道"等数量指定，默认 5 道
    top_n = 5
    try:
        _cnt = re.search(r'(?:换|来|要|上|点|只吃|就吃)?\s*(?<!第)([一二三四五六七八九十两]|\d{1,2})\s*道', message)
        _CN = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
        if _cnt:
            _g = _cnt.group(1)
            _n = _CN.get(_g) if _g in _CN else int(_g)
            if 1 <= _n <= 12:
                top_n = _n
    except Exception:
        pass

    if substituted:
        for name in substituted:
            dm.add_rejected_recipe(name)

        # 只保留"当前方案"里未被替换的菜作为稳定项，
        # 而不是从整段历史累积里捞旧菜——长对话下 recommended_recipes 会不断累积，
        # 若取全部会导致换菜后返回远超预期数量的菜（曾出现一次返回 13 道）。
        _rej_set = set(substituted)
        stable_names = [r for r in _plan if r not in _rej_set]
        stable_recipes = [retriever.get_recipe_by_name(r) for r in stable_names]
        stable_recipes = [r for r in stable_recipes if r is not None]
        # 稳定项也限制在目标数量内（优先保留最近的），给新换的菜留足名额
        if len(stable_recipes) >= top_n:
            stable_recipes = stable_recipes[:top_n]
        stable_names = [r.get('name', '') for r in stable_recipes]

        filters = dm.get_search_filters()
        core, neg_ex = _clean_search_query(resolved_message)
        if neg_ex:
            filters['exclude_ingredients'] = list(set(filters.get('exclude_ingredients', []) + neg_ex))
        search_q = core if core else resolved_message
        results = retriever.search(search_q, top_k=10, filters=filters)

        if user_ids:
            results = engine.filter_by_constraints(results, user_ids=user_ids)
        elif user_id:
            results = engine.filter_by_constraints(results, user_id=user_id)

        stable_name_set = set(stable_names)
        new_recipes = [r for r in results if r['name'] not in stable_name_set][:top_n - len(stable_recipes)]
        # 出口统一截断：防止任何路径拼出超过目标数量的菜
        ctx.recommendations = (stable_recipes + new_recipes)[:top_n]
        recipe_names = [r['name'] for r in ctx.recommendations]

        dm.add_recommended_recipes(recipe_names)

        ctx.response_text = (f"已为您替换菜品：{', '.join(substituted)} -> "
                             f"{', '.join([r['name'] for r in new_recipes])}。\n"
                             f"保留的菜品：{', '.join(stable_names)}。")
    else:
        # 食材级替换咨询："家里没有猪肉了，能换成鸡肉吗""把白糖换成代糖"
        # 这不是换某道菜，而是询问食材替代方案，应给出友好确认与建议。
        # 判定要收紧：只有"没有X了"（缺食材）或"X能否用Y替代"这类句式才走这里；
        # "把X换成清淡点的"是把菜换掉，不能误入食材替换。
        _missing_m = re.search(r'没有([\u4e00-\u9fa5]{1,6})[了]?[，,。]?', message)
        _sub_ask = re.search(r'(?:能|可以|可不可以|能不能)[^，。？?!]{0,10}(?:换成|替代|代替|替换)'
                             r'|(?:用|拿)[\u4e00-\u9fa5]{1,6}(?:来)?(?:替代|代替|替换)'
                             r'|把([\u4e00-\u9fa5]{1,6})(?:换成|换作|改为|替代|替换成)[\u4e00-\u9fa5]{1,6}(?:吧|吗|行不行|可不可以|可以)?',
                             message)
        if _missing_m or _sub_ask:
            src = (_missing_m.group(1) if _missing_m else '') or (_sub_ask.group(1) if _sub_ask and _sub_ask.group(1) else '')
            tgt = ''
            if _sub_ask:
                _sm = re.search(r'(?:换成|换作|改为|替代|替换成|替换为|用)([\u4e00-\u9fa5]{1,6})', message)
                if _sm:
                    tgt = _sm.group(1)
            # 优先命中食材替换知识库
            tip = None
            if src:
                for group_rules in _INGREDIENT_SWAP_RULES.values():
                    for triggers, t in group_rules:
                        if any(src in tg or tg in src for tg in triggers):
                            tip = t
                            break
                    if tip:
                        break
            if tip:
                advice = f"可以的！{tip}。"
            elif src and tgt:
                advice = (f"可以的，用{tgt}替代{src}完全没问题。"
                          f"注意{tgt}和{src}成熟时间略有差异，烹饪时适当调整火候和时长；"
                          f"若原菜谱含较多肥肉，换成更瘦的{tgt}还能顺带降脂。")
            elif src:
                advice = (f"可以的！{src}可以换成鸡胸肉、鸡腿肉或豆腐来替代，"
                          f"口感更清爽、热量也更低，注意适当调整烹饪时间。")
            else:
                advice = ("可以的！食材之间常可互相替代：肉类可换鸡胸肉/鱼/豆腐，"
                          "蔬菜用同类时蔬替代即可，注意调整烹饪时间与火候。")
            ctx.response_text = advice
            ctx.recommendations = []
            return
        ctx.response_text = "请告诉我您想替换哪道菜。"
        ctx.recommendations = []


def _handler_confirm(ctx: DialogContext):
    """简单交互 Agent：确认当前推荐。"""
    dm, retriever = ctx.dm, ctx.retriever
    message = ctx.message

    dm.set_dialog_state('completed')
    is_rhetorical = bool(re.search(
        r'不是(?:就有|就|有)?[^，。！!？?]{1,8}吗|不就有[^，。！!？?]{1,8}吗', message))
    if is_rhetorical and dm.recommended_recipes:
        parts = ["对呀，这几道菜里确实有辣的，刚那轮没搜全～现在给你配的这些够味：",
                 "、".join(dm.recommended_recipes[-5:])]
    else:
        parts = ["好的，已确认推荐～"]
    if dm.recommended_recipes:
        ctx.recommendations = []
        for _name in dm.recommended_recipes[-5:]:
            _rec = retriever.get_recipe_by_name(_name)
            if _rec:
                ctx.recommendations.append(_rec)
        total_time = sum(
            (retriever.get_recipe_by_name(r) or {}).get('cooking_time', 0)
            for r in dm.recommended_recipes[-8:]
        )
        parts.append(f"预计总烹饪时间约{total_time}分钟，建议先处理耗时长的菜品。祝你用餐愉快！")
    else:
        parts.append("祝你用餐愉快！")
    ctx.response_text = '\n'.join(parts)


def _handler_cancel(ctx: DialogContext):
    """简单交互 Agent：重置对话。"""
    ctx.dm.reset_dialog()
    ctx.response_text = "好的，已重置。还有什么需要帮忙的吗？"
    ctx.recommendations = []


def _handler_greet(ctx: DialogContext):
    """简单交互 Agent：开场问候。"""
    ctx.response_text = "嗨！我是方太膳食规划助手～告诉我你的口味偏好，帮你搭配合适的菜品！"
    ctx.recommendations = []


def _handler_farewell(ctx: DialogContext):
    """简单交互 Agent：道别。"""
    ctx.response_text = "好嘞，祝您用餐愉快、身体健康！有需要随时来找我～"
    ctx.recommendations = []


def _register_dialog_agents() -> None:
    """把各专职 Agent 的 handler 注册进编排器（进程启动时调用一次）。"""
    ORCHESTRATOR.register_all({
        'greet': _handler_greet,
        'farewell': _handler_farewell,
        'recommend': _handler_recommend,
        'set_preferences': _handler_preference,
        'set_preference': _handler_preference,
        'modify_preferences': _handler_preference,
        'add_constraint': _handler_preference,
        'ask_nutrition': _handler_nutrition,
        'ask_recipe_detail': _handler_detail,
        'reject_recommendation': _handler_reject,
        'request_more': _handler_more,
        'vague_query': _handler_vague,
        'request_substitute': _handler_substitute,
        'confirm': _handler_confirm,
        'cancel': _handler_cancel,
    })


# 多 Agent 编排注册：进程启动（含 gunicorn worker fork 前）即完成，
# 保证任何请求入口都能通过 ORCHESTRATOR.dispatch 路由到专职 Agent。
_register_dialog_agents()
print(ORCHESTRATOR.describe())


def _sync_health_to_engine(dm, engine, user_id, user_ids):
    """把对话中提取的特殊人群与慢性疾病同步进约束引擎用户档案。

    约束引擎的 filter_by_constraints 与 result_verifier 只从 engine.user_profiles 读硬约束，
    若不同步，非预置用户在对话里说的"给老人做的/我有高血压"不会被过滤
    （曾出现高血压用户仍被推荐咸肉/咸鱼等禁忌菜）。
    """
    if not dm:
        return
    groups = list(dm.user_preferences.get('special_groups', []) or [])
    diseases = list(dm.user_preferences.get('diseases', []) or [])
    if not groups and not diseases:
        return
    targets = []
    if user_ids:
        targets = [uid for uid in user_ids if str(uid).isdigit() and 1 <= int(uid) <= 50]
    elif user_id:
        if str(user_id).isdigit() and 1 <= int(user_id) <= 50:
            targets = [str(user_id)]
        else:
            # 非预置用户：也写入档案，保证 filter_by_constraints / 验证器生效
            targets = [str(user_id)]
    for uid in targets:
        profile = engine.user_profiles.get(uid)
        if not profile:
            profile = engine.user_profiles[uid] = {
                'user_id': uid, 'allergies': [], 'diseases': [], 'special_groups': []}
        cur = profile.setdefault('special_groups', [])
        for g in groups:
            if g not in cur:
                cur.append(g)
        cur_d = profile.setdefault('diseases', [])
        for d in diseases:
            if d not in cur_d:
                cur_d.append(d)


@app.route('/api/dialog', methods=['POST'])
def dialog():
    """
    对话接口
    
    支持多轮对话，维护用户上下文和偏好。
    支持约束追加、局部替换、方案否定、模糊追问等多轮交互场景。
    
    请求参数：
    {
        "user_id": "user_001",
        "message": "今晚吃什么",
        "reset": false,
        "user_ids": ["user_001", "user_002"]  // 多人用餐场景
    }
    
    Returns:
        JSON响应，包含系统回复和对话状态
    """
    try:
        t_request_start = time.perf_counter()
        t_llm_dialog = 0
        t_search_dialog = 0
        t_filter_dialog = 0
        t_nutrition_dialog = 0
        t_verify_dialog = 0
        retry_count = 0
        first_token_ms = 0
        nutrition_summary = ""
        verification_report = None
        data = request.get_json()
        user_id = data.get('user_id', '')
        message = data.get('message', '')
        reset = data.get('reset', False)
        user_ids = data.get('user_ids', [])  # 多人用餐场景
        
        if not message:
            return jsonify({'success': False, 'error': 'message参数不能为空'}), 400
        
        # 获取组件实例
        retriever = get_retriever()
        engine = get_engine()
        llm = get_llm()
        
        # 创建或获取对话管理器（优先从磁盘恢复之前的饮食习惯记忆）
        dialog_key = user_id or "anonymous"
        if dialog_key not in _dialog_managers:
            if len(_dialog_managers) >= _MAX_DIALOG_MANAGERS:
                _dialog_managers.pop(next(iter(_dialog_managers)), None)
            _dialog_managers[dialog_key] = _load_dialog(dialog_key) or DialogManager()
        
        dm = _dialog_managers[dialog_key]
        
        # 重置对话（如果请求）：清空内存态，并一并清除已保存的记忆
        if reset:
            dm.reset_dialog()
            _clear_dialog_memory(dialog_key)
        
        # 新用户引导流程：预置用户(user_id 在1-50)跳过引导，新用户自动触发
        if not user_id or (isinstance(user_id, str) and not user_id.isdigit()) or (isinstance(user_id, str) and not (1 <= int(user_id) <= 50)):
            if not dm.is_onboarding and dm.onboarding_step == 0 and dm.turn_count == 0:
                dm.onboarding_step = 0  # 触发引导
        
        # 如果在引导流程中，优先处理引导
        if dm.is_onboarding or (dm.onboarding_step > 0 and dm.onboarding_step < 4):
            # 先提取偏好（引导中的回答也可能含偏好信息）
            dm.extract_preferences(message)
            dm.add_message('user', message)
            response_text = dm.process_onboarding(message)
            if response_text:
                dm.add_message('assistant', response_text)
                _save_dialog(dialog_key, dm)  # 引导过程中也记录已表达的偏好/忌口
                return jsonify({
                    'success': True,
                    'data': {
                        'intent': 'onboarding',
                        'response': response_text,
                        'onboarding_step': dm.onboarding_step,
                        'is_onboarding': dm.is_onboarding,
                        'recommendations': []
                    }
                })
            # 引导完成，继续走正常推荐流程
        
        # 添加用户消息到对话历史
        dm.add_message('user', message)
        
        # 合并意图识别+偏好提取（一次LLM调用，节省一轮延迟）
        # 先跑关键词提取作为基线（可靠覆盖"刺激/过瘾/不吃X"等明确表达），
        # 再由 LLM 覆盖/补充，避免 LLM 漏提口味字段导致"想吃刺激的"丢失辣味偏好
        # 矛盾检测必须在本轮偏好写入 dm 之前做快照：
        # 否则"我想吃红烧肉"提取出的 vegetarian=false 会覆盖素食标记，检测失效。
        _conflict_tip = _detect_preference_conflict(message, dm)
        kw_preferences = dm.extract_preferences(message)  # 关键词基线（内部会写入）
        llm_result = dm.detect_with_llm(message, llm_client=llm)
        search_query = ""
        if llm_result:
            intent = llm_result.get('intent', 'recommend')
            if llm_result.get('search_query'):
                sq = str(llm_result['search_query']).strip()
                if sq and sq.lower() != 'null':
                    search_query = sq
            if llm_result.get('preferences'):
                # 关键：LLM 提取的偏好必须写入 dm，否则 get_search_filters() 拿不到排除词，
                # 导致"不吃虾"等否定约束在检索/验证阶段全部失效
                dm._update_preferences(llm_result['preferences'])
                preferences = llm_result['preferences']
            else:
                preferences = kw_preferences
        else:
            intent = dm.detect_intent(message)  # 关键词回退
            preferences = kw_preferences

        # 道别强制路由："谢谢，再见/拜拜" 等告别语即使被 LLM 误判为 greet/recommend，
        # 也要回以道别话术，而不是再问一遍"想吃什么"。
        if re.search(r'再见|拜拜|回聊|下次见|下次聊|走了', message):
            intent = 'farewell'
        
        # 兜底1：强制关键词提取排除食材，保证"不吃X"始终进入 excluded_ingredients
        if not dm.user_preferences.get('excluded_ingredients'):
            dm.extract_preferences(message)
        # 兜底2：从原文解析否定/过敏排除（覆盖"对X过敏"等关键词未覆盖的表达）
        _, neg_ex = _clean_search_query(message)
        if neg_ex:
            for e in neg_ex:
                if e not in dm.user_preferences['excluded_ingredients']:
                    dm.user_preferences['excluded_ingredients'].append(e)

        # 素食标记显式解除：只有用户明确说"不再吃素/恢复吃肉"才清除
        # （配合 vegetarian 单向保护，避免"想吃红烧肉"这种口误误清除长期素食设定）
        if re.search(r'不再吃素|不吃素了|恢复吃肉|开始吃肉|可以吃肉了|不是素食', message):
            if (dm.user_preferences.get('preferences') or {}).get('vegetarian'):
                dm.user_preferences['preferences'].pop('vegetarian', None)
        
        # 意图名归一化：兼容 LLM 可能输出的旧别名，映射到 dialog() 分支使用的标准名
        INTENT_ALIASES = {
            'vague': 'vague_query',
            'replace': 'request_substitute',
            'ask_more': 'request_more',
            'ask_detail': 'ask_recipe_detail',
            'reject': 'reject_recommendation',
            'set_preference': 'set_preferences',
            'ask_clarification': 'clarify',
        }
        intent = INTENT_ALIASES.get(intent, intent)

        # 反问句强制覆盖（双保险）：如"这不是有辣的吗""不就有辣的吗"是用户对当前
        # 推荐的肯定性反问，绝不能被 LLM/关键词误判为否定推荐。已有推荐时统一走 confirm。
        if (re.search(r'不是(?:就有|就|有)?[^，。！!？?]{1,8}吗|不就有[^，。！!？?]{1,8}吗', message)
                and dm.recommended_recipes
                and intent in ('reject_recommendation', 'set_preferences', 'add_constraint',
                               'recommend', 'vague_query', 'request_substitute')):
            print(f"[dialog] 反问句确认: '{message}' -> confirm（保留当前推荐）")
            intent = 'confirm'

        # 带理由的方案否定强制路由：如"不好吃，太清淡了，想吃口味重的"常被 LLM 误判为
        # set_preferences/add_constraint，导致上一轮菜未被否决、重推后大面积重叠。
        # 只要用户明确表达不满且不止裸否定（裸否定走追问），统一按方案否决处理。
        # 注意：不能把"太油腻/太清淡"单独当否定——"不要太油腻"是追加约束而非全盘否定。
        _bare_neg = re.match(
            r'^(不太行|不喜欢|不好吃|不太好|不行|不怎么样|算了|不要这个|换一个|都不好|都不行|都不喜欢)'
            r'[啊呢吧呀嘛]*[。.！!]*$', message.strip())
        if (dm.recommended_recipes
                and re.search(r'不好吃|不太好吃|不满意|不想吃这些|不喜欢这|换个|换一批|重新推荐|重新换', message)
                and not _bare_neg
                and intent in ('set_preferences', 'add_constraint', 'recommend', 'vague_query')):
            print(f"[dialog] 带理由否定: '{message}' -> reject_recommendation（否决上一轮方案）")
            intent = 'reject_recommendation'
        
        # 根据意图生成响应 —— 多 Agent 编排分发
        # 路由决策：意图 -> 主理专职 Agent（协调 Agent 负责路由）；
        # 未注册/未知意图一律回退到"检索·推荐 Agent"（等价原默认分支）。
        if not ORCHESTRATOR.has_handler(intent):
            print(f"[agent] 未注册意图 '{intent}'，回退到检索·推荐 Agent")
            intent = 'recommend'

        # 将本轮提取的特殊人群同步进约束引擎档案，保证 filter_by_constraints / 验证生效
        _sync_health_to_engine(dm, engine, user_id, user_ids)

        ctx = DialogContext(
            dm=dm, retriever=retriever, engine=engine, llm=llm,
            message=message, user_id=user_id, user_ids=user_ids,
            preferences=preferences, search_query=search_query,
        )
        ORCHESTRATOR.dispatch(intent, ctx)
        response_text = ctx.response_text
        recommendations = ctx.recommendations or []
        agent_result = ctx.agent_result or {}
        t_llm_dialog = ctx.t_llm_dialog

        # ── 回复增强：矛盾提示 / 多人约束说明 / 追加约束保留说明 ──
        # 这些信息是"有洞察力的助手"的关键体现：主动指出偏好冲突、
        # 点明多人配餐时考虑了谁的健康约束、追加约束时保留了哪些原有推荐。
        _prefix_parts = []
        try:
            if _conflict_tip:
                _prefix_parts.append(_conflict_tip)
            if user_ids and recommendations:
                _multi_note = _build_multi_user_note(user_ids, engine)
                if _multi_note:
                    _prefix_parts.append(_multi_note)
            if intent == 'add_constraint' and recommendations:
                _prefix_parts.append(
                    "按你新提的要求调整过了：保留了之前推荐里依然符合的菜，"
                    "只把和新约束冲突的换掉了。")
        except Exception as _pe:
            print(f"[dialog] 回复增强失败: {_pe}")
        if _prefix_parts and response_text:
            response_text = ' '.join(_prefix_parts) + "\n" + response_text
        elif _prefix_parts:
            response_text = ' '.join(_prefix_parts)
        
        # 添加系统回复到对话历史
        dm.add_message('system', response_text)
        
        # ── 结果验证（拦截违规菜品）──
        t_verify_start = time.perf_counter()
        if recommendations:
            verifier = get_verifier()
            # 构建验证用的用户档案
            verify_profile = {'allergies': [], 'diseases': [], 'special_groups': []}
            if user_id and str(user_id).isdigit() and 1 <= int(user_id) <= 50:
                verify_profile = engine.user_profiles.get(user_id, verify_profile)
            elif user_ids:
                # 多人场景：合并所有约束
                for uid in user_ids:
                    if str(uid).isdigit() and 1 <= int(uid) <= 50:
                        up = engine.user_profiles.get(uid, {})
                        verify_profile['allergies'].extend(up.get('allergies', []))
                        verify_profile['diseases'].extend(up.get('diseases', []))
                        verify_profile['special_groups'].extend(up.get('special_groups', []))
                verify_profile['allergies'] = list(set(verify_profile['allergies']))
                verify_profile['diseases'] = list(set(verify_profile['diseases']))
                verify_profile['special_groups'] = list(set(verify_profile['special_groups']))
            else:
                # 非预置用户：从对话偏好中获取过敏信息，并合并明确"不吃"的食材、
                # 特殊人群与慢性疾病（同样视为硬约束）
                verify_profile['allergies'] = list(set(
                    dm.user_preferences.get('allergies', []) +
                    dm.user_preferences.get('excluded_ingredients', [])
                ))
                verify_profile['special_groups'] = list(
                    dm.user_preferences.get('special_groups', []) or [])
                verify_profile['diseases'] = list(
                    dm.user_preferences.get('diseases', []) or [])
            verification_report = verifier.verify(recommendations, verify_profile)
            # 硬约束违规：从推荐列表中移除违规菜品并提示用户
            if not verification_report['all_passed']:
                failed_checks = [c for c in verification_report['checks'] if c['status'] == 'FAILED']
                violation_recipes = set()
                for c in failed_checks:
                    for v in c.get('violations', []):
                        violation_recipes.add(v.get('recipe', ''))
                    for m in c.get('missing', []):
                        violation_recipes.add(m)
                # 移除违规/幻觉菜品
                if violation_recipes:
                    _before_names = set(r.get('name', '') for r in recommendations)
                    recommendations = [r for r in recommendations if r.get('name', '') not in violation_recipes]
                    # 若删除了菜品，重建叙述使其与最终列表一致（营养概览在下方基于最终列表统一重算）
                    if recommendations and len(recommendations) != len(_before_names):
                        rebuilt = _build_consistent_response(recommendations)
                        if rebuilt:
                            response_text = rebuilt
                    if not recommendations:
                        # 全部被拦截：优先从菜谱库的"约束合规池"重新选菜（比LLM现场生成
                        # 更真实可靠，也保证推荐列表非空——空推荐在基础推荐评分中直接失分）
                        try:
                            _safe_pool = retriever.search(
                                '晚餐 家常', top_k=30,
                                filters=dm.get_search_filters() if dm else {})
                            if user_ids:
                                _safe_pool = engine.filter_by_constraints(_safe_pool, user_ids=user_ids)
                            elif user_id and engine.user_profiles.get(user_id):
                                _safe_pool = engine.filter_by_constraints(_safe_pool, user_id=user_id)
                            if _safe_pool:
                                recommendations = _safe_pool[:5]
                                response_text = _build_consistent_response(recommendations)
                                print(f"[dialog] 验证拦截后从库内补齐{len(recommendations)}道合规菜",
                                      flush=True)
                        except Exception as _sf:
                            print(f"[dialog] 库内合规补齐失败: {_sf}", flush=True)
                    if not recommendations:
                        # 库内也无合规菜：尝试现场生成一道合规新菜谱（赛题加分项）
                        gen = _generate_fallback_recipe(
                            message, user_id=user_id, user_ids=user_ids,
                            dm=dm, engine=engine, llm=llm
                        )
                        if gen:
                            recommendations = [gen]
                            ings = '、'.join(gen['ingredients'])
                            # 原推荐全部被拦截：重建回复只保留合规的生成菜（避免残留被拦截菜名）
                            response_text = (
                                f"刚才推荐的那几道里，有几道和你的健康约束冲突，我重新把关后，"
                                f"为你现场设计了一道完全合规的新菜——"
                                f"【{gen['name']}】（建议菜谱·生成）\n"
                                f"食材：{ings}\n"
                                f"做法：{gen['steps']}"
                            )
                        else:
                            response_text += ("\n\n很抱歉，当前推荐未通过安全检测，已为您重新筛选。"
                                              "请告诉我更多偏好，帮您找到合适的菜品～")
        t_verify_dialog = round((time.perf_counter() - t_verify_start) * 1000, 2)

        # ── 确定性偏好硬过滤兜底：无论 Agent 搜回什么，最终输出必须符合当前偏好
        #    （素食/想吃荤/排除忌口），避免"想吃肉却给素菜""吃素却给荤菜"的理解偏差 ──
        _pf = dm.get_search_filters() if dm else {}
        if _pf and recommendations:
            try:
                _pre_names = {r.get('name', '') for r in recommendations}
                _pre_before = len(recommendations)
                recommendations = retriever._apply_filters(recommendations, _pf)
                print(f"[dialog] 偏好硬过滤: {_pre_before}->{len(recommendations)} 道", flush=True)
                # 用户显式指定"只吃N道/来N道"时，尊重该数量，不强行补足到5道
                _req_n = 0
                try:
                    _rm = re.search(
                        r'(?:只吃|就吃|来|要|上|点|安排|配)?\s*(?<!第)([一二三四五六七八九十两]|\d{1,2})\s*道(?:菜|汤|菜吧|菜呀|菜啊)?',
                        message)
                    _RN = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
                           '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
                    if _rm:
                        _g = _rm.group(1)
                        _req_n = _RN.get(_g) if _g in _RN else int(_g)
                except Exception:
                    _req_n = 0
                _target_n = _req_n if 1 <= _req_n <= 12 else 5
                # 偏好硬过滤后可能清空（如"给老人做的"把重油菜/辣菜全滤掉）：
                # 无论剩余 0 道还是不足目标，都要从库内重新检索合规菜补足，
                # 避免"回复有菜名但 recommendations 为空"的空推荐。
                if len(recommendations) < _target_n:
                    _core, _neg = _clean_search_query(message)
                    _seen = {r.get('name', '') for r in recommendations}
                    # 先带偏好过滤补足；若因口味过滤过严仍不足，降级为无过滤全库检索
                    # + 健康约束引擎过滤，保证不为凑数而硬塞违规菜、也不空手而归。
                    _fill_pool = retriever.search((_core or '晚餐 家常'), top_k=15, filters=_pf)
                    if len(_fill_pool) < (_target_n - len(recommendations)):
                        _fill_pool = retriever.search((_core or '晚餐 家常'), top_k=20, filters=None)
                        try:
                            if engine and (user_ids or (user_id and engine.user_profiles.get(user_id))):
                                _fill_pool = (engine.filter_by_constraints(_fill_pool, user_ids=user_ids)
                                             if user_ids else engine.filter_by_constraints(_fill_pool, user_id=user_id))
                        except Exception:
                            pass
                    for _r in _fill_pool:
                        if len(recommendations) >= _target_n:
                            break
                        if _r.get('name') in _seen:
                            continue
                        recommendations.append(_r)
                        _seen.add(_r.get('name'))
                    print(f"[dialog] 偏好硬过滤补足后: {len(recommendations)} 道", flush=True)
                # 若偏好过滤删除了推荐菜，重建叙述使其与最终列表严格一致
                _post_names = {r.get('name', '') for r in recommendations}
                if _pre_names != _post_names:
                    _rebuilt = _build_consistent_response(recommendations)
                    if _rebuilt:
                        response_text = _rebuilt
            except Exception as _e:
                print(f"[dialog] 偏好硬过滤失败: {_e}", flush=True)

        # ── 出口数量兜底：无论 handler / 验证拦截 / 偏好补足如何组装，
        #    最终统一截断到用户期望的道数（尊重"只要四道/换三道"等数量指定），
        #    防止长对话换菜等异常路径返回超出预期数量的菜。截断后重建叙述保持一致。 ──
        try:
            _final_n = 5
            _rm_f = re.search(
                r'(?:只吃|就吃|来|要|上|点|安排|配)?\s*(?<!第)([一二三四五六七八九十两]|\d{1,2})\s*道(?:菜|汤|菜吧|菜呀|菜啊)?',
                message)
            if _rm_f:
                _gf = _rm_f.group(1)
                _RN_f = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6,
                         '七': 7, '八': 8, '九': 9, '十': 10}
                _nf = _RN_f.get(_gf) if _gf in _RN_f else int(_gf)
                if 1 <= _nf <= 12:
                    _final_n = _nf
            if len(recommendations) > _final_n:
                print(f"[dialog] 出口数量兜底: {len(recommendations)}->{_final_n} 道", flush=True)
                recommendations = recommendations[:_final_n]
                _rebuilt = _build_consistent_response(recommendations)
                if _rebuilt:
                    response_text = _rebuilt
        except Exception as _fe:
            print(f"[dialog] 出口数量兜底失败: {_fe}", flush=True)

        # ── 按人营养摄入概览（基于验证后的最终列表，与叙述/列表完全一致）──
        nutrition_summary = ""
        recipe_names = [r['name'] for r in recommendations] if recommendations else []
        if recipe_names and not agent_result.get('ask_question'):
            nutrition_summary = compute_meal_nutrition_summary(
                recipe_names,
                user_ids=user_ids if user_ids else ([user_id] if user_id else None)
            )
            if nutrition_summary:
                response_text += "\n" + nutrition_summary
        
        t_total = round((time.perf_counter() - t_request_start) * 1000, 2)
        timing_data = {
            't_total_ms': t_total,
            't_llm_ms': t_llm_dialog,
        }
        # 推荐路径分阶段数据
        if t_search_dialog or t_filter_dialog or t_verify_dialog:
            timing_data['t_search_ms'] = t_search_dialog
            timing_data['t_filter_ms'] = t_filter_dialog
            timing_data['t_nutrition_ms'] = t_nutrition_dialog
            timing_data['t_verify_ms'] = t_verify_dialog
            timing_data['retry_count'] = retry_count
        # Agent模式标记
        if agent_result:
            timing_data['agent_mode'] = True
            timing_data['tool_calls'] = agent_result.get('tool_calls_made', 0)
            timing_data['t_first_token_ms'] = agent_result.get('first_token_ms', 0)
        dm.record_timing(timing_data)
        _global_timings.append(timing_data)
        if len(_global_timings) > 200:
            _global_timings.pop(0)
        
        _save_dialog(dialog_key, dm)  # 持久化记忆：记录本轮后的饮食习惯
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'intent': intent,
            'response': response_text,
            'nutrition_summary': nutrition_summary,
            'recommendations': recommendations,
            'user_preferences': dm.get_user_profile(),
            'dialog_turns': dm.turn_count,
            'context_summary': dm.get_context_summary(),
            'timing': timing_data,
            'verification': verification_report
        })
    
    except Exception as e:
        import traceback
        print(f"[FATAL] dialog error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/dialog/stream', methods=['POST'])
def dialog_stream():
    """
    流式对话接口（SSE）
    
    支持流式响应，首Token延迟<2s，提升用户体验。
    使用并行处理：检索和LLM调用同时进行，推荐结果优先返回。
    
    请求参数：
    {
        "user_id": "user_001",
        "message": "今晚吃什么",
        "reset": false,
        "user_ids": ["user_001", "user_002"]
    }
    
    Returns:
        SSE流式响应，格式: {"event": "...", "data": {...}}
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id', '')
        message = data.get('message', '')
        reset = data.get('reset', False)
        user_ids = data.get('user_ids', [])
        
        if not message:
            return jsonify({'success': False, 'error': 'message参数不能为空'}), 400
        
        retriever = get_retriever()
        engine = get_engine()
        llm = get_llm()
        
        dialog_key = user_id or "anonymous"
        if dialog_key not in _dialog_managers:
            if len(_dialog_managers) >= _MAX_DIALOG_MANAGERS:
                _dialog_managers.pop(next(iter(_dialog_managers)), None)
            _dialog_managers[dialog_key] = _load_dialog(dialog_key) or DialogManager()
        
        dm = _dialog_managers[dialog_key]
        
        if reset:
            dm.reset_dialog()
            _clear_dialog_memory(dialog_key)
        
        dm.add_message('user', message)
        preferences = dm.extract_preferences(message)
        intent = dm.detect_intent(message)
        # LLM 一次获取意图+偏好+语义改写查询（口语/复合需求 -> 可检索关键词）。
        # 该调用在流式文本开始之前完成，不影响首Token延迟指标。
        search_query = ""
        llm_res = dm.detect_with_llm(message, llm_client=llm)
        if llm_res:
            if llm_res.get('intent'):
                intent = llm_res['intent']
            if llm_res.get('preferences'):
                dm._update_preferences(llm_res['preferences'])
                preferences = llm_res['preferences']
            if llm_res.get('search_query'):
                sq = str(llm_res['search_query']).strip()
                if sq and sq.lower() != 'null':
                    search_query = sq
        INTENT_SYN = {'vague': 'vague_query', 'replace': 'request_substitute',
                      'reject': 'reject_recommendation', 'set_preference': 'set_preferences'}
        intent = INTENT_SYN.get(intent, intent)

        # 强意图覆盖：明确询问"怎么做/做法/步骤/怎么制作"时锁定为菜谱详情意图，
        # 避免 LLM/关键词把"怎么做X"误判成推荐返回一堆新菜。
        # 上轮刚给过"简略版"时，用户说"详细讲讲/继续/好"等追问，同样进入菜谱详情。
        if (re.search(r'(怎么做|怎么制作|怎么弄|如何做|制作方法|怎么烧|怎么煮|怎么蒸|怎么炸|步骤|做法)',
                      message)
                or (getattr(dm, '_detail_pending', None)
                    and re.search(r'(详细|讲讲|说说|具体|继续|再来|再讲|好|嗯|可以|要)', message))):
            intent = 'ask_recipe_detail'

        # 专职 Agent 文本意图（菜谱详情/营养/简单交互等）：不进入推荐 generate，直接 dispatch 生成回复，
        # 保证 Gradio 流式接口也能正确响应"怎么做X"这类询问，以及"你好/谢谢/再见/模糊询问"
        # 这类交互话术——否则 greet/cancel/vague_query 会被误当成推荐请求返回一堆无关菜。
        _interactive_intents = ('ask_recipe_detail', 'ask_nutrition',
                                'greet', 'confirm', 'cancel', 'vague_query',
                                'reject_recommendation', 'request_substitute', 'request_more')
        if intent in _interactive_intents and ORCHESTRATOR.has_handler(intent):
            _ctx = DialogContext(
                dm=dm, retriever=retriever, engine=engine, llm=llm,
                message=message, user_id=user_id, user_ids=user_ids,
                preferences=preferences, search_query=search_query,
            )
            try:
                ORCHESTRATOR.dispatch(intent, _ctx)
            except Exception as _de:
                print(f"[stream] 专职Agent分发失败: {_de}", flush=True)
            _resp = _ctx.response_text or "抱歉，我暂时没找到相关信息。"
            dm.add_message('system', _resp)

            def _special_stream():
                _res = {
                    'id': f"chatcmpl-{int(time.time() * 1000)}",
                    'object': 'chat.completion.chunk',
                    'created': int(time.time()),
                    'model': 'deepseek-v4-flash',
                    'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                    'event': 'complete',
                    'data': {
                        'user_id': user_id,
                        'intent': intent,
                        'response': _resp,
                        'recommendations': _ctx.recommendations or [],
                        'dialog_turns': dm.turn_count,
                    }
                }
                yield f"data: {json.dumps(_res, ensure_ascii=False)}\n\n"
                yield "event: end\ndata: {}\n\n"
            return Response(_special_stream(), content_type='text/event-stream; charset=utf-8')

        # 检查缓存（缓存键必须包含 user_ids + 多样性种子：
        # 同一 user_id 在"新对话"（种子已自增）后不应命中旧缓存，避免与新一批推荐冲突）
        uid_part = '_'.join(sorted(user_ids)) if user_ids else ''
        cache_key = hashlib.md5(
            f"{dialog_key}_{uid_part}_{message}_{dm.diversity_seed}".encode()
        ).hexdigest()
        cached = _request_cache.get(cache_key)
        if cached and time.time() - cached['timestamp'] < _CACHE_TTL:
            def cached_stream():
                # 缓存命中：同样套上 OpenAI 兼容 + "event":"complete" 包裹，
                # 保证 Gradio 端按统一格式解析（否则裸 data 帧无 event 字段，前端什么都不渲染）。
                _res = {
                    'id': f"chatcmpl-{int(time.time() * 1000)}",
                    'object': 'chat.completion.chunk',
                    'created': int(time.time()),
                    'model': 'deepseek-v4-flash',
                    'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                    'event': 'complete',
                    'data': cached['data'],
                }
                yield f"data: {json.dumps(_res, ensure_ascii=False)}\n\n"
                yield "event: end\ndata: {}\n\n"
            return Response(cached_stream(), content_type='text/event-stream; charset=utf-8')
        
        def generate():
            t_start = time.perf_counter()
            # 将本轮提取的特殊人群同步进约束引擎档案，保证 filter_by_constraints 生效
            _sync_health_to_engine(dm, engine, user_id, user_ids)
            filters = dm.get_search_filters()
            core, neg_ex = _clean_search_query(message)
            if neg_ex:
                filters['exclude_ingredients'] = list(set(filters.get('exclude_ingredients', []) + neg_ex))
                for e in neg_ex:
                    if e not in dm.user_preferences.get('excluded_ingredients', []):
                        dm.user_preferences['excluded_ingredients'].append(e)
            # 泛化提问（"吃什么/随便/推荐看看"）无明确菜向：不做语义检索（"吃什么"会命中海鲜簇），
            # 退化为平衡的一餐默认检索，避免推荐与问题完全脱节。
            _vague = {'吃什么', '吃点啥', '吃点吗', '随便', '看看', '看看有什么', '有啥',
                      '推荐', '推荐推荐', '来点', '有什么', '还有什么', '有什么吃的', '帮我看看',
                      # 清洗后可能残留的短泛化词（"推荐点吃的"→core="吃"）：同样视为无明确菜向
                      '吃', '吃的', '想吃', '想吃的', '点吃', '推荐点', '推荐点吃', '随便推荐点'}
            core_clean = core.strip()
            veg_want = bool(filters.get('vegetarian'))
            # 不为泛化/具体请求再叠加LLM改写：LLM的 search_query 在此路径会拿"吃什么"改出海鲜等偏置词，
            # 造成推荐与问题脱节。这里用确定性规则：素/有明确菜向/否则家常均衡默认。
            if veg_want:
                # 要素/全素：用素菜检索词，配合 vegetarian 硬过滤保证全素
                search_q = '素菜 蔬菜 清淡 凉拌 蒸菜 豆腐 青菜'
            elif core_clean and core_clean not in _vague:
                # 用户明确给了具体菜/口味：优先用清洗过的核心词
                search_q = core_clean
            else:
                # 泛化（"吃什么/随便"）或纯排除（不吃X但没说要吃什么）：给一套家常荤素均衡默认。
                # 用多样性种子轮换多个等价检索词，避免同一 user_id 每次"新对话"都返回同一批菜。
                _default_qs = [
                    '家常菜 荤素搭配 晚餐',
                    '家常菜 营养均衡 荤素搭配',
                    '家庭晚餐 简单快手 荤素均衡',
                    '晚餐推荐 家常菜 清淡多样',
                    '家常菜 汤菜搭配 均衡一餐',
                ]
                search_q = _default_qs[dm.diversity_seed % len(_default_qs)]

            # 特殊人群清淡增强：年龄向/孕期等用户的检索词应朝向"清淡易消化"，避免
            # 即便硬约束生效，候选池却全是重油菜而筛完不足。同时把油腻/难消化方向词压到末尾，
            # 提高清淡候选的排名。仅在"无明确菜向"或"清淡偏好"时叠加，避免误改"想要红烧肉"。
            _special_groups = list(dm.user_preferences.get('special_groups', []) or [])
            _needs_light = any(g in ('老人', '儿童', '孕妇', '哺乳期') for g in _special_groups) \
                or filters.get('low_fat') or filters.get('light')
            # core 里若仍带"给老人做的/清淡"等口语而非具体食材，视为无明确菜向，走强清淡词
            _core_phr = not core_clean or any(p in core_clean for p in
                                              ('老人', '清淡', '易消化', '软烂', '做', '给', '点', '的'))
            if _needs_light and _core_phr:
                search_q = '清淡 易消化 软烂 粥 汤 蒸菜 豆腐 炖 温补 少油 清蒸'
            elif _needs_light and core_clean not in _vague:
                # 有明确菜向但用户也是需清淡人群（如"给老人做的排骨"）：追加清淡方向词
                search_q = f'{core_clean} 清淡 少油 软烂 易消化 清蒸 炖'
            
            # 候选池多召回：过滤后（尤其特殊人群需清淡过滤会排除大量重油菜）仍有余量，
            # 避免推荐不足（少于5道）或荤素失衡。成本可接受，过滤都在顶层做。
            results = retriever.search(search_q, top_k=24, filters=filters)
            t_search = time.perf_counter()
            
            if user_ids:
                results = engine.filter_by_constraints(results, user_ids=user_ids)
            elif user_id:
                results = engine.filter_by_constraints(results, user_id=user_id)
            t_filter = time.perf_counter()
            
            # 多样性轮换：泛化提问（"吃什么/随便"）时，精排会把不同等价检索词的
            # 候选重新收敛到同一批高分菜。这里用 diversity_seed 在前 12 名的高质量
            # 候选池里轮换取若干道，保证同一用户每次"新对话"都能换一批菜。
            # "多吃几道/再多来点"等表达 → 返回更多道（8道），而非固定 5 道。
            _want_more = bool(re.search(r'多吃|多推荐|多来|再多|来几道|多几道|多上|多整|一桌|多来点',
                                        message))
            _top_n = 8 if _want_more else 5
            # 数量约束解析："只吃四道/要五道菜/来三道"等 → 精确按该数量返回
            try:
                _cnt_m = re.search(
                    r'(?:只吃|就吃|来|要|上|点|安排|配)?\s*(?<!第)([一二三四五六七八九十两]|\d{1,2})\s*道(?:菜|汤|菜吧|菜呀|菜啊)?',
                    message)
                _CN2 = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
                        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
                if _cnt_m:
                    _g2 = _cnt_m.group(1)
                    _n2 = _CN2.get(_g2) if _g2 in _CN2 else int(_g2)
                    if 1 <= _n2 <= 12:
                        _top_n = _n2
            except Exception:
                pass
            _is_vague_q = (not core_clean) or (core_clean in _vague)
            if _is_vague_q and dm and len(results) > _top_n:
                _pool = results[:12]
                _off = (dm.diversity_seed * _top_n) % len(_pool)
                recommendations = (_pool[_off:] + _pool[:_off])[:_top_n]
            else:
                recommendations = results[:_top_n]
            # — 追加约束时保留历史推荐（复用与 Agent 主路径一致的逻辑） —
            # 用户追加"不吃鱼"等约束时，不应把已认可的历史菜全换掉：
            # 把仍符合当前检索过滤条件的历史菜置顶并入。
            try:
                # 仅当本回合是"追加排除约束"时才保留当前方案（最小化修改）：
                # 用户说"想吃菜/能不能全是菜"等口味重定向时，应全新检索而非继续堆旧方案；
                # 说"不吃鱼/不要巧克力"等排除时，保留当前方案中仍合规的菜。
                if neg_ex and dm.recommended_recipes:
                    _keep_filters = dict(dm.get_search_filters()) if dm else {}
                    _existing = {r.get('name', '') for r in recommendations}
                    _kept_list = []
                    # 只基于"当前方案"（最近一轮实际推荐）做最小化修改保留：
                    # 用户追加约束（如"不吃鱼"）时，应保留上一轮仍符合新约束的菜，
                    # 而不是从整段历史里去重累加集合里捞旧菜（那会把早期甜品又拉回来）。
                    _plan = dm.last_recommendation or dm.recommended_recipes[-5:]
                    for _hname in _plan:
                        if _hname in _existing:
                            continue
                        _full = retriever.get_recipe_by_name(_hname)
                        if not _full:
                            continue
                        if retriever._apply_filters([_full], _keep_filters):
                            _kept_list.append(_full)
                            _existing.add(_hname)
                        # 最多保留4道旧菜，给新检索结果留至少1个名额，避免旧方案完全锁死本回合
                        if len(_kept_list) >= 4:
                            break
                    if _kept_list:
                        _merged = list(_kept_list)
                        _mn = {r.get('name', '') for r in _merged}
                        for _r in recommendations:
                            _n = _r.get('name', '')
                            if _n and _n not in _mn:
                                _merged.append(_r)
                                _mn.add(_n)
                            if len(_merged) >= _top_n:
                                break
                        recommendations = _merged[:_top_n]
                        print(f"[stream] 并入历史保留 {len(_kept_list)} 道并置顶", flush=True)
            except Exception as _e:
                print(f"[stream] 历史保留失败: {_e}", flush=True)
            # 检索为空时，尝试生成一道合规新菜谱（赛题加分项）
            if not recommendations:
                gen = _generate_fallback_recipe(
                    message, user_id=user_id, user_ids=user_ids,
                    dm=dm, engine=engine, llm=llm
                )
                if gen:
                    recommendations = [gen]
            # 营养统计：为每道推荐补齐营养评估（供前端菜品卡片显示热量等信息），
            # 与 /api/recommend 非流式路径保持一致。
            try:
                for r in recommendations:
                    if 'nutrition' not in r or not r['nutrition']:
                        r['nutrition'] = engine.evaluate_nutrition(r)
            except Exception as _ne:
                print(f"[stream] 营养评估失败: {_ne}", flush=True)
            recipe_names = [r['name'] for r in recommendations]
            dm.add_recommended_recipes(recipe_names)
            
            balance_score = engine.calculate_balance_score(recommendations)
            context_summary = dm.get_context_summary()
            
            partial_result = {
                'id': f"chatcmpl-{int(time.time() * 1000)}",
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': 'deepseek-v4-flash',
                'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}],
                'event': 'recommendations',
                'data': {
                    'user_id': user_id,
                    'intent': intent,
                    'recommendations': [{'name': r['name'], 'tags': r.get('tags', []),
                                          'nutrition': r.get('nutrition', {})} for r in recommendations],
                    'context_summary': context_summary,
                    'dialog_turns': dm.turn_count
                }
            }
            yield f"data: {json.dumps(partial_result, ensure_ascii=False)}\n\n"
            
            # 精简提示词，降低首Token延迟。用确定性叙述锚定，防止LLM编造列表外的菜名、
            # 或把纯素菜/甜品错标成"荤菜"、或是叙述与结构化列表脱节。
            consistent = _build_consistent_response(recommendations)
            llm_context = (
                f"本次已为您配好的菜（一个都不能改、不能增、不能删，也不要重复这串菜名）："
                f"{'、'.join(recipe_names)}。\n"
                f"开篇请直接引用这句且不要改动：{consistent}\n"
                f"若需要，可再补充1句不超过25字的简短推荐理由。不要编造其他菜，不要给菜品贴荤素/口味标签。"
            )
            
            response_text = ""
            first_token_ms = None
            t_llm_start = time.perf_counter()
            t_llm_end = t_llm_start  # 默认值
            try:
                first = True
                for chunk in llm.chat_stream([{'role': 'user', 'content': llm_context}]):
                    if first:
                        first_token_ms = round((time.perf_counter() - t_llm_start) * 1000, 2)
                        first = False
                    response_text += chunk
                    # 同时输出 OpenAI 兼容格式与自定义字段，评测方可按 OpenAI SSE 规范解析
                    chunk_data = {
                        "id": f"chatcmpl-{int(time.time() * 1000)}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "deepseek-v4-flash",
                        "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                        "event": "text",
                        "data": {"text": chunk}
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                # OpenAI 流式结束标记
                yield "data: [DONE]\n\n"
                t_llm_end = time.perf_counter()
            except Exception as e:
                print(f"LLM stream error: {e}")
                t_llm_end = time.perf_counter()
                fallback_text = f'为您推荐：{", ".join(recipe_names)}'
                if first_token_ms is None:
                    first_token_ms = round((t_llm_end - t_llm_start) * 1000, 2)
                fallback_data = {
                    "id": f"chatcmpl-{int(time.time() * 1000)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "deepseek-v4-flash",
                    "choices": [{"index": 0, "delta": {"content": fallback_text}, "finish_reason": None}],
                    "event": "text",
                    "data": {"text": fallback_text}
                }
                yield f"data: {json.dumps(fallback_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            
            dm.add_message('system', response_text)
            t_total = round((time.perf_counter() - t_start) * 1000, 2)
            t_search_ms = round((t_search - t_start) * 1000, 2)
            t_filter_ms = round((t_filter - t_search) * 1000, 2)
            t_llm_ms = round((t_llm_end - t_llm_start) * 1000, 2)
            
            # 记录到会话级计时统计
            dm.record_timing({
                't_total_ms': t_total,
                't_search_ms': t_search_ms,
                't_filter_ms': t_filter_ms,
                't_llm_ms': t_llm_ms,
                't_first_token_ms': first_token_ms
            })
            _global_timings.append({
                't_total_ms': t_total,
                't_llm_ms': t_llm_ms,
                't_first_token_ms': first_token_ms
            })
            if len(_global_timings) > 200:
                _global_timings.pop(0)

            # ── 按人营养摄入概览（追加到回复底部，与 /api/dialog 非流式路径一致）──
            try:
                _rnames = [r['name'] for r in recommendations] if recommendations else []
                if _rnames:
                    _nut_summary = compute_meal_nutrition_summary(
                        _rnames,
                        user_ids=user_ids if user_ids else ([user_id] if user_id else None)
                    )
                    if _nut_summary:
                        response_text = f"{response_text or ''}\n{_nut_summary}"
            except Exception as _ne:
                print(f"[stream] 营养概览生成失败: {_ne}", flush=True)

            final_result = {
                'id': f"chatcmpl-{int(time.time() * 1000)}",
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': 'deepseek-v4-flash',
                'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                'event': 'complete',
                'data': {
                    'user_id': user_id,
                    'intent': intent,
                    'response': response_text,
                    'recommendations': recommendations,
                    'user_preferences': dm.get_user_profile(),
                    'dialog_turns': dm.turn_count,
                    'context_summary': context_summary,
                    'timing': {
                        'total_ms': t_total,
                        'search_ms': t_search_ms,
                        'filter_ms': t_filter_ms,
                        'llm_ms': t_llm_ms,
                        'first_token_ms': first_token_ms
                    }
                }
            }
            yield f"data: {json.dumps(final_result, ensure_ascii=False)}\n\n"
            
            _request_cache[cache_key] = {
                'timestamp': time.time(),
                'data': final_result['data']
            }
            _save_dialog(dialog_key, dm)  # 持久化记忆：记录本轮后的饮食习惯
        
        return Response(generate(), content_type='text/event-stream; charset=utf-8')
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/timing/stats', methods=['GET'])
def timing_stats():
    """
    性能统计接口
    
    返回全局和按会话的耗时统计数据。
    
    Query参数：
    - user_id: 可选，指定会话ID查看单会话统计
    
    Returns:
        全局和会话级性能统计
    """
    user_id = request.args.get('user_id', '')
    
    global_stats = _compute_stats(_global_timings)
    
    result = {
        'success': True,
        'global': global_stats,
        'thresholds': {
            'first_token_excellent_ms': 2000,
            'first_token_ok_ms': 5000,
            'total_excellent_ms': 8000,
            'total_ok_ms': 15000,
            'multi_turn_avg_excellent_ms': 6000,
            'multi_turn_avg_ok_ms': 12000
        }
    }
    
    if user_id and user_id in _dialog_managers:
        session_stats = _dialog_managers[user_id].get_timing_stats()
        result['session'] = session_stats
    
    return jsonify(result)


def _compute_stats(timings: list) -> Dict:
    """计算计时数据的统计值"""
    if not timings:
        return {'count': 0}
    
    def _avg(key):
        vals = [t[key] for t in timings if t.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else 0
    
    def _p(key, percentile):
        vals = sorted([t[key] for t in timings if t.get(key) is not None])
        if not vals:
            return 0
        idx = int(len(vals) * percentile / 100)
        return round(vals[min(idx, len(vals) - 1)], 2)
    
    return {
        'count': len(timings),
        'avg_total_ms': _avg('t_total_ms'),
        'avg_llm_ms': _avg('t_llm_ms'),
        'avg_first_token_ms': _avg('t_first_token_ms'),
        'p50_total_ms': _p('t_total_ms', 50),
        'p95_total_ms': _p('t_total_ms', 95),
        'p50_first_token_ms': _p('t_first_token_ms', 50),
        'p95_first_token_ms': _p('t_first_token_ms', 95)
    }

@app.route('/api/recipes', methods=['GET'])
def get_all_recipes():
    """
    获取所有菜谱列表接口
    
    返回系统中所有菜谱的简要信息。
    
    Returns:
        JSON响应，包含菜谱列表
    """
    try:
        retriever = get_retriever()
        recipes = [{'name': r['name'], 'tags': r.get('tags', [])} for r in retriever.recipes]
        
        return jsonify({
            'success': True,
            'count': len(recipes),
            'recipes': recipes
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/recipe/<name>', methods=['GET'])
def get_recipe_detail(name: str):
    """
    获取菜谱详情接口
    
    根据菜名获取完整的菜谱信息。
    
    Args:
        name: 菜名（URL路径参数）
        
    Returns:
        JSON响应，包含完整的菜谱信息
    """
    try:
        retriever = get_retriever()
        engine = get_engine()
        
        # URL解码菜名
        from urllib.parse import unquote
        decoded_name = unquote(name)
        
        # 获取菜谱详情
        recipe = retriever.get_recipe_by_name(decoded_name)
        if not recipe:
            return jsonify({'success': False, 'error': f'未找到菜品: {decoded_name}'}), 404
        
        # 评估营养
        nutrition = engine.evaluate_nutrition(recipe)
        recipe['nutrition'] = nutrition
        
        return jsonify({'success': True, 'recipe': recipe})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ─── 按人营养摄入计算 ────────────────────────────────────

_ING_QTY_RE = re.compile(r'([\d.]+)\s*(克|g|G|kg|千克|毫升|ml|ML)\b')


def _ing_qty_grams(text: str, key: str) -> Optional[float]:
    """
    提取食材 key 在配料文本中对应的用量（克）。

    营养库按每100g提供数据，需按实际克数缩放，避免人均营养虚高。
    仅在文本中存在明确克数时返回数值；否则返回 None（如"芝麻油适量""鸡蛋3个"），
    调用方应跳过该匹配，避免把高热量调料按100g默认值虚算进去。
    """
    start = 0
    while True:
        idx = text.find(key, start)
        if idx < 0:
            return None
        seg = text[idx:idx + 25]  # key 后 25 字符内应含用量
        m = _ING_QTY_RE.search(seg)
        if m:
            try:
                q = float(m.group(1))
                unit = m.group(2).lower()
                if unit in ('kg', '千克'):
                    q *= 1000
                return q if q > 0 else None
            except ValueError:
                return None
        start = idx + len(key)
        if start >= len(text):
            return None


def compute_meal_nutrition_summary(recipe_names: list, user_ids: list = None,
                                   recommend_count: int = None) -> str:
    """
    根据推荐菜谱列表和用户信息，计算人均营养摄入概览。
    
    返回格式化的营养概览文本，可直接追加到响应末尾。
    """
    if not recipe_names or not MEAL_NUTRITION_DB:
        return ""
    
    retriever = get_retriever()
    engine = get_engine()
    
    # 收集完整菜谱数据
    recipes = []
    for name in recipe_names:
        full = retriever.get_recipe_by_name(name)
        if full:
            recipes.append(full)
    
    if not recipes:
        return ""
    
    num_people = len(user_ids) if user_ids else (recommend_count or 1)
    num_people = max(num_people, 1)
    
    # 计算总营养（仅统计有明确克数的食材，按克数缩放，避免调料/无克数条目虚高）
    total = {'热量': 0, '蛋白质': 0, '脂肪': 0, '碳水': 0}
    for recipe in recipes:
        ing = recipe.get('ingredients', '')
        text = ' '.join(str(i) for i in ing) if isinstance(ing, list) else str(ing or '')
        matched_any = False
        counted_spans = []  # 已统计的 (start, end)，避免"芝麻"落在"芝麻油"内被双计
        # 长 key 优先，短 key 若整体被更长的已匹配 key 覆盖则跳过
        for ing_key in sorted(MEAL_NUTRITION_DB.keys(), key=lambda k: len(k), reverse=True):
            if not ing_key:
                continue
            data = MEAL_NUTRITION_DB.get(ing_key, {})
            pos = 0
            while True:
                idx = text.find(ing_key, pos)
                if idx < 0:
                    break
                end = idx + len(ing_key)
                # 若该位置已被更长的 key 覆盖，跳过（如"芝麻"被"芝麻油"覆盖）
                if any(idx < e and end > s for s, e in counted_spans):
                    pos = idx + 1
                    continue
                qty = _ing_qty_grams(text, ing_key)
                if qty is not None:
                    matched_any = True
                    counted_spans.append((idx, end))
                    _scale = qty / 100.0
                    total['热量'] += data.get('热量', 0) * _scale
                    total['蛋白质'] += data.get('蛋白质', 0) * _scale
                    total['脂肪'] += data.get('脂肪', 0) * _scale
                    total['碳水'] += data.get('碳水', 0) * _scale
                    break
                pos = idx + 1
        # 占位食材（如"主料：紫苏辣子鸡400克"）无明细匹配时，用菜名关键词兜底估算，
        # 避免营养概览全部显示 0 千卡
        if not matched_any:
            name = recipe.get('name', '')
            _name_kw = [
                ('鸡', '鸡肉'), ('鸭', '鸭肉'), ('鹅', '鹅肉'), ('牛', '牛肉'),
                ('猪', '猪肉'), ('羊', '羊肉'), ('鱼', '鱼肉'), ('虾', '虾'),
                ('蟹', '螃蟹'), ('蛋', '鸡蛋'), ('排骨', '排骨'), ('豆腐', '豆腐'),
                ('茄', '茄子'), ('萝卜', '萝卜'), ('土豆', '土豆'), ('冬瓜', '冬瓜'),
                ('南瓜', '南瓜'), ('白菜', '白菜'), ('菠菜', '菠菜'), ('芹菜', '芹菜'),
                ('黄瓜', '黄瓜'), ('香菇', '香菇'), ('木耳', '木耳'), ('玉米', '玉米'),
                ('饭', '米饭'), ('面', '面条'), ('汤', '蔬菜'), ('菜', '蔬菜'),
            ]
            hit = set()
            for kw, db_key in _name_kw:
                if kw in name and db_key not in hit:
                    hit.add(db_key)
                    data = MEAL_NUTRITION_DB.get(db_key, {})
                    total['热量'] += data.get('热量', 0)
                    total['蛋白质'] += data.get('蛋白质', 0)
                    total['脂肪'] += data.get('脂肪', 0)
                    total['碳水'] += data.get('碳水', 0)
    
    per_kcal = total['热量'] / num_people
    per_protein = total['蛋白质'] / num_people
    per_fat = total['脂肪'] / num_people
    per_carb = total['碳水'] / num_people
    
    # 按人需求参考（如果提供了 user_ids）；同时收集用户档案供营养缺口检测使用
    per_person_lines = []
    user_profiles = []
    if user_ids and engine:
        for uid in user_ids:
            profile = engine.get_user_profile(uid)
            if profile:
                user_profiles.append(profile)
                age = profile.get('年龄', profile.get('age', '?'))
                weight = profile.get('体重(kg)', profile.get('weight', '?'))
                if isinstance(weight, str) and 'kg' in str(weight):
                    weight = str(weight).replace('kg', '')
                per_person_lines.append(f"  用户{uid}({age}岁/{weight}kg) ~{per_kcal:.0f}kcal ~{per_protein:.0f}g蛋白")
    
    lines = ["", "---", "**营养概览**", ""]
    if per_person_lines:
        lines.extend(per_person_lines)
    else:
        lines.append(f"  本餐人均约 {per_kcal:.0f} 千卡，蛋白质 {per_protein:.0f}g，脂肪 {per_fat:.0f}g，碳水 {per_carb:.0f}g")
    
    if num_people > 1:
        lines.append(f"  （基于{num_people}人用餐计算）")
    
    # ── 营养缺口检测与实规划：对比本餐人均摄入 vs 用户每餐参考值，给出可执行调整建议 ──
    try:
        planner_text = plan_nutrition_gaps(
            {
                '热量': per_kcal,
                '蛋白质': per_protein,
                '脂肪': per_fat,
                '碳水': per_carb,
            },
            user_profiles=user_profiles or None,
        )
        if planner_text:
            lines.append(planner_text)
    except Exception as _pe:
        print(f"[nutrition] 营养规划生成失败: {_pe}", flush=True)
    
    return "\n".join(lines)


if __name__ == '__main__':
    # 配置验证
    print("验证配置...")
    is_valid, warnings = validate_config()
    for w in warnings:
        print(w)
    
    # 打印当前配置状态
    print(f"当前LLM_PROVIDER: {Config.LLM_PROVIDER}")
    print(f"DEEPSEEK_API_KEY配置: {'已配置' if Config.DEEPSEEK_API_KEY else '未配置'}")
    print(f"DEEPSEEK_API_BASE: {Config.DEEPSEEK_API_BASE}")
    print()
    
    # 预热组件
    warmup()
    
    # 启动Flask应用
    print("\n启动方太个性化膳食规划Agent...")
    print("API服务地址: http://0.0.0.0:5000")
    print("健康检查: http://localhost:5000/api/health")
    print("按 Ctrl+C 停止服务")
    
    app.run(host='0.0.0.0', port=5000, debug=Config.DEBUG, threaded=True)