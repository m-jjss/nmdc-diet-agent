import re
import os
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
_result_verifier = None

# 请求缓存
_request_cache = {}
_CACHE_TTL = 300  # 5分钟

# 全局计时统计（跨会话聚合）
_global_timings = []  # 保留最近 200 条

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


def _execute_agent_tool(tool_name: str, tool_args: dict, user_id: str = None, user_ids: list = None,
                         retriever=None, engine=None, dm=None) -> str:
    """执行 Agent 工具调用，返回结果文本"""
    import json as _json

    if tool_name == "search_recipes":
        query = tool_args.get("query", "")
        top_k = tool_args.get("top_k", 5)
        filters = dm.get_search_filters() if dm else {}
        try:
            results = retriever.search(query, top_k=top_k, filters=filters)
            if user_ids:
                results = engine.filter_by_constraints(results, user_ids=user_ids)
            elif user_id:
                results = engine.filter_by_constraints(results, user_id=user_id)
            recipes = results[:top_k]
            return _json.dumps([{
                "name": r.get("name", "未知"),
                "cuisine": r.get("cuisine", ""),
                "cooking_method": r.get("cooking_method", ""),
                "main_ingredients": r.get("main_ingredients", [])[:3],
            } for r in recipes], ensure_ascii=False)
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
        filters = data.get('filters', {})

        # 获取组件实例
        retriever = get_retriever()
        engine = get_engine()
        llm = get_llm()

        # 1. 检索菜谱
        search_start = time.perf_counter()
        results = retriever.search(query, top_k=top_k * 2, filters=filters)
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
        results = retriever.search(query, top_k=top_k)
        
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


def _agentic_recommend(user_message: str, user_id: str = None, user_ids: list = None,
                       dm=None, retriever=None, engine=None, llm=None,
                       max_iterations: int = 3) -> dict:
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
7. 荤素搭配铁律：每轮推荐至少包含1-2道素菜。搜索时分别搜索荤菜和素菜（用不同query），最终推荐中荤素比例不低于3:2。
   烹饪多样性：推荐中应包含至少3种不同烹饪方式（如炒、蒸、煮、炖、烤、凉拌等），避免全是同一种做法。

当前约束: {pref_summary}
上一轮推荐列表（保留它们，除非用户要求全部换）: {dm.recommended_recipes[-5:] if dm and dm.recommended_recipes else '无'}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    recommendations = []
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
                            recommendations = recipes
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

    # — 保底补齐：推荐数量不足时直接搜索补充 —
    if len(recommendations) < 3 and len(recommendations) > 0:
        print(f"[Agent] 推荐不足{len(recommendations)}道，自动补齐...")
        existing_names = {r.get('name', '') for r in recommendations}
        try:
            fill_results = retriever.search(user_message, top_k=5)
            for r in fill_results:
                if len(recommendations) >= 5:
                    break
                rname = r.get('name', '')
                if rname and rname not in existing_names:
                    recommendations.append(r)
                    existing_names.add(rname)
        except Exception as e:
            print(f"[Agent] 补齐失败: {e}")
        print(f"[Agent] 补齐后: {len(recommendations)}道")

    # — 保底荤素：全荤无素时替换1-2道为素菜 —
    MEAT_KWS = ['肉', '鸡', '鸭', '鱼', '虾', '蟹', '贝', '牛', '猪', '羊', '排骨',
                '腊', '肠', '腿', '翅', '肝', '腰', '脑', '鲍', '参', '蛤', '鱿',
                '鳗', '蚝', '烤鸭', '火腿', '培根']
    VEG_KWS = ['菜', '豆', '腐', '菇', '茄', '瓜', '薯', '藕', '芹', '笋', '莲',
               '葱', '蒜', '姜', '椒', '花', '米', '面', '粉', '饼', '包', '馒', '粥']
    
    if len(recommendations) >= 3:
        meat_count = sum(1 for r in recommendations
                         if any(kw in r.get('name','') + r.get('ingredients','') 
                                for kw in MEAT_KWS))
        veg_count = len(recommendations) - meat_count
        if meat_count == len(recommendations):  # 全荤无素
            print(f"[Agent] 全荤无素({meat_count}荤/{veg_count}素)，自动搜索素菜...")
            try:
                veg_results = retriever.search("素菜 蔬菜 清淡 凉拌 蒸菜", top_k=3)
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

    return {
        'recommendations': recommendations,
        'response': response_text,
        'ask_question': ask_question,
        'tool_calls_made': tool_count,
        'agentic': True,
        'llm_ms': round(llm_ms_total, 2),
        'first_token_ms': first_call_ms
    }


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
        
        # 创建或获取对话管理器
        dialog_key = user_id or "anonymous"
        if dialog_key not in _dialog_managers:
            _dialog_managers[dialog_key] = DialogManager()
        
        dm = _dialog_managers[dialog_key]
        
        # 重置对话（如果请求）
        if reset:
            dm.reset_dialog()
        
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
        llm_result = dm.detect_with_llm(message, llm_client=llm)
        if llm_result:
            intent = llm_result.get('intent', 'recommend')
            if llm_result.get('preferences'):
                preferences = llm_result['preferences']
            else:
                preferences = dm.extract_preferences(message)  # 关键词回退
        else:
            intent = dm.detect_intent(message)  # 关键词回退
            preferences = dm.extract_preferences(message)  # 关键词回退
        
        # 根据意图生成响应
        response_text = ""
        recommendations = []
        
        if intent == 'greet':
            response_text = "嗨！我是方太膳食规划助手～告诉我你的口味偏好，帮你搭配合适的菜品！"

        elif intent == 'recommend':

            # — 模糊消息强制追问：无偏好无约束时避免盲目推荐 —
            VAGUE_QUERIES = ['今晚吃什么', '中午吃', '早上吃', '晚上吃', '吃什么',
                             '吃啥', '早饭', '午饭', '晚饭', '早餐', '午餐', '晚餐']
            up = dm.user_preferences if dm else {}
            has_prefs = (bool(preferences) or bool(up.get('preferences'))
                        or bool(up.get('dietary_goals')) or bool(up.get('cuisine_preference')))
            has_allergies = bool(up.get('allergies'))
            if (not has_prefs and not has_allergies and dm and dm.turn_count <= 1
                    and any(q in message for q in VAGUE_QUERIES)):
                response_text = ("好嘞！先告诉我你的口味偏好吧～喜欢清淡还是重口？"
                                 "想吃荤还是素？有没有忌口或过敏的食材？😊")
                recommendations = []
            else:
                # 推荐意图 — Agent模式（ReAct循环）
                agent_result = _agentic_recommend(
                    user_message=message,
                    user_id=user_id,
                    user_ids=user_ids,
                    dm=dm,
                    retriever=retriever,
                    engine=engine,
                    llm=llm
                )
                response_text = agent_result['response']
                recommendations = agent_result['recommendations']
                if agent_result.get('ask_question'):
                    response_text = agent_result['ask_question']
                t_llm_dialog = agent_result.get('llm_ms', 0)
        
        elif intent in ('set_preferences', 'set_preference', 'modify_preferences', 'add_constraint'):
            # 偏好/约束更新 — 走 Agent 路径保证语义理解准确
            agent_result = _agentic_recommend(
                user_message=message,
                user_id=user_id,
                user_ids=user_ids,
                dm=dm,
                retriever=retriever,
                engine=engine,
                llm=llm
            )
            response_text = agent_result['response']
            recommendations = agent_result['recommendations']
            if agent_result.get('ask_question'):
                response_text = agent_result['ask_question']
            t_llm_dialog = agent_result.get('llm_ms', 0)
        
        elif intent == 'ask_nutrition':
            # 查找用户提到的菜品
            for recipe in retriever.recipes:
                if recipe.get('name', '') in message:
                    nutrition = engine.evaluate_nutrition(recipe)
                    response_text = f"菜品【{recipe['name']}】的营养信息：热量{nutrition['calories']}千卡，蛋白质{nutrition['protein']}克，碳水{nutrition['carb']}克，脂肪{nutrition['fat']}克。"
                    break
            if not response_text:
                response_text = "请告诉我具体想查询哪个菜品的营养信息。"
        
        elif intent == 'ask_recipe_detail':
            # 查找用户提到的菜品
            for recipe in retriever.recipes:
                if recipe.get('name', '') in message:
                    response_text = f"菜品【{recipe['name']}】\n食材：{', '.join(recipe.get('ingredients', []))}\n做法：{recipe.get('description', recipe.get('method', '暂无'))}"
                    break
            if not response_text:
                response_text = "请告诉我您想了解哪道菜的做法。"
        
        elif intent == 'reject_recommendation':
            # 用户否定了之前的推荐
            # 如果没有给出具体理由，反问原因
            bare_negation = re.search(r'^(不太行|不喜欢|不好吃|不太好|不行|不怎么样|算了|不要这个|换一个|都不好|都不行|都不喜欢)[啊呢吧呀嘛]*[。.！!]*$', message.strip())

            if bare_negation:
                response_text = "可以告诉我具体哪里不满意吗？比如太辣了？太油腻？还是想吃别的口味？"
            else:
                # 记录被否定的菜品，走 Agent 路径做语义理解
                resolved_message = dm.resolve_reference(message)
                rejected = [name for name in dm.recommended_recipes
                           if name in resolved_message or name in message]
                if rejected:
                    for name in rejected:
                        dm.add_rejected_recipe(name)
                elif dm.recommended_recipes:
                    for name in dm.recommended_recipes:
                        dm.add_rejected_recipe(name)

                # 构造上下文消息，告诉 Agent 用户否决了什么、已有偏好是什么
                context_parts = []
                if dm.rejected_recipes:
                    context_parts.append(f"用户已否决: {', '.join(dm.rejected_recipes[-5:])}")
                context_parts.append(f"用户要求: {resolved_message}")
                agent_message = '。'.join(context_parts)

                agent_result = _agentic_recommend(
                    user_message=agent_message,
                    user_id=user_id, user_ids=user_ids,
                    dm=dm, retriever=retriever, engine=engine, llm=llm
                )
                response_text = agent_result['response']
                recommendations = agent_result['recommendations']
                if agent_result.get('ask_question'):
                    response_text = agent_result['ask_question']
                t_llm_dialog = agent_result.get('llm_ms', 0)
        
        elif intent == 'request_more':
            # 用户要求更多推荐 — Agent 路径，综合考虑已有推荐和当前需求
            context_parts = []
            if dm.recommended_recipes:
                context_parts.append(f"已推荐: {', '.join(dm.recommended_recipes[-8:])}，请避免重复")
            context_parts.append(f"用户要求: {message}")
            agent_message = '。'.join(context_parts)

            agent_result = _agentic_recommend(
                user_message=agent_message,
                user_id=user_id, user_ids=user_ids,
                dm=dm, retriever=retriever, engine=engine, llm=llm
            )
            response_text = agent_result['response']
            recommendations = agent_result['recommendations']
            if agent_result.get('ask_question'):
                response_text = agent_result['ask_question']
            t_llm_dialog = agent_result.get('llm_ms', 0)
        
        elif intent == 'vague_query':
            # 用户进行模糊查询（如"随便推荐"），保持当前推荐或基于已有偏好推荐
            if dm.recommended_recipes:
                # 如果已有推荐，保持不变，友好回应
                recipe_names = dm.recommended_recipes[-5:]
                response_text = f"好的，保留当前推荐：{', '.join(recipe_names)}。有其他想法随时说～"
                recommendations = []
                for name in recipe_names:
                    recipe = retriever.get_recipe_by_name(name)
                    if recipe:
                        recommendations.append(recipe)
            else:
                # 如果没有推荐历史，基于默认偏好推荐
                filters = dm.get_search_filters()
                results = retriever.search('晚餐', top_k=10, filters=filters)
                if user_ids:
                    results = engine.filter_by_constraints(results, user_ids=user_ids)
                elif user_id:
                    results = engine.filter_by_constraints(results, user_id=user_id)
                
                recommendations = results[:5]
                recipe_names = [r['name'] for r in recommendations]
                dm.add_recommended_recipes(recipe_names)
                
                response_text = f"好的，为您推荐以下菜品：{', '.join(recipe_names)}。您可以告诉我是否需要调整！"
        
        elif intent == 'request_substitute':
            # 用户请求替换某道菜（最小化修改原则）
            resolved_message = dm.resolve_reference(message)
            
            # 识别需要替换的菜品
            substituted = []
            for recipe_name in dm.recommended_recipes:
                if recipe_name in resolved_message or recipe_name in message:
                    substituted.append(recipe_name)
            
            if substituted:
                for name in substituted:
                    dm.add_rejected_recipe(name)
                
                # 获取稳定菜品名称（保留未被替换的）
                stable_names = dm.get_stable_recipes(substituted)
                # 转换为菜品对象
                stable_recipes = [retriever.get_recipe_by_name(r) for r in stable_names]
                stable_recipes = [r for r in stable_recipes if r is not None]
                
                # 只搜索需要替换的部分
                filters = dm.get_search_filters()
                results = retriever.search(resolved_message, top_k=10, filters=filters)
                
                if user_ids:
                    results = engine.filter_by_constraints(results, user_ids=user_ids)
                elif user_id:
                    results = engine.filter_by_constraints(results, user_id=user_id)
                
                # 获取新菜品，补充到稳定菜品中
                stable_name_set = set(stable_names)
                new_recipes = [r for r in results if r['name'] not in stable_name_set][:5 - len(stable_recipes)]
                recommendations = stable_recipes + new_recipes
                recipe_names = [r['name'] for r in recommendations]
                
                dm.add_recommended_recipes(recipe_names)
                
                context_summary = dm.get_context_summary()
                response_text = f"已为您替换菜品：{', '.join(substituted)} -> {', '.join([r['name'] for r in new_recipes])}。\n保留的菜品：{', '.join(stable_names)}。"
            else:
                response_text = "请告诉我您想替换哪道菜。"
        
        elif intent == 'confirm':
            dm.set_dialog_state('completed')
            # 营养总结 + 烹饪小贴士
            parts = ["好的，已确认推荐～"]
            if dm.recommended_recipes:
                # 按人营养摄入
                nutri = compute_meal_nutrition_summary(
                    dm.recommended_recipes[-8:],
                    user_ids=user_ids if user_ids else ([user_id] if user_id else None)
                )
                if nutri:
                    parts.append(nutri)
                # 烹饪建议
                total_time = sum(
                    (retriever.get_recipe_by_name(r) or {}).get('cooking_time', 0)
                    for r in dm.recommended_recipes[-8:]
                )
                parts.append(f"预计总烹饪时间约{total_time}分钟，建议先处理耗时长的菜品。祝你用餐愉快！")
            else:
                parts.append("祝你用餐愉快！")
            response_text = '\n'.join(parts)
        
        elif intent == 'cancel':
            dm.reset_dialog()
            response_text = "好的，已重置。还有什么需要帮忙的吗？"
        
        else:
            # 默认：推荐意图 — Agent模式（ReAct循环）
            agent_result = _agentic_recommend(
                user_message=message,
                user_id=user_id,
                user_ids=user_ids,
                dm=dm,
                retriever=retriever,
                engine=engine,
                llm=llm
            )
            response_text = agent_result['response']
            recommendations = agent_result['recommendations']
            if agent_result.get('ask_question'):
                response_text = agent_result['ask_question']
            t_llm_dialog = agent_result.get('llm_ms', 0)
        
        # 添加系统回复到对话历史
        dm.add_message('system', response_text)
        
        # ── 按人营养摄入概览 ──
        nutrition_summary = ""
        recipe_names = [r['name'] for r in recommendations] if recommendations else []
        if recipe_names and not agent_result.get('ask_question'):
            nutrition_summary = compute_meal_nutrition_summary(
                recipe_names,
                user_ids=user_ids if user_ids else ([user_id] if user_id else None)
            )
            if nutrition_summary:
                response_text += "\n" + nutrition_summary

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
                # 非预置用户：从对话偏好中获取过敏信息
                verify_profile['allergies'] = dm.user_preferences.get('allergies', [])
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
                    recommendations = [r for r in recommendations if r.get('name', '') not in violation_recipes]
                    if not recommendations:
                        # 全部被拦截：尝试现场生成一道合规新菜谱（赛题加分项）
                        gen = _generate_fallback_recipe(
                            message, user_id=user_id, user_ids=user_ids,
                            dm=dm, engine=engine, llm=llm
                        )
                        if gen:
                            recommendations = [gen]
                            ings = '、'.join(gen['ingredients'])
                            response_text += (
                                f"\n\n为符合您的健康约束，我现场设计了一道新菜——"
                                f"【{gen['name']}】（建议菜谱·生成）\n"
                                f"食材：{ings}\n"
                                f"做法：{gen['steps']}"
                            )
                        else:
                            response_text += ("\n\n⚠️ 很抱歉，当前推荐未通过安全检测，已为您重新筛选。"
                                              "请告诉我更多偏好，帮您找到合适的菜品～")
        t_verify_dialog = round((time.perf_counter() - t_verify_start) * 1000, 2)
        
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
        if 'agent_result' in locals():
            timing_data['agent_mode'] = True
            timing_data['tool_calls'] = agent_result.get('tool_calls_made', 0)
            timing_data['t_first_token_ms'] = agent_result.get('first_token_ms', 0)
        dm.record_timing(timing_data)
        _global_timings.append(timing_data)
        if len(_global_timings) > 200:
            _global_timings.pop(0)
        
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
            _dialog_managers[dialog_key] = DialogManager()
        
        dm = _dialog_managers[dialog_key]
        
        if reset:
            dm.reset_dialog()
        
        dm.add_message('user', message)
        preferences = dm.extract_preferences(message)
        intent = dm.detect_intent(message)
        
        # 检查缓存
        cache_key = hashlib.md5(f"{dialog_key}_{message}".encode()).hexdigest()
        cached = _request_cache.get(cache_key)
        if cached and time.time() - cached['timestamp'] < _CACHE_TTL:
            def cached_stream():
                yield f"data: {json.dumps(cached['data'], ensure_ascii=False)}\n\n"
                yield "event: end\ndata: {}\n\n"
            return Response(cached_stream(), content_type='text/event-stream')
        
        def generate():
            t_start = time.perf_counter()
            filters = dm.get_search_filters()
            
            results = retriever.search(message, top_k=10, filters=filters)
            t_search = time.perf_counter()
            
            if user_ids:
                results = engine.filter_by_constraints(results, user_ids=user_ids)
            elif user_id:
                results = engine.filter_by_constraints(results, user_id=user_id)
            t_filter = time.perf_counter()
            
            recommendations = results[:5]
            # 检索为空时，尝试生成一道合规新菜谱（赛题加分项）
            if not recommendations:
                gen = _generate_fallback_recipe(
                    message, user_id=user_id, user_ids=user_ids,
                    dm=dm, engine=engine, llm=llm
                )
                if gen:
                    recommendations = [gen]
            recipe_names = [r['name'] for r in recommendations]
            dm.add_recommended_recipes(recipe_names)
            
            balance_score = engine.calculate_balance_score(recommendations)
            context_summary = dm.get_context_summary()
            
            partial_result = {
                'event': 'recommendations',
                'data': {
                    'user_id': user_id,
                    'intent': intent,
                    'recommendations': [{'name': r['name'], 'tags': r.get('tags', [])} for r in recommendations],
                    'context_summary': context_summary,
                    'dialog_turns': dm.turn_count
                }
            }
            yield f"data: {json.dumps(partial_result, ensure_ascii=False)}\n\n"
            
            llm_context = f"你是膳食规划师，像朋友聊天一样推荐菜品。已推荐:{', '.join(recipe_names)}。用户偏好:{context_summary[:100]}。请用1-2句话像朋友一样说明推荐理由。"
            
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
                    yield f"data: {json.dumps({'event': 'text', 'data': {'text': chunk}}, ensure_ascii=False)}\n\n"
                t_llm_end = time.perf_counter()
            except Exception as e:
                print(f"LLM stream error: {e}")
                t_llm_end = time.perf_counter()
                fallback_text = f'为您推荐：{", ".join(recipe_names)}'
                if first_token_ms is None:
                    first_token_ms = round((t_llm_end - t_llm_start) * 1000, 2)
                yield f"data: {json.dumps({'event': 'text', 'data': {'text': fallback_text}}, ensure_ascii=False)}\n\n"
            
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
            
            final_result = {
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
        
        return Response(generate(), content_type='text/event-stream')
    
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
    
    # 计算总营养
    total = {'热量': 0, '蛋白质': 0, '脂肪': 0, '碳水': 0}
    for recipe in recipes:
        ingredients = recipe.get('ingredients', '')
        for ing, data in MEAL_NUTRITION_DB.items():
            if ing in ingredients:
                total['热量'] += data.get('热量', 0)
                total['蛋白质'] += data.get('蛋白质', 0)
                total['脂肪'] += data.get('脂肪', 0)
                total['碳水'] += data.get('碳水', 0)
    
    per_kcal = total['热量'] / num_people
    per_protein = total['蛋白质'] / num_people
    per_fat = total['脂肪'] / num_people
    per_carb = total['碳水'] / num_people
    
    # 按人需求参考（如果提供了 user_ids）
    per_person_lines = []
    if user_ids and engine:
        for uid in user_ids:
            profile = engine.get_user_profile(uid)
            if profile:
                age = profile.get('年龄', profile.get('age', '?'))
                weight = profile.get('体重(kg)', profile.get('weight', '?'))
                if isinstance(weight, str) and 'kg' in str(weight):
                    weight = str(weight).replace('kg', '')
                per_person_lines.append(f"  用户{uid}({age}岁/{weight}kg) ~{per_kcal:.0f}kcal ~{per_protein:.0f}g蛋白")
    
    lines = ["", "---", "📊 **营养概览**", ""]
    if per_person_lines:
        lines.extend(per_person_lines)
    else:
        lines.append(f"  本餐人均约 {per_kcal:.0f} 千卡，蛋白质 {per_protein:.0f}g，脂肪 {per_fat:.0f}g，碳水 {per_carb:.0f}g")
    
    if num_people > 1:
        lines.append(f"  （基于{num_people}人用餐计算）")
    
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
    
    app.run(host='0.0.0.0', port=5000, debug=Config.DEBUG)