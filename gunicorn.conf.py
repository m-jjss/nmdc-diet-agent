"""Gunicorn 生产配置

- 单进程多线程：多轮对话状态（进程内 DialogManager）共享，且支持并发
- preload_app + on_starting 预热：fork worker 前构建 FAISS 索引、加载约束引擎和
  LLM 客户端，worker 继承预热状态，避免评测首次请求冷启动超时
"""

# 服务绑定
bind = "0.0.0.0:5000"
workers = 1          # 多进程会丢失多轮对话状态，必须为 1
threads = 8          # 多线程并发处理
worker_class = "gthread"
timeout = 120        # 单请求超时（秒）
graceful_timeout = 30
preload_app = True   # fork 前加载应用并预热，worker 继承预热状态

accesslog = "-"
errorlog = "-"


def on_starting(server):
    """fork worker 前预热组件（FAISS 索引、约束引擎、LLM 客户端）"""
    import app as app_module
    app_module.warmup()
