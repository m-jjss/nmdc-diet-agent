import os
from pathlib import Path

project_root = Path(__file__).parent

try:
    from dotenv import load_dotenv
    load_dotenv(project_root / '.env')
except ImportError:
    pass


class Config:
    """
    方太个性化膳食规划Agent - 配置类
    
    该类集中管理所有配置参数，支持通过环境变量进行配置覆盖。
    默认使用本地模式运行，无需外部API Key即可进行基础功能测试。
    
    配置说明：
    - LLM_PROVIDER: 选择LLM提供商，可选值: local, qwen, ernie, openai, deepseek
    - API Key相关: 各提供商的API密钥和基础URL
    - 索引相关: FAISS索引路径和向量维度
    - 数据文件路径: 菜谱、营养数据库、用户档案的存储路径
    - 检索参数: 最大检索结果数、上下文token限制
    - 生成参数: 温度、最大token数
    """
    
    # LLM提供商选择: local(本地模式)/qwen(阿里云)/ernie(百度)/openai(OpenAI)
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")
    DEBUG = os.getenv("FLASK_DEBUG", "False").lower() in ("1", "true", "yes")
    
    # 阿里云Qwen配置
    QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
    QWEN_API_BASE = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/api/v1")
    QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
    
    # 百度文心一言配置
    ERNIE_API_KEY = os.getenv("ERNIE_API_KEY", "")
    ERNIE_SECRET_KEY = os.getenv("ERNIE_SECRET_KEY", "")
    ERNIE_MODEL = os.getenv("ERNIE_MODEL", "ernie-3.5")
    
    # OpenAI配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # DeepSeek配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    
    # ─── Embedding 向量化配置 ───
    # 模型来源 (默认 sentence-transformers 本地模型，免费、离线)
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
    # 本地模型名 (支持 huggingface 上任何 sentence-transformers 模型)
    # 默认采用中文语义模型 bge-small-zh-v1.5（512维），中文检索精度优于多语言 MiniLM。
    # 若希望覆盖，通过环境变量/`.env` 指定 EMBEDDING_MODEL 与 VECTOR_DIMENSION 即可。
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    # 向量维度 (必须与所选模型输出维度一致)
    #   BAAI/bge-small-zh-v1.5 → 512
    #   paraphrase-multilingual-MiniLM-L12-v2 → 384
    #   distiluse-base-multilingual-cased-v2 → 512
    #   paraphrase-multilingual-mpnet-base-v2 → 768
    VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "512"))
    
    # FAISS向量索引配置
    FAISS_INDEX_PATH = project_root / "idx"

    # sentence-transformers 模型缓存目录（本地离线下发/复用，避免每次都重新下载）
    SBERT_CACHE_DIR = project_root / ".model_cache"
    SBERT_ALLOW_CACHE = True
    
    # 数据文件路径
    RECIPES_JSON_PATH = project_root / "recipes_parsed.json"
    NUTRITION_DB_PATH = project_root / "nutrition_database.json"
    USER_PROFILES_PATH = project_root / "user_profiles_standardized.json"
    
    # 检索参数
    MAX_RETRIEVAL_RESULTS = 10  # 单次检索最大返回结果数
    MAX_CONTEXT_TOKENS = 4096   # 上下文最大token数
    
    # LLM生成参数
    TEMPERATURE = 0.3  # 温度参数，越低越确定性，越高越多样性
    MAX_TOKENS = 2000  # 单次生成最大token数


class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True


class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    MAX_RETRIEVAL_RESULTS = 20


def get_config() -> Config:
    """获取当前环境配置"""
    env = os.getenv("ENV", "development")
    if env == "production":
        return ProductionConfig()
    return DevelopmentConfig()


def validate_config() -> tuple[bool, list[str]]:
    """
    验证配置是否正确
    
    检查API Key是否已配置，给出明确的提示信息。
    
    Returns:
        (是否有效, 警告/错误信息列表)
    """
    config = get_config()
    warnings = []
    
    provider = config.LLM_PROVIDER
    
    if provider == "local":
        warnings.append("使用本地模式运行，无需API Key")
    elif provider == "deepseek":
        if not config.DEEPSEEK_API_KEY:
            warnings.append("[WARN] DEEPSEEK_API_KEY 未配置，请在 .env 文件中设置")
            warnings.append("  格式: DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx")
            warnings.append("  或设置 LLM_PROVIDER=local 使用本地模式")
    elif provider == "qwen":
        if not config.QWEN_API_KEY:
            warnings.append("[WARN] QWEN_API_KEY 未配置，请在 .env 文件中设置")
    elif provider == "ernie":
        if not config.ERNIE_API_KEY or not config.ERNIE_SECRET_KEY:
            warnings.append("[WARN] ERNIE_API_KEY 或 ERNIE_SECRET_KEY 未配置，请在 .env 文件中设置")
    elif provider == "openai":
        if not config.OPENAI_API_KEY:
            warnings.append("[WARN] OPENAI_API_KEY 未配置，请在 .env 文件中设置")
    else:
        warnings.append(f"[WARN] 未知的 LLM_PROVIDER: {provider}")
    
    has_errors = any("未配置" in w for w in warnings)
    
    return (not has_errors, warnings)