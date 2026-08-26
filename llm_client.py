import json
import time
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

try:
    import requests
except ImportError:
    requests = None

from config import Config


# ─────────────────────────────────────────────────────────────
# 共享向量化：所有客户端统一"语义优先 + 哈希兜底"，避免某客户端用随机向量。
# 默认 local 模式若用随机向量，FAISS 检索对所有 query 返回相同结果（agent"变笨"根因）。
# ─────────────────────────────────────────────────────────────
_SBERT_MODEL = None  # 全局单例，避免重复加载

def _init_sbert():
    """懒加载 sentence-transformers 模型（全局单例）。返回模型对象或 None/False。"""
    global _SBERT_MODEL
    if _SBERT_MODEL is not None:
        return _SBERT_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        _SBERT_MODEL = SentenceTransformer(
            Config.EMBEDDING_MODEL, device='cpu',
            cache_folder=str(Config.SBERT_CACHE_DIR),
        )
        print(f"[OK] Embedding模型已加载: {Config.EMBEDDING_MODEL} (dim={Config.VECTOR_DIMENSION})")
    except Exception as e:
        print(f"[WARN] sentence-transformers不可用({e})，使用确定性哈希回退")
        _SBERT_MODEL = False
    return _SBERT_MODEL


def _hash_embed(text: str, dim: int) -> List[float]:
    """确定性字符 n-gram 哈希回退向量（同文同向量，非随机）。"""
    text = text.lower().strip()
    vec = [0.0] * dim
    for i in range(len(text) - 1):
        bg = text[i:i + 2]
        h = int(hashlib.md5(bg.encode()).hexdigest(), 16) % dim
        vec[h] += 1.0
    for i in range(len(text) - 2):
        tg = text[i:i + 3]
        h = int(hashlib.md5(tg.encode()).hexdigest(), 16) % dim
        vec[h] += 0.5
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _semantic_embed(text: str, dim: int) -> Optional[List[float]]:
    """语义向量：sentence-transformers 优先，失败返回 None（调用方回退哈希）。"""
    model = _init_sbert()
    if model and model is not False:
        try:
            vec = model.encode(text, normalize_embeddings=True)
            if len(vec) < dim:
                vec = list(vec) + [0.0] * (dim - len(vec))
            return list(vec[:dim])
        except Exception:
            return None
    return None


def smart_embed(text: str, dim: int) -> List[float]:
    """通用向量化：语义优先 + 哈希兜底。所有客户端统一走这里。"""
    vec = _semantic_embed(text, dim)
    if vec is not None:
        return vec
    return _hash_embed(text, dim)


class LLMClient(ABC):
    """
    LLM客户端抽象基类
    
    定义了所有LLM客户端必须实现的核心接口：
    - chat: 对话生成接口
    - embed: 文本向量化接口
    
    所有具体客户端（Qwen、Ernie、OpenAI、Local）都继承此类。
    """
    
    @abstractmethod
    def chat(self, messages: List[Dict], temperature: float = 0.3, max_tokens: int = 2000) -> str:
        """
        对话生成接口
        
        Args:
            messages: 对话历史消息列表，格式: [{'role': 'user', 'content': '...'}, ...]
            temperature: 生成温度，范围0-1，越低越确定，越高越多样
            max_tokens: 最大生成token数
            
        Returns:
            LLM生成的回复文本
        """
        pass
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        文本向量化接口
        
        Args:
            text: 待向量化的文本
            
        Returns:
            文本的向量表示，维度由配置的VECTOR_DIMENSION决定
        """
        pass


class LocalLLMClient(LLMClient):
    """
    本地LLM客户端（模拟模式）
    
    用于开发测试，不调用远程API，直接返回预定义响应。
    特点：
    - 零延迟，适合快速验证业务逻辑
    - 内置缓存机制，避免重复计算
    - 支持模拟Embedding生成（基于hash的确定性随机向量）
    """
    
    def __init__(self):
        self.cache = {}  # 缓存对话和Embedding结果，格式: {hash_key: result}
    
    def chat(self, messages: List[Dict], temperature: float = 0.3, max_tokens: int = 2000) -> str:
        """
        本地对话生成
        
        优先从缓存获取，缓存未命中时返回预定义响应。
        
        Args:
            messages: 对话消息列表
            temperature: 生成温度（本地模式忽略）
            max_tokens: 最大token数（本地模式忽略）
            
        Returns:
            预定义的回复文本
        """
        # 提取最后一条用户消息作为查询
        query = messages[-1]['content']
        
        # 计算查询的MD5哈希作为缓存键
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 预定义响应映射表
        responses = {
            '今晚吃什么': '为您推荐以下菜品：1. 清蒸鲈鱼；2. 番茄炒蛋；3. 清炒西兰花；4. 小米粥',
            '推荐午餐': '推荐午餐搭配：1. 宫保鸡丁；2. 蒜蓉菠菜；3. 杂粮饭；4. 冬瓜海带汤',
            '不要辣的': '已调整推荐：1. 清蒸鱼；2. 白灼虾；3. 清炒时蔬；4. 南瓜粥',
            '四个人吃饭': '为四位推荐套餐：1. 清蒸鲈鱼；2. 红烧肉；3. 蒜蓉西兰花；4. 番茄蛋汤',
        }
        
        # 获取响应，未匹配则返回默认响应
        response = responses.get(query, '根据您的需求，推荐：清蒸鲈鱼、番茄炒蛋、清炒时蔬、小米粥。')
        
        # 缓存结果
        self.cache[cache_key] = response
        return response
    
    def embed(self, text: str) -> List[float]:
        """
        本地文本向量化

        使用语义模型（sentence-transformers）+ 哈希回退，而非随机向量。
        语义模型不可用时回退到确定性字符 n-gram 哈希，保证检索对同义/相关表达稳定有意义。

        Args:
            text: 待向量化的文本

        Returns:
            Config.VECTOR_DIMENSION 维向量
        """
        return smart_embed(text, Config.VECTOR_DIMENSION)
    
    def chat_with_tools(self, messages: List[Dict], tools: List[Dict] = None,
                        temperature: float = 0.1, max_tokens: int = 500) -> Dict:
        """
        本地模式工具调用
        
        不执行真实工具，直接返回本地聊天回复，保证 local 模式下 Agent 流程可运行。
        
        Returns:
            {'content': 回复文本, 'tool_calls': None, 'finish_reason': 'stop'}
        """
        return {
            'content': self.chat(messages),
            'tool_calls': None,
            'finish_reason': 'stop',
            'reasoning_content': None,
        }
    
    def chat_stream(self, messages: List[Dict], temperature: float = 0.3, max_tokens: int = 300):
        """
        本地模式流式输出
        
        将本地聊天回复按小片段逐步 yield，模拟流式效果。
        """
        response = self.chat(messages)
        for i in range(0, len(response), 5):
            yield response[i:i + 5]
            time.sleep(0.05)


class QwenClient(LLMClient):
    """
    阿里云Qwen大模型客户端
    
    使用阿里云DashScope API进行对话生成和文本向量化。
    当API Key未配置时自动降级为本地模式。
    """
    
    def __init__(self):
        self.api_key = Config.QWEN_API_KEY
        self.api_base = Config.QWEN_API_BASE
        self.model = Config.QWEN_MODEL
        self._session = requests.Session() if requests else None
    
    def chat(self, messages: List[Dict], temperature: float = 0.3, max_tokens: int = 2000) -> str:
        """
        Qwen对话生成
        
        Args:
            messages: 对话消息列表
            temperature: 生成温度
            max_tokens: 最大token数
            
        Returns:
            Qwen模型生成的回复文本
        """
        # API Key未配置时降级为本地模式
        if not self.api_key:
            return LocalLLMClient().chat(messages)
        
        # 构建请求头和负载
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        payload = {
            'model': self.model,
            'input': messages,
            'parameters': {
                'temperature': temperature,
                'max_tokens': max_tokens
            }
        }
        
        try:
            # 发送请求
            resp = self._session.post(
                f'{self.api_base}/chat/completions',
                headers=headers,
                json=payload,
                timeout=30
            )
            # 解析响应
            return resp.json()['output']['choices'][0]['message']['content']
        except Exception:
            # 异常时降级为本地模式
            return LocalLLMClient().chat(messages)
    
    def embed(self, text: str) -> List[float]:
        """
        Qwen文本向量化
        
        Args:
            text: 待向量化的文本
            
        Returns:
            Qwen Embedding模型生成的向量
        """
        if not self.api_key:
            return LocalLLMClient().embed(text)
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        payload = {'model': 'text-embedding-v1', 'input': text}
        
        try:
            resp = self._session.post(
                f'{self.api_base}/embeddings',
                headers=headers,
                json=payload,
                timeout=30
            )
            return resp.json()['output']['embeddings'][0]['embedding']
        except Exception:
            return LocalLLMClient().embed(text)


class ErnieClient(LLMClient):
    """
    百度文心一言大模型客户端
    
    使用百度AI开放平台API进行对话生成和文本向量化。
    支持自动获取和缓存access_token。
    """
    
    def __init__(self):
        self.api_key = Config.ERNIE_API_KEY
        self.secret_key = Config.ERNIE_SECRET_KEY
        self.token = None  # 缓存access_token
        self._session = requests.Session() if requests else None
    
    def _get_token(self):
        """
        获取百度API的access_token
        
        Returns:
            access_token字符串，获取失败返回None
        """
        # 优先使用缓存的token
        if self.token:
            return self.token
        
        # API Key未配置时返回None
        if not self.api_key or not self.secret_key:
            return None
        
        try:
            # 请求token
            resp = self._session.post(
                'https://aip.baidubce.com/oauth/2.0/token',
                params={
                    'grant_type': 'client_credentials',
                    'client_id': self.api_key,
                    'client_secret': self.secret_key
                },
                timeout=10
            )
            self.token = resp.json().get('access_token')
            return self.token
        except Exception:
            return None
    
    def chat(self, messages: List[Dict], temperature: float = 0.3, max_tokens: int = 2000) -> str:
        """
        文心一言对话生成
        
        Args:
            messages: 对话消息列表
            temperature: 生成温度
            max_tokens: 最大token数
            
        Returns:
            文心一言生成的回复文本
        """
        token = self._get_token()
        if not token:
            return LocalLLMClient().chat(messages)
        
        payload = {
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        try:
            resp = self._session.post(
                f'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-3.5?access_token={token}',
                json=payload,
                timeout=30
            )
            return resp.json()['result']
        except Exception:
            return LocalLLMClient().chat(messages)
    
    def embed(self, text: str) -> List[float]:
        """
        文心一言文本向量化
        
        Args:
            text: 待向量化的文本
            
        Returns:
            文心一言Embedding模型生成的向量
        """
        token = self._get_token()
        if not token:
            return LocalLLMClient().embed(text)
        
        payload = {'input': [text]}
        
        try:
            resp = self._session.post(
                f'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token={token}',
                json=payload,
                timeout=30
            )
            return resp.json()['data'][0]['embedding']
        except Exception:
            return LocalLLMClient().embed(text)


class OpenAIClient(LLMClient):
    """
    OpenAI大模型客户端
    
    使用OpenAI API进行对话生成和文本向量化。
    支持自定义API Base（用于代理或本地部署）。
    """
    
    def __init__(self):
        self.api_key = Config.OPENAI_API_KEY
        self.api_base = Config.OPENAI_API_BASE
        self.model = Config.OPENAI_MODEL
        self._session = requests.Session() if requests else None
    
    def chat(self, messages: List[Dict], temperature: float = 0.3, max_tokens: int = 2000) -> str:
        """
        OpenAI对话生成
        
        Args:
            messages: 对话消息列表
            temperature: 生成温度
            max_tokens: 最大token数
            
        Returns:
            OpenAI模型生成的回复文本
        """
        if not self.api_key:
            return LocalLLMClient().chat(messages)
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        payload = {
            'model': self.model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        try:
            resp = self._session.post(
                f'{self.api_base}/chat/completions',
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if resp.status_code == 401 or resp.status_code == 403:
                print(f"[WARN] OpenAI API鉴权失败: status={resp.status_code}, url={self.api_base}")
                print(f"   请检查 .env 文件中的 OPENAI_API_KEY 是否正确")
                raise ValueError(f"OpenAI API鉴权失败: {resp.status_code}")
            
            if resp.status_code != 200:
                error_info = resp.json() if resp.content else {}
                import datetime
                err_msg = f"[WARN] OpenAI API调用失败: status={resp.status_code}, error={error_info}"
                print(err_msg)
                with open('_err_log.txt', 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.datetime.now()} {err_msg}\n")
                raise ValueError(f"OpenAI API调用失败: {resp.status_code}")
            
            return resp.json()['choices'][0]['message']['content']
        except ValueError as e:
            raise e
        except Exception as e:
            print(f"[WARN] DeepSeek API调用异常: {type(e).__name__}: {e}")
            print(f"   降级使用本地模式...")
            return LocalLLMClient().chat(messages)
    
    def embed(self, text: str) -> List[float]:
        """
        OpenAI文本向量化
        
        Args:
            text: 待向量化的文本
            
        Returns:
            OpenAI Embedding模型生成的向量
        """
        if not self.api_key:
            return LocalLLMClient().embed(text)
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        payload = {'model': Config.EMBEDDING_MODEL, 'input': text}
        
        try:
            resp = self._session.post(
                f'{self.api_base}/embeddings',
                headers=headers,
                json=payload,
                timeout=30
            )
            return resp.json()['data'][0]['embedding']
        except Exception:
            return LocalLLMClient().embed(text)


class DeepSeekClient(LLMClient):
    """
    DeepSeek大模型客户端
    
    使用DeepSeek API进行对话生成和文本向量化。
    DeepSeek API与OpenAI API格式兼容，使用相同的请求格式。
    当API Key未配置时自动降级为本地模式。
    
    性能优化：
    - 使用HTTP连接池复用连接
    - 设置合理的超时时间
    - 优化流式响应处理
    """
    
    def __init__(self):
        self.api_key = Config.DEEPSEEK_API_KEY
        self.api_base = Config.DEEPSEEK_API_BASE
        self.model = Config.DEEPSEEK_MODEL
        self.embed_model = Config.EMBEDDING_MODEL
        self.embed_dim = Config.VECTOR_DIMENSION
        self._session = self._create_session()
    
    def _create_session(self):
        """
        创建优化的HTTP会话
        
        Returns:
            requests.Session实例，带有连接池优化
        """
        if not requests:
            return None
        
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=2
        )
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        
        session.headers.update({
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        })
        
        return session
    
    def chat(self, messages: List[Dict], temperature: float = 0.1, max_tokens: int = 100) -> str:
        """
        DeepSeek对话生成
        
        Args:
            messages: 对话消息列表
            temperature: 生成温度（默认0.1，更确定性=更快首Token）
            max_tokens: 最大token数（默认200，推荐理由只需简短回复）
            
        Returns:
            DeepSeek模型生成的回复文本
        """
        if not self.api_key:
            return LocalLLMClient().chat(messages)
        
        payload = {
            'model': self.model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'thinking': {'type': 'disabled'}
        }
        
        try:
            resp = self._session.post(
                f'{self.api_base}/chat/completions',
                json=payload,
                timeout=15
            )
            
            if resp.status_code == 401 or resp.status_code == 403:
                print(f"[WARN] DeepSeek API鉴权失败: status={resp.status_code}, url={self.api_base}")
                raise ValueError(f"DeepSeek API鉴权失败: {resp.status_code}")
            
            if resp.status_code != 200:
                error_info = resp.json() if resp.content else {}
                import datetime
                err_msg = f"[WARN] DeepSeek API调用失败: status={resp.status_code}, error={error_info}"
                print(err_msg)
                with open('_err_log.txt', 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.datetime.now()} {err_msg}\n")
                raise ValueError(f"DeepSeek API调用失败: {resp.status_code}")
            
            return resp.json()['choices'][0]['message']['content']
        except ValueError as e:
            raise e
        except Exception as e:
            print(f"[WARN] DeepSeek API调用异常: {type(e).__name__}: {e}")
            return LocalLLMClient().chat(messages)
    
    def chat_stream(self, messages: List[Dict], temperature: float = 0.3, max_tokens: int = 300):
        """
        DeepSeek流式对话生成（SSE）
        
        Args:
            messages: 对话消息列表
            temperature: 生成温度
            max_tokens: 最大token数（默认300，减少响应时间）
            
        Returns:
            生成器，逐token返回
        """
        if not self.api_key:
            response = LocalLLMClient().chat(messages)
            for i in range(0, len(response), 5):
                yield response[i:i+5]
                time.sleep(0.05)
            return
        
        payload = {
            'model': self.model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': True,
            # 禁用 thinking mode：减少首Token延迟（rethink 会先输出 reasoning_content 再输出正文）
            'thinking': {'type': 'disabled'}
        }
        
        try:
            resp = self._session.post(
                f'{self.api_base}/chat/completions',
                json=payload,
                stream=True,
                timeout=30
            )
            
            if resp.status_code == 401 or resp.status_code == 403:
                print(f"[WARN] DeepSeek API鉴权失败: status={resp.status_code}, url={self.api_base}")
                raise ValueError(f"DeepSeek API鉴权失败: {resp.status_code}")
            
            if resp.status_code != 200:
                error_info = resp.json() if resp.content else {}
                import datetime
                err_msg = f"[WARN] DeepSeek API调用失败: status={resp.status_code}, error={error_info}"
                print(err_msg)
                with open('_err_log.txt', 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.datetime.now()} {err_msg}\n")
                raise ValueError(f"DeepSeek API调用失败: {resp.status_code}")
            
            for line in resp.iter_lines(chunk_size=1024):
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data = line_str[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            content = chunk['choices'][0]['delta'].get('content', '')
                            if content:
                                yield content
                        except:
                            pass
        except ValueError as e:
            raise e
        except Exception as e:
            print(f"[WARN] DeepSeek API调用异常: {type(e).__name__}: {e}")
            response = LocalLLMClient().chat(messages)
            for i in range(0, len(response), 5):
                yield response[i:i+5]
                time.sleep(0.05)

    def chat_with_tools(self, messages: List[Dict], tools: List[Dict],
                        temperature: float = 0.1, max_tokens: int = 500) -> Dict:
        """
        DeepSeek Function Calling / Tool Use

        支持 Agent 模式的工具调用。LLM 可以决定调用哪些工具来完成任务。

        Args:
            messages: 对话消息列表
            tools: 工具定义列表 (OpenAI function calling 格式)
            temperature: 生成温度
            max_tokens: 最大token数

        Returns:
            {
                'content': str | None,
                'tool_calls': list | None,
                'finish_reason': str
            }
        """
        if not self.api_key:
            return {'content': LocalLLMClient().chat(messages), 'tool_calls': None, 'finish_reason': 'stop'}

        payload = {
            'model': self.model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'tools': tools,
            'tool_choice': 'auto',
            'thinking': {'type': 'disabled'}
        }

        try:
            resp = self._session.post(
                f'{self.api_base}/chat/completions',
                json=payload,
                timeout=30
            )

            if resp.status_code == 200:
                data = resp.json()
                choice = data['choices'][0]
                message = choice.get('message', {})
                return {
                    'content': message.get('content'),
                    'tool_calls': message.get('tool_calls'),
                    'finish_reason': choice.get('finish_reason', 'stop'),
                    'reasoning_content': message.get('reasoning_content'),
                }
            else:
                import sys, datetime
                err_msg = f"[ERR] DeepSeek tools call failed: {resp.status_code} | payload_chars={len(str(payload))} | {resp.text[:500]}"
                print(err_msg)
                with open('_err_log.txt', 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.datetime.now()} {err_msg}\n")
                return {
                    'content': f"抱歉，服务暂时不可用（{resp.status_code}）",
                    'tool_calls': None,
                    'finish_reason': 'error'
                }

        except Exception as e:
            print(f"[ERR] DeepSeek tools request exception: {e}")
            return {
                'content': "抱歉，服务暂时不可用",
                'tool_calls': None,
                'finish_reason': 'error'
            }

    def embed(self, text: str) -> List[float]:
        """
        DeepSeek文本向量化

        优先使用sentence-transformers本地模型（免费、高质量），
        不可用时回退到确定性字符n-gram哈希（同文同向量，非随机）。

        Args:
            text: 待向量化的文本

        Returns:
            向量 (维度由 Config.VECTOR_DIMENSION 决定)
        """
        # 统一走 smart_embed：语义优先 + 哈希兜底（与 local 模式一致）
        vec = _semantic_embed(text, self.embed_dim)
        if vec is not None:
            return vec
        return _hash_embed(text, self.embed_dim)


def get_llm_client(provider: str = None) -> LLMClient:
    """
    获取LLM客户端实例
    
    根据配置或指定的提供商返回相应的客户端实例。
    
    Args:
        provider: 提供商名称，可选值: qwen, ernie, openai, deepseek, local
                  未指定时使用配置中的LLM_PROVIDER
    
    Returns:
        LLMClient实例
    """
    provider = provider or Config.LLM_PROVIDER
    
    # 提供商映射表
    clients = {
        'qwen': QwenClient,
        'ernie': ErnieClient,
        'openai': OpenAIClient,
        'deepseek': DeepSeekClient,
        'local': LocalLLMClient
    }
    
    # 返回对应客户端实例，未匹配则使用本地模式
    return clients.get(provider, LocalLLMClient)()


if __name__ == "__main__":
    print("=" * 60)
    print("LLM客户端测试")
    print("=" * 60)
    
    # 创建本地客户端实例
    client = get_llm_client('local')
    
    print("\n1. 测试聊天...")
    response = client.chat([{'role': 'user', 'content': '今晚吃什么'}])
    print(f"   响应: {response}")
    
    print("\n2. 测试Embedding...")
    embedding = client.embed("清蒸鲈鱼")
    print(f"   维度: {len(embedding)}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)