# 方太个性化膳食规划系统

基于大语言模型的智能膳食推荐系统，综合用户健康档案与菜谱知识库，实现从单人单餐到多人多约束宴请场景的膳食规划。

## 功能特性

- **多约束推理** — 过敏原、慢性疾病、特殊人群禁忌自动过滤，对话中实时识别健康约束

- **多人场景合并** — 并集硬约束 + 交集软约束策略

- **RAG 检索增强** — FAISS 向量索引 + 中文语义检索（bge-small-zh）+ 交叉编码器精排

- **Agent 多轮交互** — ReAct 模式，支持约束追加、局部替换、方案否定、食材替换咨询、道别等交互

- **推荐多样性与数量理解** — 排除最近已推菜避免重复，识别"只吃四道"等数量表达，单一食材自动荤素搭配均衡

- **流式响应** — SSE 协议，首 Token 延迟 < 2s

- **营养计算** — Mifflin-St Jeor 公式，7 维度膳食平衡度评分

- **性能优化** — ReAct 迭代收敛 + 泛化查询缓存，端到端延迟显著降低

- **自纠错机制** — 每轮对推荐做质量诊断，用户换措辞重复追问时识别并主动承认上一轮问题，同时更换搭配，实现"发现问题—自我纠正"闭环

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key（复制 .env.example 为 .env 并填写）
cp .env.example .env

# 3. 启动服务
python app.py

# 4. 运行评测
python eval.py
```

## Docker 部署

```bash
docker-compose up -d
```

## 文档

| 文档                          | 说明             |
| --------------------------- | -------------- |
| [技术方案文档](docs/技术方案文档.md)    | 系统架构、数据设计、检索策略 |
| [API 接口文档](docs/API接口文档.md) | 接口定义与调用示例      |
| [部署文档](docs/部署文档.md)        | 环境配置与部署步骤      |
| [作品信息摘要](docs/作品信息摘要.md)    | 项目概览与创新点       |

## 技术栈

Python · Flask · DeepSeek · FAISS · sentence-transformers · Docker

## 项目结构

```
├── app.py                  # Flask 主应用，API 路由 + Agent ReAct 循环
├── chat.py                 # 终端对话界面（python chat.py 1）
├── llm_client.py           # LLM 客户端封装（DeepSeek + Function Calling）
├── rag_retriever.py        # RAG 检索器（FAISS 索引 + hash 回退）
├── constraint_engine.py    # 约束引擎（过敏/疾病/营养计算）
├── dialog_enhancer.py      # 对话管理器（意图识别/偏好/疾病提取）
├── result_verifier.py      # 结果验证器（幻觉/过敏/约束检测）
├── gradio_app.py           # Gradio Web 前端界面
├── eval.py                 # 竞赛自动评分脚本（基础/复杂/多轮/性能4维度，含25组对话用例）
├── test_daily_dialog.py    # 日常场景多轮对话压力测试（10 组场景、39 轮）
├── config.py               # 配置中心（环境变量 + 默认值）
├── requirements.txt        # Python 依赖
├── .env.example            # 环境变量模板
├── docker-compose.yml      # Docker Compose 编排
├── Dockerfile              # Docker 镜像构建
├── docs/                   # 项目文档（技术方案/API/部署/摘要/演示脚本）
├── recipes_parsed.json     # 菜谱库（约 2000 道）
├── user_profiles_standardized.json  # 用户健康档案（50 份）
├── nutrition_database.json           # 营养数据库（185 种食材）
├── ingredient_synonym_map.json       # 食材同义词映射
└── 对话用例.json           # 大赛提供的 25 组对话用例
```

