# 方太个性化膳食规划系统

基于大语言模型的智能膳食推荐系统，综合用户健康档案与菜谱知识库，实现从单人单餐到多人多约束宴请场景的膳食规划。

## 功能特性

- **多约束推理** — 过敏原、慢性疾病、特殊人群禁忌自动过滤
- **多人场景合并** — 并集硬约束 + 交集软约束策略
- **RAG 检索增强** — FAISS 向量索引 + 菜谱语义检索
- **Agent 多轮交互** — ReAct 模式，支持约束追加、局部替换、方案否定等 6 种交互
- **流式响应** — SSE 协议，首 Token 延迟 < 2s
- **营养计算** — Mifflin-St Jeor 公式，7 维度膳食平衡度评分

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

| 文档 | 说明 |
|------|------|
| [技术方案文档](docs/技术方案文档.md) | 系统架构、数据设计、检索策略 |
| [API 接口文档](docs/API接口文档.md) | 接口定义与调用示例 |
| [部署文档](docs/部署文档.md) | 环境配置与部署步骤 |
| [作品信息摘要](docs/作品信息摘要.md) | 项目概览与创新点 |

## 技术栈

Python · Flask · DeepSeek · FAISS · sentence-transformers · Docker

## 项目结构

```
├── app.py                  # Flask 主应用，API 路由 + Agent ReAct 循环
├── llm_client.py           # LLM 客户端封装（多提供商支持）
├── rag_retriever.py        # RAG 检索器（FAISS 索引）
├── constraint_engine.py    # 约束引擎（过敏/疾病/营养计算）
├── dialog_enhancer.py      # 对话管理器（意图识别/偏好提取）
├── result_verifier.py      # 结果验证器（幻觉/过敏/约束检测）
├── gradio_app.py           # Gradio 前端界面
├── eval.py                 # 竞赛自动评分脚本
├── config.py               # 配置中心
├── docs/                   # 项目文档
└── Dockerfile              # Docker 镜像构建
```
