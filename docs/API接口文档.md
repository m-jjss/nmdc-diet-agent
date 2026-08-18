# 方太个性化膳食规划 Agent API 接口文档

## 基础信息

| 项目 | 内容 |
|------|------|
| 服务地址 | `http://localhost:5000` |
| 数据格式 | JSON |
| 字符编码 | UTF-8 |
| 健康检查 | `GET /api/health` |

通用成功响应包含 `success: true` 或服务状态字段；通用失败响应如下：

```json
{
  "success": false,
  "error": "错误描述"
}
```

## 健康检查

### `GET /api/health`

用于检查服务是否启动、核心组件是否已加载。

响应示例：

```json
{
  "status": "ok",
  "service": "方太个性化膳食规划Agent",
  "version": "1.0.0",
  "components": {
    "llm_client": true,
    "rag_retriever": true,
    "constraint_engine": true
  }
}
```

调用示例：

```bash
curl http://localhost:5000/api/health
```

## 用户档案

### `GET /api/user/profile`

根据用户 ID 获取健康档案。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `user_id` | string | 是 | 用户 ID |

调用示例：

```bash
curl "http://localhost:5000/api/user/profile?user_id=3"
```

### `POST /api/user/profile`

创建或更新用户档案。

请求示例：

```json
{
  "user_id": "demo_user",
  "allergies": ["海鲜"],
  "diseases": ["高血压"],
  "preferences": {
    "low_spicy": true,
    "light": true
  }
}
```

响应示例：

```json
{
  "success": true,
  "profile": {
    "user_id": "demo_user",
    "allergies": ["海鲜"],
    "diseases": ["高血压"],
    "preferences": {
      "low_spicy": true,
      "light": true
    }
  }
}
```

## 膳食推荐

### `POST /api/recommend`

根据用户输入、用户档案和过滤条件返回推荐菜品。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 用户自然语言需求 |
| `user_id` | string | 否 | 用户 ID；传入后会加载健康档案并过滤约束 |
| `top_k` | int | 否 | 返回数量，默认 5 |
| `filters` | object | 否 | 额外过滤条件 |

`filters` 支持字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `exclude_ingredients` | array | 需要排除的食材 |
| `tags` | array | 希望包含的标签 |
| `max_calories` | int | 最大热量限制 |

请求示例：

```bash
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "query": "推荐一份清淡晚餐",
    "user_id": "3",
    "top_k": 5,
    "filters": {
      "exclude_ingredients": ["辣椒"],
      "tags": ["清淡"]
    }
  }'
```

响应示例：

```json
{
  "success": true,
  "query": "推荐一份清淡晚餐",
  "user_id": "3",
  "recommendations": [
    {
      "name": "示例菜品",
      "ingredients": ["食材1", "食材2"],
      "tags": ["清淡"],
      "score": 0.91,
      "nutrition": {
        "calories": 180,
        "protein": 18,
        "carb": 12,
        "fat": 6,
        "fiber": 2
      }
    }
  ],
  "balance_score": 0.85,
  "recommendation_reason": "根据您的需求，为您推荐以上菜品。",
  "timing": {
    "total": 8.21,
    "search": 2.16,
    "filter": 0.42,
    "nutrition": 0.71,
    "llm": 4.92
  }
}
```

## 菜谱搜索

### `GET /api/search`

根据关键词搜索菜谱。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `q` | string | 是 | 搜索关键词 |
| `top_k` | int | 否 | 返回数量，默认 10 |

调用示例：

```bash
curl "http://localhost:5000/api/search?q=清淡&top_k=5"
```

## 菜谱详情

### `GET /api/recipe/<name>`

根据菜名获取菜谱详情。中文菜名建议做 URL 编码，或直接使用支持 UTF-8 的终端。

调用示例：

```bash
curl "http://localhost:5000/api/recipe/番茄炒蛋"
```

## 全量菜谱列表

### `GET /api/recipes`

返回系统内所有菜谱的简要信息。

调用示例：

```bash
curl http://localhost:5000/api/recipes
```

## 营养计算

### `POST /api/nutrition/calculate`

根据菜谱名估算营养成分。

请求示例：

```bash
curl -X POST http://localhost:5000/api/nutrition/calculate \
  -H "Content-Type: application/json" \
  -d '{"recipe_name":"番茄炒蛋"}'
```

## 约束检查

### `POST /api/constraint/check`

检查某道菜是否符合指定用户的健康约束。

请求示例：

```bash
curl -X POST http://localhost:5000/api/constraint/check \
  -H "Content-Type: application/json" \
  -d '{"recipe_name":"番茄炒蛋","user_id":"3"}'
```

响应示例：

```json
{
  "success": true,
  "recipe_name": "番茄炒蛋",
  "user_id": "3",
  "passed": true,
  "violations": []
}
```

## 多轮对话

### `POST /api/dialog`

支持连续对话、偏好提取和上下文延续。内置 Agent ReAct 循环，通过 Function Calling 自动调用菜谱搜索、约束检查等工具。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `message` | string | 是 | 用户消息 |
| `user_id` | string | 否 | 用户 ID（1-50 为预置档案），同一 ID 复用对话状态 |
| `user_ids` | array | 否 | 多人场景时传入多个用户 ID，合并约束 |
| `existing_user_id` | string | 否 | 继承已有对话的上下文（跨会话） |
| `reset` | bool | 否 | 是否重置对话，默认 `false` |

调用示例：

```bash
# 单人推荐
curl -X POST http://localhost:5000/api/dialog \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo_user","message":"今晚想吃清淡一点"}'

# 追加约束
curl -X POST http://localhost:5000/api/dialog \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo_user","message":"不要辣，也不要海鲜"}'

# 多人场景
curl -X POST http://localhost:5000/api/dialog \
  -H "Content-Type: application/json" \
  -d '{"user_ids":["2","3"],"message":"推荐3人晚餐，一人高血压一人痛风"}'
```

响应示例：

```json
{
  "success": true,
  "user_id": "demo_user",
  "intent": "recommend",
  "response": "为您推荐：...",
  "recommendations": [
    {
      "name": "菜品名",
      "ingredients": ["食材1", "食材2"],
      "cooking_method": "蒸",
      "calories": 180
    }
  ],
  "agentic": true,
  "tool_calls_made": 2,
  "nutrition_summary": "📊 营养概览\n总热量: ~850kcal ...",
  "user_preferences": {
    "allergies": ["海鲜"],
    "preferences": {"low_spicy": true, "light": true}
  },
  "timing": {
    "total": 6.8,
    "search": 0.5,
    "filter": 0.2,
    "nutrition": 0.3,
    "llm": 4.2,
    "verify": 0.1
  },
  "dialog_turns": 4
}
```

### `POST /api/dialog/stream`

流式对话接口（SSE 协议），支持首 Token 延迟追踪。参数同 `POST /api/dialog`。

> **格式说明：** 本接口采用 OpenAI Chat Completion Stream 兼容格式（`data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"..."}}]}`，结束标记 `data: [DONE]`），可在 `delta.content` 中直接累计得到完整回复，首 Token 延迟可由首个 `delta.content` 正确统计。同时保留自定义事件字段（`event` / `data`）便于业务解析，评测方按 OpenAI 规范解析或按自定义事件解析均可。

调用示例：

```bash
curl -X POST http://localhost:5000/api/dialog/stream \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo_user","message":"推荐晚餐"}'
```

响应格式（SSE 事件流，OpenAI 兼容 + 自定义扩展）：

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}],"event":"recommendations","data":{"recommendations":[...],"context_summary":"..."}}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"好的"},"finish_reason":null}],"event":"text","data":{"text":"好的"}}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"，为您推荐..."},"finish_reason":null}],"event":"text","data":{"text":"，为您推荐..."}}
...
data: [DONE]

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"event":"complete","data":{"recommendations":[...],"timing":{"total_ms":1800,"first_token_ms":890}}}
```

| 事件类型 | 说明 | 包含字段 |
|----------|------|----------|
| `recommendations` | 检索结果就绪（首token前发送），`choices[].delta` 仅含 `role` | `recommendations`, `context_summary` |
| `text` | LLM 流式文本片段（OpenAI 兼容 `choices[].delta.content`） | `text`（逐 token 增量） |
| `complete` | 推荐完成，`finish_reason: "stop"` | `recommendations`, `response`, `timing`, `user_preferences` |

## 性能监控

### `GET /api/timing/stats`

获取全局或会话级性能统计数据。数据来源包括 `/api/dialog` 和 `/api/dialog/stream` 的分阶段计时。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `user_id` | string | 否 | 指定会话 ID 查看单会话统计，不传则返回全局统计 |

响应示例：

```json
{
  "success": true,
  "stats": {
    "total_requests": 25,
    "avg_total_ms": 6175.0,
    "avg_llm_ms": 3100.0,
    "p50_total_ms": 6084.0,
    "p95_total_ms": 7675.0,
    "first_token_avg_ms": 1445.0,
    "first_token_p50_ms": 1372.0,
    "ranking": {
      "first_token": {"score": 10, "level": "优秀 (<2s)"},
      "end_to_end": {"score": 10, "level": "优秀 (<8s)"},
      "multi_turn_avg": {"score": 7, "level": "合格 (6~12s)"}
    }
  }
}
```

## 性能说明

本地模式下，检索、约束检查、营养计算主要在本机完成，通常可达到毫秒级响应。远程 LLM 模式会额外受到模型服务、网络延迟和生成长度影响，因此端到端时间应以实际环境测试结果为准。
