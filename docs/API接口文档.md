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

支持连续对话、偏好提取和上下文延续。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `user_id` | string | 否 | 用户 ID；同一 ID 会复用对话状态 |
| `message` | string | 是 | 用户消息 |
| `reset` | bool | 否 | 是否重置对话，默认 `false` |

调用示例：

```bash
curl -X POST http://localhost:5000/api/dialog \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo_user","message":"今晚想吃清淡一点"}'

curl -X POST http://localhost:5000/api/dialog \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo_user","message":"不要辣，也不要海鲜"}'
```

响应示例：

```json
{
  "success": true,
  "user_id": "demo_user",
  "intent": "recommend",
  "response": "为您推荐：...",
  "recommendations": [],
  "user_preferences": {
    "allergies": ["海鲜"],
    "preferences": {
      "low_spicy": true,
      "light": true
    },
    "dietary_goals": [],
    "dialog_turns": 4
  },
  "dialog_turns": 4
}
```

## 性能说明

本地模式下，检索、约束检查、营养计算主要在本机完成，通常可达到毫秒级响应。远程 LLM 模式会额外受到模型服务、网络延迟和生成长度影响，因此端到端时间应以实际环境测试结果为准。
