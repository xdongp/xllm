# xLLM API 参考文档

## 1. 概述

xLLM提供了一套RESTful API接口，用于与推理引擎进行交互。所有API端点都遵循标准的HTTP协议，支持JSON格式的数据交换。

## 2. API端点

### 2.1 文本生成接口

#### POST /generate

生成文本内容。

**请求参数：**

| 参数名 | 类型 | 必需 | 描述 |
|--------|------|------|------|
| prompt | string | 是 | 输入提示文本 |
| temperature | float | 否 | 采样温度，默认为0.7 |
| max_tokens | integer | 否 | 最大生成token数，默认为100 |
| stream | boolean | 否 | 是否启用流式输出，默认为false |
| stop | string/array | 否 | 停止词 |

**请求示例：**
```json
{
  "prompt": "你好，世界！",
  "temperature": 0.7,
  "max_tokens": 100,
  "stream": true
}
```

**响应示例（非流式）：**
```json
{
  "request_id": "uuid-string",
  "prompt": "你好，世界！",
  "generated_text": "这是一个美好的一天。",
  "finish_reason": "length"
}
```

**响应示例（流式）：**
```
data: {"token": "这"}

data: {"token": "是"}

data: {"token": "一个"}

data: [DONE]
```

### 2.2 文本编码接口

#### POST /encode

将文本编码为token ID序列。

**请求参数：**

| 参数名 | 类型 | 必需 | 描述 |
|--------|------|------|------|
| text | string | 是 | 要编码的文本 |

**请求示例：**
```json
{
  "text": "你好，世界！"
}
```

**响应示例：**
```json
{
  "token_ids": [12345, 67890, 54321],
  "text": "你好，世界！"
}
```

### 2.3 健康检查接口

#### GET /health

检查服务健康状态。

**响应示例：**
```json
{
  "status": "healthy"
}
```

### 2.4 模型信息接口

#### GET /models

获取支持的模型列表。

**响应示例：**
```json
{
  "models": [
    {
      "id": "qwen3-0.6b",
      "name": "Qwen3 0.6B",
      "description": "Qwen3 model with 0.6 billion parameters"
    },
    {
      "id": "deepseek-r1-0.6b",
      "name": "DeepSeek R1 0.6B",
      "description": "DeepSeek R1 model with 0.6 billion parameters"
    }
  ]
}
```

## 3. 错误处理

所有API错误都遵循标准的HTTP状态码和错误消息格式。

**错误响应格式：**
```json
{
  "detail": "错误描述信息"
}
```

**常见错误码：**
- 400 Bad Request: 请求参数错误
- 404 Not Found: 请求的资源不存在
- 500 Internal Server Error: 服务器内部错误
- 503 Service Unavailable: 服务暂时不可用

## 4. 流式响应

对于启用流式输出的请求，服务器将通过Server-Sent Events (SSE)协议返回结果。客户端需要正确处理SSE格式的数据流。

**SSE格式说明：**
- 每个数据块以`data: `开头
- 数据块以两个换行符`\n\n`结尾
- 流结束时发送`data: [DONE]\n\n`