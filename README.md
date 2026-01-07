# xLLM - 高效大语言模型推理服务器

xLLM是一个高效的大语言模型推理服务器，支持批处理、量化和流式响应。

## 功能特性

- ✅ 高效批处理调度
- ✅ 多种量化支持 (int8, fp16)
- ✅ 流式和非流式响应
- ✅ 健康检查端点
- ✅ 文本编码/解码功能

## 系统要求

- Python 3.8+
- PyTorch
- Transformers
- FastAPI
- uvicorn

## 快速开始

### 1. 启动服务器

```bash
# 使用默认设置启动服务器
./start_server.sh

# 或者使用自定参数
./start_server.sh --model-path ./model/Qwen/Qwen3-0.6B --port 8001 --quantization int8
```

### 2. API端点

#### 健康检查
```
GET /health
```

#### 文本生成
```
POST /generate
```

请求参数:
- `prompt`: 输入文本 (string)
- `messages`: 消息数组 (array, 可选)
- `max_tokens`: 最大生成token数 (int, 默认100)
- `temperature`: 温度参数 (float, 默认0.7)
- `stream`: 是否流式输出 (bool, 默认false)

示例请求:
```bash
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "你好，世界！",
    "max_tokens": 50,
    "temperature": 0.7,
    "stream": false
  }'
```

#### 文本编码
```
POST /encode
```

请求参数:
- `text`: 要编码的文本 (string)

### 3. 使用客户端

```bash
python xllm_client.py
```

## 配置参数

- `--model-path`: 模型路径 (必需)
- `--port`: 服务器端口 (默认8001)
- `--quantization`: 量化方式 (int8, fp16, 默认int8)
- `--max-batch-size`: 最大批处理大小 (默认4)
- `--max-context-length`: 最大上下文长度 (默认1024)
- `--debug`: 启用调试模式

## 停止服务器

```bash
./stop_server.sh
```

## 架构组件

- **Scheduler**: 请求调度和批处理管理
- **TokenizerManager**: 令牌化和流式生成管理
- **ModelExecutor**: 模型推理执行器
- **KVCache**: 键值缓存管理