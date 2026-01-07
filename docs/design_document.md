# xLLM 推理引擎设计文档

## 1. 概述

xLLM是一个基于纯CPU实现的高性能推理引擎，灵感来源于SGLang的设计理念。它采用分层架构设计，包含前端API接口和后端高性能运行时系统，专门针对Qwen3、DeepSeek-R1等大语言模型进行优化。

### 1.1 设计目标

- 纯CPU实现，无需GPU依赖
- 支持主流大语言模型（Qwen3、DeepSeek-R1等）
- 高性能推理能力，通过批处理和缓存优化提升效率
- 易于部署和使用，提供RESTful API接口
- 可扩展架构，便于后续功能增强

### 1.2 核心特性

- 连续批处理处理机制
- 基于Radix树的KV缓存管理
- 流式输出支持
- 多模型并发支持
- 请求优先级调度

## 2. 架构设计

### 2.1 整体架构

xLLM采用与SGLang类似的分层架构设计：

```
┌─────────────────────────────────────┐
│           前端API层                 │
├─────────────────────────────────────┤
│         HTTP服务接口                │
├─────────────────────────────────────┤
│        Tokenizer管理器              │
├─────────────────────────────────────┤
│          调度器                    │
├─────────────────────────────────────┤
│       模型执行器 (CPU优化)          │
└─────────────────────────────────────┘
```

### 2.2 组件说明

#### 2.2.1 HTTP服务接口
- 提供RESTful API接口用于接收推理请求
- 处理健康检查、模型信息查询等管理接口
- 将请求转发给Tokenizer管理器

#### 2.2.2 Tokenizer管理器
- 负责文本tokenization和detokenization
- 管理请求状态和生命周期
- 处理流式输出和异步响应

#### 2.2.3 调度器
- 实现请求调度和批处理策略
- 管理RadixCache KV缓存
- 协调模型执行器的工作负载

#### 2.2.4 模型执行器
- 加载和管理模型权重
- 执行模型前向计算（纯CPU实现）
- 优化矩阵运算和注意力机制计算

## 3. 核心模块设计

### 3.1 HTTP服务接口

#### 3.1.1 主要端点
- `POST /generate` - 文本生成接口
- `POST /encode` - 文本编码接口
- `GET /health` - 健康检查接口
- `GET /models` - 模型信息查询接口

#### 3.1.2 请求格式示例
```json
{
  "prompt": "你好，世界！",
  "temperature": 0.7,
  "max_tokens": 100,
  "stream": true
}
```

### 3.2 Tokenizer管理器

#### 3.2.1 功能特性
- 支持多种tokenizer（SentencePiece、HuggingFace Tokenizers等）
- 异步请求处理
- 请求ID生成和管理
- 流式响应处理

#### 3.2.2 数据结构
```python
class RequestState:
    def __init__(self, request_id: str, prompt: str, tokenized_prompt: List[int], 
                 temperature: float = 0.7, max_tokens: int = 100, 
                 stream: bool = False, stop: Union[str, list, None] = None):
        self.request_id = request_id
        self.prompt = prompt
        self.tokenized_prompt = tokenized_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.stop = stop
        
        # Generation state
        self.output_tokens = []
        self.finished = False
        self.generated_tokens = 0
        self.stop_strings = []
```

### 3.3 调度器

#### 3.3.1 批处理策略
- 连续批处理：动态组合多个请求形成批次
- 优先级调度：根据请求长度和等待时间分配优先级
- KV缓存复用：利用RadixCache提高缓存命中率

#### 3.3.2 RadixCache设计
- 基于Radix树的数据结构管理KV缓存
- 支持前缀匹配和缓存共享
- 自动缓存淘汰机制

### 3.4 模型执行器

#### 3.4.1 CPU优化技术
- 使用OpenMP进行多线程并行计算
- 利用Intel MKL库优化矩阵运算
- 实现高效的注意力机制计算
- 模型量化支持（INT8、FP16）
- 设置合适的CPU线程数以充分利用多核资源
- **异步模型推理**：使用线程池执行模型推理，释放事件循环以处理更多并发请求

#### 3.4.2 模型支持
- Qwen3系列模型（0.6B测试版本）
- DeepSeek-R1系列模型
- 模型插件化加载机制

## 4. 性能优化策略

### 4.1 计算优化
- 多线程并行处理
- SIMD指令集优化
- 内存访问模式优化
- 计算图优化
- 减少调度器循环中的不必要延迟
- **异步模型推理**：将模型推理移到线程池执行，避免阻塞事件循环，提高并发处理能力

### 4.2 内存优化
- KV缓存管理优化
- 动态内存分配策略
- 内存池技术减少分配开销
- 对象复用减少GC压力

### 4.3 批处理优化
- 动态调整批处理大小
- 优化请求打包策略
- 减少批处理间的空闲时间

### 4.4 缓存优化策略

#### 4.4.1 KV缓存优化
- **LRU淘汰算法改进**：基于访问时间和访问频率的混合淘汰策略
- **缓存预热**：对热门前缀进行预加载
- **分层缓存**：实现多级缓存结构，提高命中率
- **缓存压缩**：对KV缓存进行量化压缩以节省内存

#### 4.4.2 缓存命中率监控
- 实时监控缓存命中率
- 缓存条目访问统计
- 缓存效率分析工具
- 动态调整缓存策略

#### 4.4.3 缓存优化效果
- 通过缓存复用减少重复计算
- 提高推理吞吐量
- 降低平均响应时间