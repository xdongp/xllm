# xLLM - CPU优化的推理引擎

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

xLLM是一个专为大型语言模型设计的CPU优化推理引擎，支持Qwen3和DeepSeek R1等模型。它提供了高效的文本生成能力，具有多种采样策略和量化选项。

## 特性

- 纯CPU执行，硬件兼容性强
- 支持Qwen3、DeepSeek R1等模型
- 多种采样策略（贪心、温度、Top-K、Top-P、束搜索、对比搜索）
- 量化支持（INT8、FP16）以减少内存使用
- 连续批处理以提高吞吐量
- RESTful API接口便于集成
- 流式输出支持
- 性能监控和日志记录

## 架构

xLLM引擎采用分层架构设计：

```
┌─────────────────┐
│   HTTP Server   │ ← 外部接口
└─────────────────┘
         │
┌─────────────────┐
│ Tokenizer Mgr   │ ← 分词和请求管理
└─────────────────┘
         │
┌─────────────────┐
│   Scheduler     │ ← 请求调度和批处理
└─────────────────┘
         │
┌─────────────────┐
│ Model Executor  │ ← 模型执行和采样
└─────────────────┘
```

## 核心组件

### 1. HTTP服务接口
处理外部API请求，提供文本生成、分词和模型管理的RESTful端点。

### 2. 分词器管理器
管理分词操作，协调HTTP接口和调度器之间的请求。

### 3. 调度器
实现连续批处理和请求优先级调度，以优化吞吐量和延迟。

### 4. 模型执行器
在CPU上执行模型推理，支持量化和各种采样策略。

## 目录结构

```
xllm/
├── xllm_server.py     # HTTP服务接口
├── tokenizer_manager.py # 分词器管理器
├── scheduler.py        # 调度器
├── model_executor.py   # 模型执行器
├── sampler.py          # 采样器
├── requirements.txt    # 依赖项
├── setup.py           # 安装脚本
├── model/             # 模型文件目录
│   └── Qwen/
│       └── Qwen3-0.6B/
└── tests/             # 测试用例
    ├── test_model_executor.py
    ├── test_sampler.py
    └── test_tokenizer_manager.py
```

## 安装

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 安装xLLM：
```bash
pip install -e .
```

## 使用方法

### 启动服务器

使用Qwen3模型启动xLLM服务器：

```bash
python xllm_server.py --model-path ./model/Qwen/Qwen3-0.6B --port 8000 --quantization fp16
```

参数：
- `--model-path`：模型目录路径
- `--port`：服务器端口（默认：8000）
- `--quantization`：量化方法（int8, fp16）

### API端点

#### 健康检查
```bash
curl http://localhost:8000/health
```

#### 文本编码
```bash
curl -X POST http://localhost:8000/encode \
  -H "Content-Type: application/json" \
  -d '{"text": "你好，世界！"}'
```

#### 文本生成
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好，你好吗？", "max_tokens": 50, "temperature": 0.7}'
```

#### 流式生成
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "给我讲个故事", "max_tokens": 100, "stream": true}'
```

### 模型量化

xLLM支持两种量化方法以优化CPU推理：

1. INT8量化：内存使用量减少约50%
2. FP16量化：平衡性能和内存使用

要使用量化，请在启动服务器时指定`--quantization`参数。

## 性能优化

xLLM实现了多项CPU推理优化：

- 连续批处理以提高吞吐量
- 使用KV缓存的高效内存管理
- 基于基数的前缀缓存以减少冗余计算
- 针对各种策略优化的采样算法
- 量化支持以减少内存占用

## 测试

运行单元测试：
```bash
python -m pytest tests/
```

## 基准测试

使用基准测试工具评估性能：
```bash
python3 tools/benchmark.py
```

## 贡献

我们欢迎贡献！请参阅[CONTRIBUTING.md](CONTRIBUTING.md)了解指南。

## 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件。

## 致谢

- [Qwen](https://github.com/QwenLM/Qwen) 提供了优秀的语言模型
- [Hugging Face Transformers](https://github.com/huggingface/transformers) 提供了强大的库
- 所有帮助改进xLLM的贡献者

## 引用

如果您在研究中使用xLLM，请引用：

```bibtex
@software{xllm2025,
  title={xLLM: CPU Optimized Inference Engine for Large Language Models},
  author={xLLM Contributors},
  year={2025},
  url={https://github.com/yourusername/xllm}
}
```

## 路线图

- [ ] GPU支持（CUDA）
- [ ] 多模型服务
- [ ] 高级缓存策略
- [ ] 分布式推理
- [ ] Web UI界面
- [ ] 模型微调支持

## 支持

- 📖 [文档](docs/)
- 🐛 [问题追踪](https://github.com/yourusername/xllm/issues)
- 💬 [讨论](https://github.com/yourusername/xllm/discussions)
- 📧 邮箱: support@xllm.dev

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/xllm&type=Date)](https://star-history.com/#yourusername/xllm&Date)
