# xLLM - CPU Optimized Inference Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

xLLM is a CPU-optimized inference engine designed for large language models, supporting models like Qwen3 and DeepSeek R1. It provides efficient text generation capabilities with various sampling strategies and quantization options.

## Features

- Pure CPU execution for broad hardware compatibility
- Support for Qwen3, DeepSeek R1 and other models
- Multiple sampling strategies (Greedy, Temperature, Top-K, Top-P, Beam Search, Contrastive Search)
- Quantization support (INT8, FP16) for reduced memory usage
- Continuous batching for improved throughput
- RESTful API interface for easy integration
- Streaming output support
- Performance monitoring and logging

## Architecture

The xLLM engine follows a layered architecture:

```
┌─────────────────┐
│   HTTP Server   │ ← External Interface
└─────────────────┘
         │
┌─────────────────┐
│ Tokenizer Mgr   │ ← Tokenization & Request Management
└─────────────────┘
         │
┌─────────────────┐
│   Scheduler     │ ← Request Scheduling & Batching
└─────────────────┘
         │
┌─────────────────┐
│ Model Executor  │ ← Model Execution & Sampling
└─────────────────┘
```

## Core Components

### 1. HTTP Service Interface
Handles external API requests and provides RESTful endpoints for text generation, tokenization, and model management.

### 2. Tokenizer Manager
Manages tokenization operations and coordinates requests between the HTTP interface and the scheduler.

### 3. Scheduler
Implements continuous batching and request prioritization to optimize throughput and latency.

### 4. Model Executor
Executes model inference on CPU with support for quantization and various sampling strategies.

## Directory Structure

```
xllm/
├── xllm_server.py     # HTTP服务接口
├── tokenizer_manager.py # Tokenizer管理器
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

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install xLLM:
```bash
pip install -e .
```

## Usage

### Starting the Server

To start the xLLM server with a Qwen3 model:

```bash
python xllm_server.py --model-path ./model/Qwen/Qwen3-0.6B --port 8000 --quantization fp16
```

Parameters:
- `--model-path`: Path to the model directory
- `--port`: Server port (default: 8000)
- `--quantization`: Quantization method (int8, fp16)

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Text Encoding
```bash
curl -X POST http://localhost:8000/encode \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'
```

#### Text Generation
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 50, "temperature": 0.7}'
```

#### Streaming Generation
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me a story", "max_tokens": 100, "stream": true}'
```

### Model Quantization

xLLM supports two quantization methods for CPU optimization:

1. INT8 quantization: Reduces memory usage by approximately 50%
2. FP16 quantization: Balances performance and memory usage

To use quantization, specify the `--quantization` parameter when starting the server.

## Performance Optimization

xLLM implements several optimizations for CPU inference:

- Continuous batching to improve throughput
- Efficient memory management with KV cache
- Radix-based prefix caching to reduce redundant computations
- Optimized sampling algorithms for various strategies
- Quantization support to reduce memory footprint

## Testing

Run unit tests:
```bash
python -m pytest tests/
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Qwen](https://github.com/QwenLM/Qwen) for the amazing language model
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for the powerful library
- All contributors who have helped make xLLM better

## Citation

If you use xLLM in your research, please cite:

```bibtex
@software{xllm2025,
  title={xLLM: CPU Optimized Inference Engine for Large Language Models},
  author={xLLM Contributors},
  year={2025},
  url={https://github.com/yourusername/xllm}
}
```

## Roadmap

- [ ] GPU support with CUDA
- [ ] Multi-model serving
- [ ] Advanced caching strategies
- [ ] Distributed inference
- [ ] Web UI interface
- [ ] Model fine-tuning support

## Support

- 📖 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/yourusername/xllm/issues)
- 💬 [Discussions](https://github.com/yourusername/xllm/discussions)
- 📧 Email: support@xllm.dev

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/xllm&type=Date)](https://star-history.com/#yourusername/xllm&Date)
