# xLLM 使用示例

这个目录包含了xLLM的各种使用示例，帮助您快速上手和了解xLLM的功能。

## 示例列表

### 1. 基本使用示例 (basic_usage.py)

演示了xLLM的基本功能，包括：
- 健康检查
- 文本编码
- 文本生成
- 流式文本生成

#### 运行方法：

1. 首先启动xLLM服务器：
   ```bash
   xllm serve --model-path /path/to/your/model --port 8080
   ```

2. 然后运行示例：
   ```bash
   python basic_usage.py
   ```

### 2. 性能测试示例 (performance_test.py)

演示了如何测试xLLM的性能，包括：
- 顺序请求测试
- 并发请求测试
- 压力测试

#### 运行方法：

1. 确保xLLM服务器正在运行

2. 运行性能测试：
   ```bash
   python performance_test.py
   ```

## 依赖安装

在运行示例之前，请确保安装了所需的依赖：

```bash
pip install requests
```

## 使用说明

1. 所有示例都假设xLLM服务器运行在 `localhost:8080`
2. 请根据实际情况修改示例中的模型路径
3. 某些示例可能需要根据您的具体环境进行调整

## 注意事项

- 这些示例仅用于演示目的
- 在生产环境中使用时，请根据实际需求进行相应的调整
- 确保服务器有足够的资源来处理请求