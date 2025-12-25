# xLLM 开发指南

## 1. 项目结构

```
xllm/
├── __init__.py
├── server.py          # HTTP服务接口
├── tokenizer_manager.py # Tokenizer管理器
├── scheduler.py        # 调度器
├── model_executor.py   # 模型执行器
├── sampler.py          # 采样器
├── requirements.txt    # 依赖包列表
├── setup.py           # 安装配置
└── docs/              # 文档目录
    ├── design_document.md  # 设计文档
    └── api_reference.md    # API参考
```

## 2. 核心组件详解

### 2.1 HTTP服务接口 (server.py)

HTTP服务接口是xLLM的入口点，负责处理外部请求并与内部组件交互。

**主要功能：**
- 启动FastAPI服务
- 定义API端点
- 处理请求路由
- 管理TokenizerManager实例

### 2.2 Tokenizer管理器 (tokenizer_manager.py)

Tokenizer管理器负责处理文本与token之间的转换，以及管理请求状态。

**主要功能：**
- 文本编码和解码
- 请求状态管理
- 流式响应处理
- 与调度器交互

### 2.3 调度器 (scheduler.py)

调度器是xLLM的核心组件，负责请求调度和批处理。

**主要功能：**
- 请求队列管理
- 批处理策略实现
- 与模型执行器交互
- KV缓存管理

### 2.4 模型执行器 (model_executor.py)

模型执行器负责加载模型并在CPU上执行推理计算。

**主要功能：**
- 模型加载和管理
- CPU优化的前向计算
- 批处理输入处理
- 结果输出处理

### 2.5 采样器 (sampler.py)

采样器负责从模型输出中采样下一个token。

**主要功能：**
- 温度采样
- Top-K采样
- Top-P采样
- 批量采样

## 3. 开发流程

### 3.1 环境设置

1. 克隆项目仓库
2. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```
3. 安装开发包：
   ```bash
   pip install -e .
   ```

### 3.2 代码规范

- 遵循PEP 8代码风格
- 使用类型注解
- 编写单元测试
- 添加适当的文档字符串

### 3.3 测试

- 单元测试位于`tests/`目录下
- 使用pytest作为测试框架
- 运行测试：
  ```bash
  pytest tests/
  ```

## 4. 扩展开发

### 4.1 添加新模型支持

1. 在[model_executor.py](file:///Users/dannypan/PycharmProjects/sglang/xllm/model_executor.py)中实现模型加载逻辑
2. 更新模型配置类
3. 修改API接口以支持新模型

### 4.2 添加新采样策略

1. 在[sampler.py](file:///Users/dannypan/PycharmProjects/sglang/xllm/sampler.py)中实现新的采样方法
2. 更新API参数以支持新策略
3. 在TokenizerManager中集成新策略

### 4.3 优化批处理策略

1. 修改[scheduler.py](file:///Users/dannypan/PycharmProjects/sglang/xllm/scheduler.py)中的批处理逻辑
2. 实现新的调度算法
3. 测试性能改进

## 5. 贡献指南

### 5.1 提交Issue

- 使用清晰的标题描述问题
- 提供详细的复现步骤
- 包含环境信息和错误日志

### 5.2 提交Pull Request

1. Fork项目仓库
2. 创建功能分支
3. 实现功能并添加测试
4. 确保所有测试通过
5. 提交Pull Request

### 5.3 代码审查

- 所有PR都需要至少一名维护者审查
- 遵循项目代码风格
- 确保测试覆盖率

## 6. 性能优化

### 6.1 CPU优化技巧

- 使用NumPy进行向量化计算
- 利用多线程并行处理
- 优化内存访问模式
- 减少不必要的数据复制

### 6.2 内存管理

- 重用预分配的缓冲区
- 及时释放不需要的对象
- 使用内存映射文件处理大模型

### 6.3 批处理优化

- 动态调整批处理大小
- 优化请求打包策略
- 减少批处理间的空闲时间