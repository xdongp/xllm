# TRAE 监控工具使用说明

## 概述

TRAE监控工具是一个用于监控命令执行结果、管理预设命令和TODO列表的实用工具。它可以帮助您：

- 实时监控命令执行过程和输出
- 预先设置常用命令以便快速执行
- 管理TODO列表以跟踪任务进度
- 自动记录所有操作日志，便于后续查看

## 安装与运行

### 环境要求

- Python 3.6 或更高版本
- 无需额外依赖（使用Python标准库）

### 运行方式

```bash
# 直接运行
python tools/trae_monitor.py [命令] [参数]

# 或添加可执行权限后直接运行
chmod +x tools/trae_monitor.py
./tools/trae_monitor.py [命令] [参数]
```

## 功能说明

### 1. 命令监控

监控并执行命令，实时显示输出并记录到日志。

**用法：**
```bash
python tools/trae_monitor.py monitor <命令> [--cwd <工作目录>]
```

**示例：**
```bash
# 监控Python脚本执行
python tools/trae_monitor.py monitor python tools/benchmark.py --test-type concurrent --requests 10 --concurrency 5 --max-tokens 10

# 在指定目录下执行命令
python tools/trae_monitor.py monitor python xllm_server.py --cwd /Users/dannypan/PycharmProjects/xllm
```

### 2. 进程管理

终止指定名称的进程。

**用法：**
```bash
python tools/trae_monitor.py kill <进程名称或关键字>
```

**示例：**
```bash
# 终止xllm_server.py进程
python tools/trae_monitor.py kill xllm_server.py

# 终止包含benchmark的进程
python tools/trae_monitor.py kill benchmark
```

### 3. 命令预设管理

预设常用命令，方便快速执行。

**用法：**
```bash
# 添加预设
python tools/trae_monitor.py preset add <预设名称> <命令> [--desc <描述>]

# 列出所有预设
python tools/trae_monitor.py preset list

# 删除预设
python tools/trae_monitor.py preset remove <预设名称>

# 执行预设
python tools/trae_monitor.py preset run <预设名称> [--cwd <工作目录>]
```

**示例：**
```bash
# 添加预设
python tools/trae_monitor.py preset add test_benchmark "python tools/benchmark.py --test-type concurrent --requests 10 --concurrency 5 --max-tokens 10" --desc "测试并发请求"

# 执行预设
python tools/trae_monitor.py preset run test_benchmark
```

### 4. TODO列表管理

管理任务列表，跟踪任务进度。

#### 3.1 添加TODO项

**用法：**
```bash
python tools/trae_monitor.py todo add <内容> [--priority <优先级>]
```

**参数：**
- `--priority`: 优先级，可选值：high, medium, low（默认：medium）

**示例：**
```bash
# 添加高优先级任务
python tools/trae_monitor.py todo add "完成基准测试报告" --priority high

# 添加普通任务
python tools/trae_monitor.py todo add "优化模型执行器"
```

#### 3.2 列出TODO项

**用法：**
```bash
python tools/trae_monitor.py todo list [--status <状态>]
```

**参数：**
- `--status`: 状态筛选，可选值：pending, in_progress, completed

**示例：**
```bash
# 列出所有TODO项
python tools/trae_monitor.py todo list

# 只列出未完成的TODO项
python tools/trae_monitor.py todo list --status pending

# 只列出已完成的TODO项
python tools/trae_monitor.py todo list --status completed
```

#### 3.3 更新TODO项状态

**用法：**
```bash
python tools/trae_monitor.py todo update <索引> <状态>
```

**参数：**
- `索引`: TODO项的序号（从1开始）
- `状态`: 新状态，可选值：pending, in_progress, completed

**示例：**
```bash
# 将第1个TODO项标记为进行中
python tools/trae_monitor.py todo update 1 in_progress

# 将第2个TODO项标记为已完成
python tools/trae_monitor.py todo update 2 completed
```

#### 3.4 删除TODO项

**用法：**
```bash
python tools/trae_monitor.py todo remove <索引>
```

**参数：**
- `索引`: TODO项的序号（从1开始）

**示例：**
```bash
# 删除第3个TODO项
python tools/trae_monitor.py todo remove 3
```

## 数据存储

工具会在当前目录下创建一个`.trae_monitor`目录，用于存储：

- `presets.json`: 命令预设数据
- `todos.json`: TODO列表数据
- `monitor.log`: 监控日志
- `command_output_*.log`: 命令执行输出日志（按时间戳命名）

## 示例工作流

### 场景：开发过程中的任务管理和测试监控

1. **添加开发任务**
   ```bash
   python tools/trae_monitor.py todo add "实现异步推理功能" --priority high
   python tools/trae_monitor.py todo add "优化批处理逻辑" --priority medium
   python tools/trae_monitor.py todo add "编写测试文档" --priority low
   ```

2. **列出任务**
   ```bash
   python tools/trae_monitor.py todo list
   ```

3. **添加测试命令预设**
   ```bash
   python tools/trae_monitor.py preset add async_test "python tools/benchmark.py --test-type concurrent --requests 20 --concurrency 10 --max-tokens 50" --desc "异步推理性能测试"
   ```

4. **开始实现功能并更新任务状态**
   ```bash
   python tools/trae_monitor.py todo update 1 in_progress
   # 实现异步推理功能...
   ```

5. **执行测试并监控结果**
   ```bash
   python tools/trae_monitor.py preset run async_test
   ```

6. **标记任务为完成**
   ```bash
   python tools/trae_monitor.py todo update 1 completed
   ```

## 注意事项

1. 工具会自动记录所有操作日志，便于后续查看和分析
2. 命令执行输出会同时显示在控制台和保存到日志文件
3. 预设命令和TODO列表会持久化存储，下次运行时仍可使用
4. 可以在不同目录下运行工具，但数据会保存在当前目录的`.trae_monitor`文件夹中
5. 长时间运行的命令会持续监控，直到命令执行完成

## 帮助信息

使用`--help`参数查看详细帮助信息：

```bash
# 查看整体帮助
python tools/trae_monitor.py --help

# 查看特定命令的帮助
python tools/trae_monitor.py preset --help
python tools/trae_monitor.py todo --help
python tools/trae_monitor.py monitor --help
```
