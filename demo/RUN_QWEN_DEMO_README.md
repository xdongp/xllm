# run_qwen_demo.py 使用说明

这是一个使用Hugging Face transformers库直接加载Qwen模型并进行交互式问答的演示程序。

## 功能特性

- 直接使用transformers库加载Qwen模型
- 支持交互式对话
- 可自定义生成参数（温度、top-p等）
- 支持清除对话历史
- 支持从命令行参数或环境变量指定模型路径

## 使用方法

### 1. 安装依赖

确保已安装必要的依赖包：

```bash
pip install torch transformers
```

### 2. 准备模型

确保你已经下载了Qwen模型文件，并知道模型文件的路径。

你可以从Hugging Face Model Hub下载模型：
```bash
# 示例：下载Qwen模型
git lfs install
git clone https://huggingface.co/Qwen/Qwen2-7B-Instruct
```

### 3. 运行程序

#### 方法一：通过命令行参数指定模型路径

```bash
python run_qwen_demo.py --model-path /path/to/your/qwen/model
```

#### 方法二：通过环境变量指定模型路径

```bash
export QWEN_MODEL_PATH=/path/to/your/qwen/model
python run_qwen_demo.py
```

### 4. 自定义生成参数

你可以通过命令行参数自定义生成参数：

```bash
python run_qwen_demo.py \
  --model-path /path/to/your/qwen/model \
  --max-new-tokens 1024 \
  --temperature 0.8 \
  --top-p 0.9 \
  --repetition-penalty 1.1
```

所有可用参数：
- `--model-path PATH`: 模型路径
- `--device DEVICE`: 运行设备（cuda/cpu）
- `--max-new-tokens NUM`: 最大生成token数
- `--temperature TEMP`: 采样温度
- `--top-p PROB`: Top-p采样参数
- `--repetition-penalty PENALTY`: 重复惩罚
- `--do-sample`: 是否使用采样

### 5. 交互命令

在程序运行时，你可以使用以下命令：

- `quit` 或 `exit`: 退出程序
- `clear`: 清除对话历史
- `help`: 显示帮助信息
- `params`: 显示当前参数设置

## 示例运行

```bash
$ python run_qwen_demo.py --model-path ./Qwen2-7B-Instruct
Loading model from ./Qwen2-7B-Instruct...
Using device: cuda
Model loaded successfully!

==================================================
欢迎使用Qwen模型交互式问答!
当前生成参数:
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.8
  repetition_penalty: 1.05
  do_sample: True

命令:
  'quit' 或 'exit' - 退出程序
  'clear' - 清除对话历史
  'help' - 显示帮助信息
  'params' - 显示当前参数设置
==================================================

User: 你好，介绍一下你自己
Assistant: 你好！我是通义千问，由通义实验室研发的超大规模语言模型。我可以帮助你回答问题、创作文字，比如写故事、写公文、写邮件、写剧本、逻辑推理、编程等等，还能表达观点，玩游戏等。如果你有任何问题或需要帮助，随时告诉我！

User: quit
Goodbye!
```

## 注意事项

1. 首次加载模型时可能需要较长时间，请耐心等待。
2. 模型需要较大的GPU内存，如果显存不足，可以尝试使用CPU运行（但速度会很慢）。
3. 对于大模型，建议使用支持的GPU以获得最佳性能。