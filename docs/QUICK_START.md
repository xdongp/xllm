# Quick Start Guide

This guide will help you get started with xLLM in just a few minutes.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- Git (for cloning the repository)

## Installation

### Option 1: Install from GitHub

```bash
# Clone the repository
git clone https://github.com/yourusername/xllm.git
cd xllm

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install xLLM in development mode
pip install -e .
```

### Option 2: Install from PyPI (when published)

```bash
pip install xllm
```

## Quick Test

### Test with Demo Script

```bash
# Run the Qwen demo
python demo/run_qwen_demo.py --model-path ./model/Qwen/Qwen3-0.6B --pd-separation
```

### Test with Server

```bash
# Start the server
python xllm_server.py --model-path ./model/Qwen/Qwen3-0.6B --port 8000

# In another terminal, test the API
curl http://localhost:8000/health

# Test text generation
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 50}'
```

## Basic Usage

### Interactive Chat

```python
from demo.run_qwen_demo import chat_with_model_pd_separation

# Start interactive chat
chat_with_model_pd_separation(
    model_path="./model/Qwen/Qwen3-0.6B",
    max_length=512,
    temperature=0.7
)
```

### API Usage

```python
import requests

# Generate text
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Tell me a joke",
        "max_tokens": 100,
        "temperature": 0.8
    }
)

print(response.json()["generated_text"])
```

### Streaming Generation

```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Write a short story",
        "max_tokens": 200,
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'), end='', flush=True)
```

## Configuration

### Server Configuration

```bash
python xllm_server.py \
  --model-path ./model/Qwen/Qwen3-0.6B \
  --port 8000 \
  --quantization fp16 \
  --max-batch-size 8 \
  --max-sequence-length 2048
```

### Model Configuration

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "./model/Qwen/Qwen3-0.6B",
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("./model/Qwen/Qwen3-0.6B")
```

## Next Steps

1. **Read the Documentation**: Check out the [docs/](docs/) folder for detailed guides
2. **Explore Examples**: Look at the [examples/](examples/) folder for code examples
3. **Run Tests**: Execute `pytest tests/` to verify your installation
4. **Customize**: Modify the configuration to fit your needs
5. **Contribute**: See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines

## Troubleshooting

### Issue: Model loading fails

**Solution**: Ensure you have downloaded the model files to the correct directory.

```bash
# Download Qwen3 model
python model/download.py
```

### Issue: Out of memory

**Solution**: Use quantization or reduce batch size.

```bash
# Use INT8 quantization
python xllm_server.py --model-path ./model/Qwen/Qwen3-0.6B --quantization int8

# Reduce batch size
python xllm_server.py --model-path ./model/Qwen/Qwen3-0.6B --max-batch-size 4
```

### Issue: Slow inference

**Solution**: Enable optimizations.

```bash
# Use optimized executor
python xllm_server.py \
  --model-path ./model/Qwen/Qwen3-0.6B \
  --use-optimized-executor \
  --quantization fp16
```

### Issue: Import errors

**Solution**: Ensure all dependencies are installed.

```bash
pip install -r requirements.txt
pip install -e .
```

## Getting Help

- 📖 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/yourusername/xllm/issues)
- 💬 [Discussions](https://github.com/yourusername/xllm/discussions)
- 📧 Email: support@xllm.dev

## Performance Tips

1. **Use Quantization**: INT8 or FP16 can significantly reduce memory usage
2. **Batch Requests**: Use continuous batching for better throughput
3. **Cache Results**: Enable KV cache for faster generation
4. **Optimize Parameters**: Adjust temperature, top-k, and top-p for your use case
5. **Use GPU**: If available, use GPU for faster inference

## Examples

### Simple Chatbot

```python
from demo.run_qwen_demo import PromptEngineer, Decoder
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize components
model = AutoModelForCausalLM.from_pretrained("./model/Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("./model/Qwen/Qwen3-0.6B")

prompt_engineer = PromptEngineer(tokenizer)
decoder = Decoder(model, tokenizer)

# Chat loop
history = []
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    
    history.append({"role": "user", "content": user_input})
    prompt = prompt_engineer.format_chat_prompt(history)
    response = decoder.decode(prompt, max_length=512)
    
    print(f"Bot: {response}")
    history.append({"role": "assistant", "content": response})
```

### Text Completion

```python
from demo.run_qwen_demo import Decoder
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./model/Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("./model/Qwen/Qwen3-0.6B")
decoder = Decoder(model, tokenizer)

prompt = "The future of AI is"
completion = decoder.decode(prompt, max_length=100, temperature=0.8)
print(completion)
```

## Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Qwen Model](https://github.com/QwenLM/Qwen)
- [DeepSeek R1](https://github.com/deepseek-ai/DeepSeek-R1)

---

Happy coding with xLLM! 🚀
