#!/usr/bin/env python3
"""
使用transformers库直接加载Qwen模型并进行交互式问答的演示程序
支持PD（Prompt Engineering + Decoding）分离架构
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import sys
import os
import json
from typing import Dict, List, Any, Optional


class PromptEngineer:
    """负责提示工程处理的类"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def format_chat_prompt(self, history: List[Dict[str, str]]) -> str:
        """使用分词器的chat模板格式化对话历史"""
        try:
            # 尝试使用chat模板
            formatted_prompt = self.tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted_prompt
        except Exception:
            # 如果没有chat模板，则手动构建
            prompt = ""
            for turn in history:
                if turn["role"] == "user":
                    prompt += f"User: {turn['content']}\n"
                elif turn["role"] == "assistant":
                    prompt += f"Assistant: {turn['content']}\n"
            prompt += "Assistant:"
            return prompt
    
    def apply_prompt_template(self, user_input: str, template_type: str = "default") -> str:
        """应用特定的提示模板"""
        if template_type == "question_answering":
            return f"请回答以下问题：{user_input}\n答案："
        elif template_type == "instruction_following":
            return f"请按照以下指示操作：{user_input}\n结果："
        elif template_type == "creative_writing":
            return f"请创作一段关于'{user_input}'的内容："
        else:
            return user_input  # 默认情况下直接返回用户输入
    
    def preprocess_input(self, text: str) -> str:
        """预处理输入文本"""
        # 移除多余的空白字符
        text = " ".join(text.split())
        return text


class Decoder:
    """负责模型解码的类"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def decode(self, input_ids, generation_params: Dict[str, Any]) -> str:
        """执行解码生成"""
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=generation_params["max_new_tokens"],
                    temperature=generation_params["temperature"],
                    top_p=generation_params["top_p"],
                    repetition_penalty=generation_params["repetition_penalty"],
                    do_sample=generation_params["do_sample"]
                )
            
            # 解码输出 - 只解码新生成的部分
            response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            return response
        except Exception as e:
            # 重新抛出异常，让调用者处理
            raise


def load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """加载模型和分词器"""
    print(f"Loading model from {model_path}...")
    print(f"Using device: {device}")
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist")
    
    # 加载分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise
    
    # 加载模型
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            trust_remote_code=True
        )
        # 手动将模型移动到指定设备
        model = model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer


def chat_with_model_pd_separation(model, tokenizer, device, generation_params):
    """与模型进行交互式对话（PD分离版本）"""
    print("\n" + "="*50)
    print("欢迎使用Qwen模型交互式问答(PD分离版)!")
    print("当前生成参数:")
    for key, value in generation_params.items():
        print(f"  {key}: {value}")
    print("\n命令:")
    print("  'quit' 或 'exit' - 退出程序")
    print("  'clear' - 清除对话历史")
    print("  'help' - 显示帮助信息")
    print("  'params' - 显示当前参数设置")
    print("  'template <type>' - 切换提示模板 (如: template question_answering)")
    print("="*50 + "\n")
    
    # 初始化PD组件
    prompt_engineer = PromptEngineer(tokenizer)
    decoder = Decoder(model, tokenizer)
    
    # 设置对话历史和当前模板
    history = []
    current_template = "default"
    
    while True:
        # 获取用户输入
        try:
            user_input = input("User: ").strip()
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        
        # 检查特殊命令
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
            
        if user_input.lower() == 'clear':
            history.clear()
            print("对话历史已清除。\n")
            continue
            
        if user_input.lower() == 'help':
            print("命令:")
            print("  'quit' 或 'exit' - 退出程序")
            print("  'clear' - 清除对话历史")
            print("  'help' - 显示帮助信息")
            print("  'params' - 显示当前参数设置")
            print("  'template <type>' - 切换提示模板")
            print("    可用模板: default, question_answering, instruction_following, creative_writing")
            print("  其他输入将作为问题发送给模型\n")
            continue
            
        if user_input.lower() == 'params':
            print("当前生成参数:")
            for key, value in generation_params.items():
                print(f"  {key}: {value}")
            print()
            continue
            
        if user_input.lower().startswith('template '):
            template_type = user_input[9:].strip()
            available_templates = ["default", "question_answering", "instruction_following", "creative_writing"]
            if template_type in available_templates:
                current_template = template_type
                print(f"提示模板已切换为: {current_template}\n")
            else:
                print(f"未知的模板类型。可用模板: {', '.join(available_templates)}\n")
            continue
            
        if not user_input:
            continue
            
        # PD分离处理流程
        # 1. Prompt Engineering阶段
        # 预处理输入
        processed_input = prompt_engineer.preprocess_input(user_input)
        
        # 应用提示模板
        templated_input = prompt_engineer.apply_prompt_template(processed_input, current_template)
        
        # 构建对话历史
        history.append({"role": "user", "content": templated_input})
        
        # 格式化完整的提示
        formatted_prompt = prompt_engineer.format_chat_prompt(history)
        
        # 编码输入
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        # 获取实际的输入张量
        input_ids = inputs["input_ids"]
        
        # 2. Decoding阶段
        try:
            # 执行解码
            response = decoder.decode(input_ids, generation_params)
            
            # 添加到历史记录
            history.append({"role": "assistant", "content": response})
            
            # 打印回复
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"生成回复时出错: {e}")
            # 移除最后添加的用户输入，避免历史记录不一致
            history.pop()
            print()

def chat_with_model_original(model, tokenizer, device, generation_params):
    """原有的交互模式（保持向后兼容）"""
    print("\n" + "="*50)
    print("欢迎使用Qwen模型交互式问答!")
    print("当前生成参数:")
    for key, value in generation_params.items():
        print(f"  {key}: {value}")
    print("\n命令:")
    print("  'quit' 或 'exit' - 退出程序")
    print("  'clear' - 清除对话历史")
    print("  'help' - 显示帮助信息")
    print("  'params' - 显示当前参数设置")
    print("="*50 + "\n")
    
    # 设置对话历史
    history = []
    
    while True:
        # 获取用户输入
        try:
            user_input = input("User: ").strip()
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        
        # 检查特殊命令
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
            
        if user_input.lower() == 'clear':
            history.clear()
            print("对话历史已清除。\n")
            continue
            
        if user_input.lower() == 'help':
            print("命令:")
            print("  'quit' 或 'exit' - 退出程序")
            print("  'clear' - 清除对话历史")
            print("  'help' - 显示帮助信息")
            print("  'params' - 显示当前参数设置")
            print("  其他输入将作为问题发送给模型\n")
            continue
            
        if user_input.lower() == 'params':
            print("当前生成参数:")
            for key, value in generation_params.items():
                print(f"  {key}: {value}")
            print()
            continue
            
        if not user_input:
            continue
            
        # 构建对话历史
        history.append({"role": "user", "content": user_input})
        
        # 使用分词器的chat模板格式化输入
        try:
            # 尝试使用chat模板
            inputs = tokenizer.apply_chat_template(
                history,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
        except Exception:
            # 如果没有chat模板，则直接编码
            text = user_input
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # 生成回复
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=generation_params["max_new_tokens"],
                    temperature=generation_params["temperature"],
                    top_p=generation_params["top_p"],
                    repetition_penalty=generation_params["repetition_penalty"],
                    do_sample=generation_params["do_sample"]
                )
                
            # 解码输出
            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            # 添加到历史记录
            history.append({"role": "assistant", "content": response})
            
            # 打印回复
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"生成回复时出错: {e}")
            # 移除最后添加的用户输入，避免历史记录不一致
            history.pop()
            print()



def main():
    parser = argparse.ArgumentParser(description="Run Qwen model demo with PD separation")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("QWEN_MODEL_PATH", ""),
        help="Path to the Qwen model directory (can also be set via QWEN_MODEL_PATH environment variable)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (cuda/cpu)"
    )
    
    # 生成参数
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="Repetition penalty"
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        default=True,
        help="Whether to use sampling for generation"
    )
    
    # PD分离相关参数
    parser.add_argument(
        "--pd-separation",
        action="store_true",
        help="Enable PD (Prompt Engineering + Decoding) separation mode"
    )
    
    args = parser.parse_args()
    
    # 检查是否提供了模型路径
    if not args.model_path:
        print("错误: 请通过 --model-path 参数指定模型路径或设置 QWEN_MODEL_PATH 环境变量")
        sys.exit(1)
    
    # 构建生成参数字典
    generation_params = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": args.do_sample
    }
    
    try:
        # 加载模型
        model, tokenizer = load_model(args.model_path, args.device)
        
        # 根据参数决定使用哪种交互模式
        if args.pd_separation:
            # 使用PD分离模式
            chat_with_model_pd_separation(model, tokenizer, args.device, generation_params)
        else:
            # 使用原有模式（为了向后兼容）
            chat_with_model_original(model, tokenizer, args.device, generation_params)
        
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        sys.exit(1)



if __name__ == "__main__":
    main()