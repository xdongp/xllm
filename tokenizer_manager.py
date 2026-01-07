"""
Tokenizer Manager - 管理分词器和模型推理
"""
import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, AsyncGenerator
import torch
from transformers import AutoTokenizer

from model_executor_optimized import OptimizedModelExecutor, create_optimized_executor
from kv_cache import get_global_kv_cache  # 导入正确的KV缓存函数


logger = logging.getLogger(__name__)


def top_p_filtering(logits, top_p=0.9, filter_value=-float('inf')):
    """Top-p (nucleus) sampling filtering"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Fill the removed indices with filter_value
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[0, indices_to_remove] = filter_value
    
    return logits


@dataclass
class RequestState:
    """请求状态"""
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    stop_tokens: List[str]
    tokenized_prompt: List[int] = None
    start_time: float = None
    generated_text: str = ""
    generated_tokens: List[int] = None
    is_finished: bool = False
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.generated_tokens is None:
            self.generated_tokens = []
        if self.tokenized_prompt is None:
            self.tokenized_prompt = []
        if self.start_time is None:
            self.start_time = time.time()


class TokenizerManager:
    """分词器管理器 - 管理分词器和模型推理"""
    
    def __init__(self, model_path: str, model_executor: Optional[OptimizedModelExecutor] = None):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 初始化stop tokens
        self.stop_tokens = [self.tokenizer.eos_token_id]
        
        # 如果没有提供模型执行器，则创建一个
        if model_executor is None:
            self.model_executor = create_optimized_executor(
                model_path=model_path,
                quantization="int8",
                enable_compile=True
            )
        else:
            self.model_executor = model_executor
        
        # 初始化KV缓存
        self.kv_cache = get_global_kv_cache()
        
        logger.info("✅ TokenizerManager initialized with real model executor")
        
    def encode(self, text: str) -> List[int]:
        """编码文本为token IDs"""
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def decode(self, token_ids: List[int]) -> str:
        """解码token IDs为文本"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    async def generate_stream(self, request_id: str, prompt: str, max_tokens: int = 100, 
                             temperature: float = 0.7, top_p: float = 0.9) -> AsyncGenerator[Dict[str, Any], None]:
        """流式生成 - 逐token返回生成的文本"""
        try:
            logger.info(f"开始流式生成请求: {request_id}")
            logger.debug(f"请求参数: prompt={repr(prompt)}, max_tokens={max_tokens}, temperature={temperature}")
            
            # 编码输入
            input_ids = self.encode(prompt)
            
            # 将输入转换为tensor
            input_tensor = torch.tensor([input_ids], device=self.model_executor.device, dtype=torch.long)
            
            # 跟踪已生成的token列表和之前的文本
            generated_tokens = []
            previous_text = ""
            
            # 调用模型执行器的生成方法，获取生成器
            token_generator = self.model_executor.generate_stream(
                input_ids=input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True
            )
            
            # 逐token生成和返回
            for token in token_generator:
                new_token = token.item()
                
                # 检查是否遇到停止token
                if new_token in self.stop_tokens:
                    logger.debug(f"遇到停止token: {new_token}, 结束生成")
                    break
                
                # 添加到已生成token列表
                generated_tokens.append(new_token)
                
                # 解码所有生成的token
                all_text = self.decode(generated_tokens)
                
                # 计算新生成的文本部分
                # 注意：这里需要处理Unicode字符，确保正确获取增量部分
                import unicodedata
                
                # 确保使用正确的编码方式处理字符串
                all_text = unicodedata.normalize('NFC', all_text)
                previous_text = unicodedata.normalize('NFC', previous_text)
                
                # 查找新文本的起始位置
                start_pos = 0
                for i, char in enumerate(previous_text):
                    if i >= len(all_text) or char != all_text[i]:
                        break
                    start_pos += 1
                
                new_text = all_text[start_pos:]
                
                # 只在有新文本时返回
                if new_text:
                    yield {"text": new_text, "finished": False}
                    # 更新之前的文本
                    previous_text = all_text
                
                # 检查token数量是否已达上限
                if len(generated_tokens) >= max_tokens:
                    logger.debug(f"已达最大token数量: {max_tokens}")
                    break
            
            logger.info(f"✅ 流式生成完成: {request_id}, 共生成 {len(generated_tokens)} 个token")
            
            # 返回最终完成状态
            yield {"text": "", "finished": True}
            
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            yield {"text": f"生成失败: {e}", "finished": True}
    
    async def generate(self, request_id: str, prompt: str, max_tokens: int = 100, 
                      temperature: float = 0.7, top_p: float = 0.9) -> str:
        """非流式生成 - 返回完整的生成文本（支持异步并行）"""
        import asyncio
        try:
            # 编码输入
            input_ids = self.encode(prompt)
            
            # 将输入转换为tensor
            input_tensor = torch.tensor([input_ids], device=self.model_executor.device, dtype=torch.long)
            
            # 使用线程池执行模型推理，支持并行处理
            loop = asyncio.get_event_loop()
            new_tokens = await loop.run_in_executor(
                self.model_executor.executor,
                self.model_executor.generate,
                input_tensor,
                max_tokens,
                temperature,
                True
            )
            
            # 检查是否生成了任何token
            if new_tokens.shape[1] == 0:
                generated_part = ""
            else:
                # 转换为列表
                new_token_list = new_tokens[0].tolist()
                
                # 拼接完整的token列表
                full_token_list = input_ids + new_token_list
                
                # 解码生成的文本
                generated_text = self.decode(full_token_list)
                
                # 从完整生成文本中提取新生成的部分（去掉输入部分）
                if generated_text.startswith(prompt):
                    generated_part = generated_text[len(prompt):]
                else:
                    # 如果生成文本不以prompt开头，说明可能出现了问题，返回完整生成文本
                    generated_part = generated_text
                    logger.warning(f"Generated text does not start with prompt for request {request_id}")
                    logger.debug(f"Prompt: {repr(prompt)}")
                    logger.debug(f"Generated text: {repr(generated_text)}")
            
            logger.info(f"✅ Generation completed for request {request_id}")
            return generated_part
        
        except Exception as e:
            logger.error(f"非流式生成失败: {e}")
            return f"生成失败: {e}"