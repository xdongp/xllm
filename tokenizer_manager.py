"""
xLLM的分词器管理器
"""
import asyncio
import uuid
import json
import logging
from typing import List, Optional, Union
import numpy as np
import sys
import os

# 将父目录添加到路径中，以便我们可以从xllm包导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer


# 设置日志
logger = logging.getLogger(__name__)


class RequestState:
    """表示生成请求的状态"""
    
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
        
        # 生成状态
        self.output_tokens = []
        self.finished = False
        self.generated_tokens = 0
        self.stop_strings = []
        self.error = None
        
        # 处理停止字符串
        if isinstance(stop, str):
            self.stop_strings = [stop]
        elif isinstance(stop, list):
            self.stop_strings = stop
        elif stop is not None:
            self.stop_strings = [str(stop)]


class TokenizerManager:
    """管理分词和请求处理"""
    
    def __init__(self, model_path: str, quantization: str = None):
        self.model_path = model_path
        self.quantization = quantization
        from xllm.scheduler import Scheduler
        self.scheduler = Scheduler(model_path, quantization=quantization)
        from xllm.sampler import Sampler
        self.sampler = Sampler()
        self.request_states = {}
        
        # 初始化分词器
        self.tokenizer = None
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self):
        """初始化分词器"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            logger.info(f"Successfully loaded tokenizer for model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer for model {self.model_path}: {e}")
            logger.info("Using fallback tokenizer")
            self.tokenizer = None
    
    def encode(self, text: str) -> List[int]:
        """将文本编码为token ID"""
        if self.tokenizer:
            return self.tokenizer.encode(text)
        else:
            # 备用实现
            return [ord(c) for c in text][:100]  # 简单的字符到整数编码演示
    
    def decode(self, token_ids: List[int]) -> str:
        """将token ID解码为文本"""
        if self.tokenizer:
            # 处理特殊标记
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        else:
            # 备用实现 - 改进的解码方法
            try:
                # 尝试更广泛的字符范围，包括常见的Unicode字符
                decoded_chars = []
                for token_id in token_ids:
                    try:
                        # 尝试直接转换为字符
                        char = chr(token_id)
                        # 只添加可打印的字符或空格
                        if char.isprintable() or char.isspace():
                            decoded_chars.append(char)
                        else:
                            # 对于不可打印字符，尝试替换为占位符
                            decoded_chars.append(f"[{token_id}]")
                    except ValueError:
                        # 如果转换失败，添加占位符
                        decoded_chars.append(f"[{token_id}]")
                return ''.join(decoded_chars)
            except Exception as e:
                # 如果所有方法都失败，返回原始token ID列表作为字符串
                logger.warning(f"Failed to decode token IDs: {e}")
                return f"<DECODE_ERROR: {str(token_ids)[:100]}>"  # 限制长度以防过长
    
    async def generate(self, prompt: str, temperature: float = 0.7, 
                       max_tokens: int = 100, stream: bool = False,
                       stop: Union[str, list, None] = None) -> dict:
        """根据提示生成文本"""
        # 创建请求ID
        request_id = str(uuid.uuid4())
        logger.info(f"Starting generation for request {request_id}: prompt='{prompt[:50]}...', max_tokens={max_tokens}")
        
        # 对提示进行分词
        tokenized_prompt = self.encode(prompt)
        logger.info(f"Tokenized prompt: {len(tokenized_prompt)} tokens")
        
        # 创建请求状态
        request_state = RequestState(
            request_id=request_id,
            prompt=prompt,
            tokenized_prompt=tokenized_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stop=stop
        )
        
        # 存储请求状态
        self.request_states[request_id] = request_state
        logger.info(f"Stored request state for {request_id}")
        
        # 将请求添加到调度器
        self.scheduler.add_request(request_state)
        logger.info(f"Added request {request_id} to scheduler")
        
        # 如果尚未启动，则启动调度器循环
        if hasattr(self.scheduler, 'start_scheduler_loop'):
            logger.info("Starting scheduler loop")
            self.scheduler.start_scheduler_loop()
        
        # 如果是流式传输，返回流式响应
        if stream:
            logger.info(f"Returning streaming response for request {request_id}")
            return self._stream_response(request_id)
        else:
            # 等待完成
            logger.info(f"Waiting for completion of request {request_id}")
            wait_start_time = asyncio.get_event_loop().time()
            while not request_state.finished:
                await asyncio.sleep(0.01)
                # 检查超时或错误
                if request_state.error:
                    logger.error(f"Request {request_id} failed with error: {request_state.error}")
                    break
                    
            wait_duration = asyncio.get_event_loop().time() - wait_start_time
            logger.info(f"Request {request_id} completed in {wait_duration:.2f} seconds")
            
            # 返回最终结果
            if request_state.error:
                return {
                    "request_id": request_id,
                    "prompt": prompt,
                    "error": request_state.error,
                    "generated_text": "",
                    "finish_reason": "error"
                }
            
            decoded_text = self.decode(request_state.output_tokens)
            logger.info(f"Decoded {len(request_state.output_tokens)} output tokens to text: '{decoded_text[:50]}...'")
            return {
                "request_id": request_id,
                "prompt": prompt,
                "generated_text": decoded_text,
                "finish_reason": "length" if request_state.generated_tokens >= max_tokens else "stop"
            }
    
    async def generate_stream(self, prompt: str, temperature: float = 0.7,
                              max_tokens: int = 100, stop: Union[str, list, None] = None):
        """根据提示生成文本并流式输出"""
        # 创建请求ID
        request_id = str(uuid.uuid4())
        logger.info(f"Starting streaming generation for request {request_id}: prompt='{prompt[:50]}...', max_tokens={max_tokens}")
        
        # 对提示进行分词
        tokenized_prompt = self.encode(prompt)
        logger.info(f"Tokenized prompt: {len(tokenized_prompt)} tokens")
        
        # 创建请求状态
        request_state = RequestState(
            request_id=request_id,
            prompt=prompt,
            tokenized_prompt=tokenized_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stop=stop
        )
        
        # 存储请求状态
        self.request_states[request_id] = request_state
        logger.info(f"Stored request state for {request_id}")
        
        # 将请求添加到调度器
        self.scheduler.add_request(request_state)
        logger.info(f"Added request {request_id} to scheduler")
        
        # 如果尚未启动，则启动调度器循环
        if hasattr(self.scheduler, 'start_scheduler_loop'):
            logger.info("Starting scheduler loop")
            self.scheduler.start_scheduler_loop()
        
        # 流式响应
        logger.info(f"Starting streaming response for request {request_id}")
        async for chunk in self._stream_response_async(request_id):
            yield chunk
    
    def _stream_response(self, request_id: str):
        """创建流式响应"""
        # 这是一个简化的流式实现
        # 在实际实现中，这将是一个异步生成器
        # 它会在生成令牌时产生令牌
        logger.info(f"Creating streaming response for request {request_id}")
        async def stream_generator():
            async for chunk in self._stream_response_async(request_id):
                yield chunk
        return stream_generator()
    
    async def _stream_response_async(self, request_id: str):
        """异步生成流式响应"""
        logger.info(f"Starting async streaming response for request {request_id}")
        request_state = self.request_states.get(request_id)
        if not request_state:
            logger.error(f"Request {request_id} not found")
            yield f'data: {{"error": "Request not found", "request_id": "{request_id}"}}\n\n'
            return
        
        last_sent_tokens = 0
        while not request_state.finished:
            # Send new tokens
            if len(request_state.output_tokens) > last_sent_tokens:
                new_tokens = request_state.output_tokens[last_sent_tokens:]
                decoded_text = self.decode(new_tokens)
                logger.debug(f"Streaming {len(new_tokens)} new tokens for request {request_id}: '{decoded_text}'")
                
                # 为每个新token创建SSE事件
                for i, token_id in enumerate(new_tokens):
                    token_text = self.decode([token_id])
                    token_data = {
                        "id": f"{request_id}-{last_sent_tokens + i}",
                        "request_id": request_id,
                        "token": token_text,
                        "generated_text": self.decode(request_state.output_tokens[:last_sent_tokens + i + 1])
                    }
                    yield f'data: {json.dumps(token_data, ensure_ascii=False)}\n\n'
                
                last_sent_tokens = len(request_state.output_tokens)
            
            await asyncio.sleep(0.01)
        
        # Send final completion message
        final_text = self.decode(request_state.output_tokens)
        completion_data = {
            "id": f"{request_id}-final",
            "request_id": request_id,
            "generated_text": final_text,
            "finish_reason": "length" if request_state.generated_tokens >= request_state.max_tokens else "stop",
            "done": True
        }
        logger.info(f"Streaming completed for request {request_id}")
        yield f'data: {json.dumps(completion_data, ensure_ascii=False)}\n\n'