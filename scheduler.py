"""
xLLM的调度器 - 管理请求调度和批处理
"""

import asyncio
import heapq
import time
import logging
from typing import List, Dict, Any, Tuple
import torch

# 设置日志
logger = logging.getLogger(__name__)

# 将父目录添加到路径中，以便我们可以从xllm包导入
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xllm.tokenizer_manager import RequestState

# 导入统一的KV缓存实现
from kv_cache import get_global_kv_cache


class Scheduler:
    """管理请求调度和批处理"""
    
    def __init__(self, model_path: str, quantization: str = None):
        from xllm.model_executor import ModelExecutor
        self.model_path = model_path
        self.quantization = quantization
        self.model_executor = ModelExecutor(model_path, quantization=quantization)
        self.request_queue = []  # 传入请求的优先队列
        self.running_requests = {}  # 当前正在处理的请求
        self.completed_requests = {}  # 已完成的请求
        
        # 调度器配置
        self.max_batch_size = 8  # 处理的最大批处理大小
        self.max_context_length = 2048  # 最大上下文长度
        
        # 初始化KV缓存 - 使用统一的KV缓存实现
        self.kv_cache = get_global_kv_cache()
        
        # 调度器任务将在事件循环可用时启动
        self._scheduler_task = None
    
    def start_scheduler_loop(self):
        """当事件循环可用时启动调度器循环"""
        logger.info("Starting scheduler loop")
        if self._scheduler_task is None:
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
    
    def add_request(self, request_state: RequestState):
        """将新请求按优先级添加到调度器队列"""
        # 根据等待时间和请求长度计算优先级
        # 较短的请求和等待时间较长的请求获得更高的优先级
        priority = time.time() + len(request_state.tokenized_prompt) * 0.01
        heapq.heappush(self.request_queue, (priority, request_state))
        logger.debug(f"Added request {request_state.request_id} to queue with priority {priority}")
    
    async def _scheduler_loop(self):
        """主调度器循环"""
        logger.info("Scheduler loop started")
        while True:
            try:
                # 处理请求
                await self._process_requests()
                
                # 短暂休眠以避免忙等待
                await asyncio.sleep(0.001)  # 1ms延迟
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                import traceback
                logger.error(traceback.format_exc())
                await asyncio.sleep(1)  # 出错时等待更长时间
    
    async def _process_requests(self):
        """处理队列中的请求"""
        if not self.request_queue:
            return
        
        logger.debug(f"Processing requests, queue size: {len(self.request_queue)}")
        
        # 形成批处理
        batches = self._form_batches()
        logger.debug(f"Formed {len(batches)} batches")
        
        # 处理每个批处理
        for i, batch in enumerate(batches):
            logger.debug(f"Processing batch {i+1}/{len(batches)} with {len(batch)} requests")
            await self._process_batch(batch)
    
    def _form_batches(self) -> List[List[RequestState]]:
        """从请求队列中形成批处理"""
        logger.debug("Forming batches from request queue")
        batches = []
        current_batch = []
        current_batch_length = 0
        
        # 计算当前运行请求的长度
        running_requests_length = sum(
            len(req.tokenized_prompt) + req.generated_tokens 
            for req in self.running_requests.values()
        )
        
        # 处理队列中的请求直到达到批处理大小限制或上下文长度限制
        while (self.request_queue and 
               len(current_batch) < self.max_batch_size and
               current_batch_length < self.max_context_length):
            
            # 检查队列中的下一个请求
            if not self.request_queue:
                break
                
            _, request = heapq.heappop(self.request_queue)
            
            # 计算添加此请求后的总长度
            request_length = len(request.tokenized_prompt) + request.generated_tokens
            total_length = running_requests_length + current_batch_length + request_length
            
            # 检查是否超出上下文长度限制
            if total_length > self.max_context_length:
                # 将请求放回队列
                priority = time.time() + len(request.tokenized_prompt) * 0.01
                heapq.heappush(self.request_queue, (priority, request))
                break
            
            # 添加请求到当前批处理
            current_batch.append(request)
            current_batch_length += request_length
            logger.debug(f"Request {request.request_id}: prompt_length={len(request.tokenized_prompt)}, generated_tokens={request.generated_tokens}, total_length={request_length}")
        
        # 如果形成了有效的批处理，添加到批次列表中
        if current_batch:
            batches.append(current_batch)
            logger.debug(f"Created final batch with {len(current_batch)} requests")
        
        # 将剩余的请求放回队列
        remaining_requests = len(self.request_queue)
        logger.debug(f"Put {remaining_requests} requests back in queue, final queue size: {remaining_requests}")
        
        logger.debug(f"Processed {len(current_batch)} requests, formed {len(batches)} batches")
        return batches
    
    async def _process_batch(self, batch: List[RequestState]):
        """处理一批请求"""
        logger.debug(f"Processing batch of {len(batch)} requests")
        batch_start_time = time.time()
        
        # 将请求移到运行状态
        for request in batch:
            self.running_requests[request.request_id] = request
            logger.debug(f"Moved request {request.request_id} to running state")
        
        try:
            logger.debug(f"Preparing inputs for batch of {len(batch)} requests")
            # 为模型准备输入
            batch_inputs = self._prepare_batch_inputs(batch)
            logger.debug(f"Prepared batch inputs: {len(batch_inputs['input_ids'])} total tokens")
            
            # 运行模型推理
            logger.debug("Running model inference")
            inference_start_time = time.time()
            batch_outputs = await self.model_executor.forward(batch_inputs)
            inference_duration = time.time() - inference_start_time
            logger.debug(f"Model inference completed in {inference_duration:.2f} seconds")
            
            # 处理输出
            logger.debug("Processing batch outputs")
            self._process_batch_outputs(batch, batch_outputs)
            logger.debug("Batch outputs processed")
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 将批次中的所有请求标记为失败
            for request in batch:
                request.finished = True
                request.error = str(e)
        
        # 清理已完成的请求并将未完成的请求重新排队
        completed_requests = [req for req in batch if req.finished]
        unfinished_requests = [req for req in batch if not req.finished]
        
        logger.debug(f"Batch completed with {len(completed_requests)} finished requests and {len(unfinished_requests)} unfinished requests")
        
        # 处理已完成的请求
        for request in completed_requests:
            if request.request_id in self.running_requests:
                del self.running_requests[request.request_id]
            self.completed_requests[request.request_id] = request
            
        # 将未完成的请求重新排队
        for request in unfinished_requests:
            if request.request_id in self.running_requests:
                del self.running_requests[request.request_id]
            # 以新优先级添加回队列
            priority = time.time()
            heapq.heappush(self.request_queue, (priority, request))
            logger.debug(f"Requeued unfinished request {request.request_id}")
            
        batch_duration = time.time() - batch_start_time
        logger.debug(f"Batch processing completed in {batch_duration:.2f} seconds")
    
    def _prepare_batch_inputs(self, batch: List[RequestState]) -> Dict:
        """准备批处理的输入"""
        logger.debug(f"Preparing inputs for batch of {len(batch)} requests")
        # 为批次中的所有请求组合提示和当前上下文
        all_input_ids = []
        request_positions = []
        sequence_ids = []
        
        current_pos = 0
        for request in batch:
            # 组合提示和生成的令牌
            input_ids = request.tokenized_prompt + request.output_tokens
            all_input_ids.extend(input_ids)
            request_positions.append((current_pos, current_pos + len(input_ids)))
            sequence_ids.append(request.request_id)
            logger.debug(f"Request {request.request_id}: {len(input_ids)} tokens at positions {current_pos}-{current_pos + len(input_ids)}")
            current_pos += len(input_ids)
        
        logger.debug(f"Prepared batch inputs: {len(all_input_ids)} total tokens for {len(batch)} requests")
        return {
            "input_ids": all_input_ids,
            "request_positions": request_positions,
            "batch_size": len(batch),
            "sequence_ids": sequence_ids
        }
    
    def _process_batch_outputs(self, batch: List[RequestState], outputs: Dict):
        """处理批次输出并更新请求状态"""
        logger.debug(f"Processing outputs for batch of {len(batch)} requests")
        logits = outputs["logits"]
        request_positions = outputs["request_positions"]
        logger.debug(f"Logits shape: {logits.shape}, Request positions: {request_positions}")
        
        # 处理批次中的每个请求
        for i, request in enumerate(batch):
            logger.debug(f"Processing output for request {request.request_id}")
            if request.finished:
                logger.debug(f"Request {request.request_id} already finished, skipping")
                continue
                
            start_pos, end_pos = request_positions[i]
            logger.debug(f"Request {request.request_id}: extracting logits from {start_pos} to {end_pos}")
            # 确保不越界
            if start_pos < logits.shape[0] and end_pos <= logits.shape[0]:
                request_logits = logits[start_pos:end_pos]
                logger.debug(f"Request {request.request_id}: extracted logits shape {request_logits.shape}")
                
                # 使用采样器采样下一个令牌
                logger.debug(f"Request {request.request_id}: sampling next token with temperature {request.temperature}")
                next_token = self.model_executor.sampler.sample(
                    request_logits, 
                    request.temperature
                )
                logger.debug(f"Request {request.request_id}: sampled token {next_token}")
                
                # 将令牌添加到输出
                request.output_tokens.append(next_token)
                request.generated_tokens += 1
                logger.debug(f"Request {request.request_id}: added token, total generated: {request.generated_tokens}")
                
                # 检查停止条件
                logger.debug(f"Request {request.request_id}: checking stopping conditions")
                self._check_stopping_conditions(request, next_token)
                logger.debug(f"Request {request.request_id}: finished={request.finished}")
                
                # 额外检查：如果我们已经生成了足够的令牌，则标记为完成
                if request.generated_tokens >= request.max_tokens and not request.finished:
                    request.finished = True
                    logger.debug(f"Request {request.request_id} marked as finished after generating {request.generated_tokens} tokens")
            else:
                logger.error(f"Index out of bounds: start_pos={start_pos}, end_pos={end_pos}, logits_shape={logits.shape}")
                request.finished = True
                request.error = "Index out of bounds"
    
    def _check_stopping_conditions(self, request: RequestState, next_token: int):
        """检查请求是否应该停止"""
        logger.debug(f"Checking stopping conditions for request {request.request_id}")
        # 检查最大令牌数
        if request.generated_tokens >= request.max_tokens:
            request.finished = True
            logger.info(f"Request {request.request_id} finished due to max tokens ({request.generated_tokens} >= {request.max_tokens})")
            return
        
        # 检查停止令牌（简化版）
        # 在实际实现中，这将检查实际的停止令牌
        if next_token == 0:  # Placeholder stop token
            request.finished = True
            logger.info(f"Request {request.request_id} finished due to stop token")
            return

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取KV缓存统计信息"""
        return self.kv_cache.get_cache_stats()

    def get_detailed_cache_stats(self) -> Dict[str, Any]:
        """获取详细的KV缓存统计信息"""
        return self.kv_cache.get_detailed_stats()