"""
xLLM的调度器 - 管理请求调度和批处理
"""

import asyncio
import heapq
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
import torch

# 设置日志
logger = logging.getLogger(__name__)

# 将父目录添加到路径中，以便我们可以从xllm包导入
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xllm.tokenizer_manager import RequestState
from xllm.model_executor import ModelExecutor

# 导入统一的KV缓存实现
from kv_cache import get_global_kv_cache


class Scheduler:
    """xLLM调度器 - 管理请求调度和批处理"""
    
    def __init__(self, model_path: str, quantization: str = None, max_batch_size: int = 8, max_context_length: int = 2048):
        """
        初始化调度器
        
        Args:
            model_path: 模型路径
            quantization: 量化方法
            max_batch_size: 最大批处理大小
            max_context_length: 最大上下文长度
        """
        self.model_executor = ModelExecutor(model_path, quantization=quantization)
        self.kv_cache = get_global_kv_cache()
        
        # 请求队列 - 使用优先队列，优先处理等待时间长或请求长度短的请求
        self.request_queue = []
        self.count = 0
        
        # 运行中的请求 - 使用字典便于快速查找和更新
        self.running_requests = {}
        
        # 已完成的请求 - 存储完成的结果供查询
        self.completed_requests = {}
        
        # 调度器配置 - 优化内存使用
        self.max_batch_size = max_batch_size  # 降低默认批处理大小以减少内存使用
        self.max_context_length = max_context_length  # 降低默认上下文长度以减少内存使用
        
        # 调度器状态
        self.running = False
        self._scheduler_task = None  # 添加缺失的_scheduler_task属性
        
        logger.info(f"Scheduler initialized with max_batch_size={self.max_batch_size}, "
                   f"max_context_length={self.max_context_length}")
    
    def start_scheduler_loop(self):
        """当事件循环可用时启动调度器循环"""
        logger.info("Starting scheduler loop")
        self.running = True
        if self._scheduler_task is None:
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
    
    def stop_scheduler_loop(self):
        """停止调度器循环"""
        logger.info("Stopping scheduler loop")
        self.running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            self._scheduler_task = None
    
    def add_request(self, request_state: RequestState):
        """添加请求到调度队列"""
        # 计算优先级 - 对于短序列请求，减少长度权重以降低等待时间
        wait_time = time.time() - request_state.start_time
        prompt_length = len(request_state.tokenized_prompt)
        
        # 对于短序列请求，使用不同的优先级计算策略
        if prompt_length <= 50:
            # 短序列请求：更注重等待时间，减少长度惩罚
            priority = -(wait_time + min(prompt_length / 10.0, 5.0))  # 减少长度权重
        else:
            # 长序列请求：保持原有的平衡
            priority = -(wait_time + min(prompt_length / 50.0, 20.0))
        
        heapq.heappush(self.request_queue, (priority, request_state))
        logger.debug(f"Added request {request_state.request_id} with priority {priority:.2f}, prompt_length={prompt_length}")
    
    async def _scheduler_loop(self):
        """主调度器循环"""
        logger.info("Scheduler loop started")
        while self.running:
            try:
                # 记录当前状态 
                self.count += 1
                if(self.count%1000==0):
                    self.count=0
                    logger.debug(f"Scheduler state: queue_size={len(self.request_queue)}, running={len(self.running_requests)}, completed={len(self.completed_requests)}")
                
                # 处理请求
                await self._process_requests()
                
                # 如果没有待处理请求，使用较长的休眠时间以减少CPU使用
                if not self.request_queue and not self.running_requests:
                    await asyncio.sleep(0.01)  # 10ms延迟，当没有请求时
                else:
                    # 如果有待处理请求，使用较短的延迟以提高响应速度
                    await asyncio.sleep(0.0001)  # 0.1ms延迟，提高响应速度
            except asyncio.CancelledError:
                logger.info("Scheduler loop was cancelled")
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                import traceback
                logger.error(traceback.format_exc())
                await asyncio.sleep(1)  # 出错时等待更长时间
    
    async def _process_requests(self):
        """处理队列中的请求"""
        if not self.request_queue:
            return
        
        logger.info(f"[SCHEDULER] Processing requests: {len(self.request_queue)} in queue, {len(self.running_requests)} running")
        
        # 形成批处理
        batches = self._form_batches()
        logger.info(f"[SCHEDULER] Formed {len(batches)} batch(es)")
        
        # 并行处理每个批处理以提高吞吐量
        if batches:
            for i, batch in enumerate(batches):
                logger.info(f"[SCHEDULER] Starting batch {i+1}/{len(batches)} with {len(batch)} requests")
            batch_tasks = [self._process_batch(batch) for batch in batches]
            await asyncio.gather(*batch_tasks, return_exceptions=True)
            logger.info(f"[SCHEDULER] All {len(batches)} batch(es) completed")
    
    def _form_batches(self) -> List[List[RequestState]]:
        """根据请求长度动态形成批处理
        
        优化策略：根据请求长度动态调整批处理大小，长请求使用较小的批处理大小，
        短请求使用较大的批处理大小，确保总token数不超过max_context_length。
        对于短序列请求，优先考虑减少延迟而非最大化吞吐量。
        """
        logger.info(f"[SCHEDULER] === Forming batches ===")
        logger.info(f"[SCHEDULER] Queue size: {len(self.request_queue)}, Running: {len(self.running_requests)}")
        
        batches = []
        temp_queue = []
        
        # 从优先队列中取出所有请求
        while self.request_queue:
            temp_queue.append(heapq.heappop(self.request_queue))
        
        logger.info(f"[SCHEDULER] Extracted {len(temp_queue)} requests from queue")
        for priority, req in temp_queue:
            logger.info(f"[SCHEDULER]   Request {req.request_id}: prompt_len={len(req.tokenized_prompt)}, priority={priority:.2f}")
        
        # 计算当前运行请求的总长度
        running_requests_length = sum(
            len(req.tokenized_prompt) + len(req.generated_tokens) 
            for req in self.running_requests.values()
        )
        
        logger.info(f"[SCHEDULER] Running requests total length: {running_requests_length} tokens (max: {self.max_context_length})")
        
        # 如果运行中的请求已经占用了大部分上下文，等待它们完成
        # 降低阈值到75%以允许更多并发处理
        if running_requests_length > self.max_context_length * 0.75:
            logger.warning(f"[SCHEDULER] Running requests using {running_requests_length/self.max_context_length*100:.1f}% of context, waiting")
            # 将所有请求放回队列
            for item in temp_queue:
                heapq.heappush(self.request_queue, item)
            return []
        
        # 尝试形成多个批处理
        remaining_requests = temp_queue.copy()
        batch_num = 0
        while remaining_requests:
            batch_num += 1
            current_batch = []
            current_batch_length = 0
            batch_requests = []
            
            # 根据剩余请求长度计算动态批处理大小
            if remaining_requests:
                # 估算平均请求长度
                avg_request_length = sum(
                    len(req[1].tokenized_prompt) for req in remaining_requests
                ) / len(remaining_requests)
                
                logger.info(f"[SCHEDULER] Average request length: {avg_request_length:.1f} tokens")
                
                # 根据平均请求长度动态调整批处理大小
                # 优化策略：更积极地使用更大的批处理大小以提高吞吐量
                if avg_request_length > 500:
                    # 长请求：使用较小的批处理大小
                    dynamic_batch_size = max(2, self.max_batch_size // 2)
                elif avg_request_length > 200:
                    # 中等长度请求：使用中等批处理大小
                    dynamic_batch_size = max(3, self.max_batch_size // 1.5)
                else:
                    # 短请求：使用最大批处理大小，甚至可以超过max_batch_size
                    dynamic_batch_size = int(self.max_batch_size * 1.5)
                
                logger.info(f"[SCHEDULER] Dynamic batch size: {dynamic_batch_size} (max: {self.max_batch_size})")
            else:
                dynamic_batch_size = self.max_batch_size
            
            # 尝试将尽可能多的请求加入当前批处理
            for i in range(len(remaining_requests)):
                _, request = remaining_requests[i]
                
                # 计算添加此请求后的总长度
                request_length = len(request.tokenized_prompt) + len(request.generated_tokens)
                total_length = running_requests_length + current_batch_length + request_length
                
                # 检查是否超出上下文长度限制或动态批处理大小限制
                if (total_length <= self.max_context_length and 
                    len(current_batch) < dynamic_batch_size):
                    current_batch.append(request)
                    current_batch_length += request_length
                    batch_requests.append(i)
                    logger.info(f"[SCHEDULER]   Added request {request.request_id} to batch {batch_num}: {request_length} tokens, batch total: {current_batch_length}")
                else:
                    logger.info(f"[SCHEDULER]   Skipped request {request.request_id}: total={total_length} > {self.max_context_length} or batch_size={len(current_batch)} >= {dynamic_batch_size}")
            
            # 如果形成了有效的批处理，添加到批次列表中
            if current_batch:
                batches.append(current_batch)
                logger.info(f"[SCHEDULER] Formed batch {batch_num}: {len(current_batch)} requests, {current_batch_length} tokens")
                
                # 从剩余请求中移除已处理的请求
                remaining_requests = [req for i, req in enumerate(remaining_requests) if i not in batch_requests]
                
                # 更新running_requests_length以反映新批处理
                running_requests_length += current_batch_length
            else:
                # 无法形成更多批处理，跳出循环
                logger.warning(f"[SCHEDULER] Could not form batch {batch_num}, no more batches can be formed")
                break
        
        # 将未处理的请求放回队列
        for item in remaining_requests:
            heapq.heappush(self.request_queue, item)
        
        logger.info(f"[SCHEDULER] === Batch formation complete: {len(batches)} batches, {len(remaining_requests)} requests remain in queue ===")
        return batches
    
    async def _process_batch(self, batch: List[RequestState]):
        """处理一批请求，持续处理直到所有请求完成"""
        batch_start_time = time.time()
        batch_id = f"batch_{int(batch_start_time)}"
        
        logger.info(f"[SCHEDULER] === Starting batch {batch_id} ===")
        logger.info(f"[SCHEDULER] Batch size: {len(batch)} requests")
        
        # 将请求移到运行状态
        for request in batch:
            self.running_requests[request.request_id] = request
            logger.info(f"[SCHEDULER] Moved request {request.request_id} to running state")
        
        try:
            # 持续处理批处理，直到所有请求都完成
            iteration = 0
            while True:
                iteration += 1
                iteration_start = time.time()
                
                logger.info(f"[SCHEDULER] --- Batch {batch_id} iteration {iteration} ---")
                
                # 准备输入 - 只包含未完成的请求
                active_batch = [req for req in batch if not req.is_finished]
                
                logger.info(f"[SCHEDULER] Active requests in iteration {iteration}: {len(active_batch)}/{len(batch)}")
                
                if not active_batch:
                    logger.info(f"[SCHEDULER] All requests completed in batch {batch_id}")
                    break
                
                batch_inputs = self._prepare_batch_inputs(active_batch)
                
                # 运行模型推理
                inference_start_time = time.time()
                logger.info(f"[SCHEDULER] Running model inference for {len(active_batch)} requests...")
                batch_outputs = await self.model_executor.forward(batch_inputs)
                inference_duration = time.time() - inference_start_time
                logger.info(f"[SCHEDULER] Model inference completed in {inference_duration:.3f}s")
                
                # 处理输出
                self._process_batch_outputs(active_batch, batch_outputs)
                
                # 检查是否所有请求都已完成
                completed_count = sum(1 for req in batch if req.is_finished)
                logger.info(f"[SCHEDULER] Completed requests: {completed_count}/{len(batch)}")
                
                iteration_duration = time.time() - iteration_start
                logger.info(f"[SCHEDULER] Iteration {iteration} completed in {iteration_duration:.3f}s")
                
                # 如果所有请求都完成，退出循环
                if completed_count == len(batch):
                    logger.info(f"[SCHEDULER] All requests completed in batch {batch_id}")
                    break
            
        except Exception as e:
            logger.error(f"[SCHEDULER] Error processing batch {batch_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 将批次中的所有请求标记为失败
            for request in batch:
                request.is_finished = True
                request.error = str(e)
        
        # 清理已完成的请求
        completed_requests = [req for req in batch if req.is_finished]
        
        # 处理已完成的请求
        for request in completed_requests:
            if request.request_id in self.running_requests:
                del self.running_requests[request.request_id]
                logger.info(f"[SCHEDULER] Removed request {request.request_id} from running state")
            self.completed_requests[request.request_id] = request
            logger.info(f"[SCHEDULER] Added request {request.request_id} to completed requests")
            
        batch_duration = time.time() - batch_start_time
        logger.info(f"[SCHEDULER] === Batch {batch_id} completed in {batch_duration:.3f}s ===")
    
    def _prepare_batch_inputs(self, batch: List[RequestState]) -> Dict:
        """准备批处理的输入"""
        logger.debug(f"[SCHEDULER] Preparing inputs for batch of {len(batch)} requests")
        # 为批次中的所有请求组合提示和当前上下文
        all_input_ids = []
        request_positions = []
        sequence_ids = []
        
        current_pos = 0
        for request in batch:
            # 组合提示和生成的令牌
            input_ids = request.tokenized_prompt + request.generated_tokens
            all_input_ids.extend(input_ids)
            request_positions.append((current_pos, current_pos + len(input_ids)))
            sequence_ids.append(request.request_id)
            logger.debug(f"[SCHEDULER] Request {request.request_id}: {len(input_ids)} tokens at positions {current_pos}-{current_pos + len(input_ids)}")
            current_pos += len(input_ids)
        
        logger.debug(f"[SCHEDULER] Prepared batch inputs: {len(all_input_ids)} total tokens for {len(batch)} requests")
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
            if request.is_finished:
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
                request.generated_tokens.append(next_token)
                logger.debug(f"Request {request.request_id}: added token, total generated: {len(request.generated_tokens)}")
                
                # 检查停止条件
                logger.debug(f"Request {request.request_id}: checking stopping conditions")
                self._check_stopping_conditions(request, next_token)
                logger.debug(f"Request {request.request_id}: finished={request.is_finished}")
                
                # 额外检查：如果我们已经生成了足够的令牌，则标记为完成
                if len(request.generated_tokens) >= request.max_tokens and not request.is_finished:
                    request.is_finished = True
                    logger.debug(f"Request {request.request_id} marked as finished after generating {len(request.generated_tokens)} tokens")
            else:
                logger.error(f"Index out of bounds: start_pos={start_pos}, end_pos={end_pos}, logits_shape={logits.shape}")
                request.is_finished = True
                request.error = "Index out of bounds"
    
    def _check_stopping_conditions(self, request: RequestState, next_token: int):
        """检查请求是否应该停止"""
        logger.debug(f"Checking stopping conditions for request {request.request_id}")
        # 检查最大令牌数
        if len(request.generated_tokens) >= request.max_tokens:
            request.is_finished = True
            logger.info(f"Request {request.request_id} finished due to max tokens ({len(request.generated_tokens)} >= {request.max_tokens})")
            return
        
        # 检查停止令牌（简化版）
        # 在实际实现中，这将检查实际的停止令牌
        if next_token == 0:  # Placeholder stop token
            request.is_finished = True
            logger.info(f"Request {request.request_id} finished due to stop token")
            return

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取KV缓存统计信息"""
        return self.kv_cache.get_cache_stats()

    def get_detailed_cache_stats(self) -> Dict[str, Any]:
        """获取详细的KV缓存统计信息"""
        return self.kv_cache.get_detailed_stats()
    
    async def wait_for_request(self, request_id: str, timeout: float = 300.0) -> RequestState:
        """等待请求完成
        
        Args:
            request_id: 请求ID
            timeout: 超时时间（秒）
            
        Returns:
            完成的请求状态
            
        Raises:
            asyncio.TimeoutError: 如果请求超时
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if request_id in self.completed_requests:
                return self.completed_requests[request_id]
            await asyncio.sleep(0.01)
        
        raise asyncio.TimeoutError(f"Request {request_id} timed out after {timeout} seconds")
    
    def get_request_result(self, request_id: str) -> Optional[RequestState]:
        """获取请求结果
        
        Args:
            request_id: 请求ID
            
        Returns:
            请求状态，如果请求不存在则返回None
        """
        return self.completed_requests.get(request_id)
    
    async def generate_with_scheduler(self, request_id: str, prompt: str, max_tokens: int = 100, 
                                      temperature: float = 0.7) -> str:
        """使用调度器生成文本（非流式）
        
        Args:
            request_id: 请求ID
            prompt: 输入文本
            max_tokens: 最大生成token数
            temperature: 温度参数
            
        Returns:
            生成的文本
        """
        from xllm.tokenizer_manager import TokenizerManager
        
        logger.info(f"[SCHEDULER] === generate_with_scheduler: {request_id} ===")
        logger.info(f"[SCHEDULER] Prompt length: {len(prompt)} chars, max_tokens: {max_tokens}, temperature: {temperature}")
        
        # 创建分词器管理器实例（用于编码和解码）
        tokenizer_manager = TokenizerManager(self.model_executor.model_path, self.model_executor)
        
        # 编码输入
        input_ids = tokenizer_manager.encode(prompt)
        logger.info(f"[SCHEDULER] Encoded prompt: {len(input_ids)} tokens")
        
        # 创建请求状态
        request_state = RequestState(
            request_id=request_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            stop_tokens=[],
            tokenized_prompt=input_ids
        )
        
        logger.info(f"[SCHEDULER] Created RequestState for {request_id}")
        
        # 添加到调度队列
        logger.info(f"[SCHEDULER] Adding request {request_id} to scheduler queue...")
        self.add_request(request_state)
        logger.info(f"[SCHEDULER] Request {request_id} added to queue, waiting for completion...")
        
        # 等待请求完成
        wait_start = time.time()
        completed_request = await self.wait_for_request(request_id)
        wait_duration = time.time() - wait_start
        logger.info(f"[SCHEDULER] Request {request_id} completed after {wait_duration:.3f}s")
        
        # 解码生成的token
        if completed_request.error:
            logger.error(f"[SCHEDULER] Request {request_id} failed with error: {completed_request.error}")
            raise Exception(completed_request.error)
        
        logger.info(f"[SCHEDULER] Generated {len(completed_request.generated_tokens)} tokens for {request_id}")
        generated_text = tokenizer_manager.decode(completed_request.generated_tokens)
        logger.info(f"[SCHEDULER] Decoded text length: {len(generated_text)} chars")
        logger.info(f"[SCHEDULER] === generate_with_scheduler complete: {request_id} ===")
        
        return generated_text