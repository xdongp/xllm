"""
优化版本的模型执行器 - 包含多种性能优化
"""
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.quantization
import math
import sys
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import gc

# 将父目录添加到路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoConfig
from xllm.sampler import Sampler

# 导入C语言采样器
try:
    from sampler_c import CSampler
    C_SAMPLER_AVAILABLE = True
except ImportError:
    C_SAMPLER_AVAILABLE = False
    logger.warning("C sampler not available, falling back to Python sampler")

# 设置日志
logger = logging.getLogger(__name__)


class OptimizedModelExecutor:
    """优化的模型执行器 - 支持多种性能优化"""
    
    def __init__(
        self,
        model_path: str,
        quantization: str = None,
        use_c_sampler: bool = False,
        enable_compile: bool = True,
        enable_batching: bool = False,
        batch_size: int = 8,
        enable_multiprocess: bool = False,
        num_processes: int = None
    ):
        self.model_path = model_path
        self.quantization = quantization
        self.use_c_sampler = use_c_sampler
        self.enable_compile = enable_compile
        self.enable_batching = enable_batching
        self.batch_size = batch_size
        self.enable_multiprocess = enable_multiprocess
        self.num_processes = num_processes or os.cpu_count()
        
        self.device = torch.device("cpu")
        self.model = None
        self.config = None
        
        # 创建线程池用于并行推理
        num_cpu_cores = os.cpu_count() or 4
        num_threads = num_cpu_cores * 2  # 使用CPU核心数的2倍作为线程数
        self.executor = ThreadPoolExecutor(
            max_workers=num_threads,
            thread_name_prefix="ModelInference"
        )
        logger.info(f"Created thread pool with {num_threads} workers (2x CPU cores)")
        
        # 选择采样器
        if self.use_c_sampler and C_SAMPLER_AVAILABLE:
            logger.info("Using C language sampler implementation")
            self.sampler = CSampler()
            self.sampler_type = "C"
        else:
            if self.use_c_sampler and not C_SAMPLER_AVAILABLE:
                logger.warning("C sampler requested but not available")
            logger.info("Using Python sampler implementation")
            self.sampler = Sampler()
            self.sampler_type = "Python"
        
        # 初始化KV缓存
        from kv_cache import get_global_kv_cache
        self.kv_cache = get_global_kv_cache()
        
        # 应用推理优化
        self._apply_inference_optimizations()
        
        # 加载模型
        self._load_model()
        
        # 预热模型（重要！）
        self._warmup_model()
        
        logger.info("="*60)
        logger.info("优化后的模型执行器初始化完成")
        logger.info(f"量化方式: {self.quantization or '无'}")
        logger.info(f"torch.compile: {'启用' if self.enable_compile else '禁用'}")
        logger.info(f"批处理: {'启用' if self.enable_batching else '禁用'}")
        logger.info(f"多进程: {'启用' if self.enable_multiprocess else '禁用'}")
        logger.info(f"线程池: {num_threads} 个工作线程")
        logger.info("="*60)
    
    def __del__(self):
        """析构函数，确保线程池正确关闭"""
        if hasattr(self, 'executor'):
            logger.info("Shutting down model inference thread pool...")
            self.executor.shutdown(wait=True)
            logger.info("Model inference thread pool shut down successfully")
    
    def _load_model(self):
        """加载模型并应用优化"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # 加载配置
            self.config = AutoConfig.from_pretrained(self.model_path)
            logger.info(f"Model type: {getattr(self.config, 'model_type', 'Unknown')}")
            
            # 根据量化选项加载模型
            if self.quantization == "int8":
                self.model = self._load_with_int8_quantization()
            elif self.quantization == "int4":
                self.model = self._load_with_int4_quantization()
            else:
                self.model = self._load_full_precision()
            
            # 应用torch.compile优化
            if self.enable_compile:
                self.model = self._apply_torch_compile(self.model)
            
            # 应用内存优化
            self.optimize_memory()
            
            # 打印模型信息
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model parameters: {total_params:,}")
            logger.info(f"Model size: {self._get_model_size():.2f} MB")
            
            logger.info(f"✅ Successfully loaded and optimized model")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _load_full_precision(self):
        """加载全精度模型"""
        logger.info("Loading model with full precision (float32)...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        model = model.to(self.device)
        return model
    
    def _load_with_int8_quantization(self):
        """使用动态int8量化加载模型"""
        logger.info("Loading model with dynamic int8 quantization...")
        
        try:
            # 首先尝试bitsandbytes
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_enable_fp32_cpu_offload=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto"
            )
            
            logger.info("✅ Applied bitsandbytes int8 quantization")
            return model
            
        except Exception as e:
            logger.warning(f"bitsandbytes量化失败: {e}")
            logger.info("Falling back to CPU-only loading...")
            
            # 回退到CPU加载
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            model = model.to(self.device)
            
            logger.info("✅ Loaded model on CPU without quantization")
            return model
    
    def _load_with_int4_quantization(self):
        """使用int4量化加载模型"""
        logger.info("Loading model with int4 quantization...")
        
        try:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto"
            )
            
            logger.info("✅ Applied int4 quantization")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load int4 model: {e}")
            logger.info("Falling back to CPU-only loading...")
            
            # 回退到CPU加载
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            model = model.to(self.device)
            
            logger.info("✅ Loaded model on CPU without quantization")
            return model
    
    def _apply_torch_compile(self, model):
        """应用torch.compile优化"""
        logger.info("Applying torch.compile optimization...")
        try:
            compiled_model = torch.compile(
                model,
                mode="max-autotune",
                backend="inductor",
                fullgraph=False
            )
            logger.info("✅ Applied torch.compile optimization")
            return compiled_model
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}, falling back to original model")
            return model
    
    def _warmup_model(self):
        """模型预热"""
        logger.info("Starting model warmup...")
        start_time = time.time()
        
        try:
            # 创建示例输入
            input_ids = torch.randint(0, 1000, (1, 10), dtype=torch.long, device=self.device)
            
            # 运行几次推理
            for i in range(3):
                with torch.no_grad():
                    _ = self.model(input_ids)
                logger.debug(f"Warmup iteration {i+1}/3 completed")
            
            warmup_time = time.time() - start_time
            logger.info(f"✅ Model warmup completed in {warmup_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Model warmup failed: {e}")
            raise
    
    def _apply_inference_optimizations(self):
        """应用推理优化"""
        # 设置PyTorch优化参数
        torch.backends.cudnn.benchmark = True  # 优化卷积运算
        torch.set_num_threads(min(8, os.cpu_count()))  # 限制CPU线程数
        torch.set_num_interop_threads(1)  # 限制并行线程数
        
        # 设置环境变量
        os.environ['OMP_NUM_THREADS'] = str(min(8, os.cpu_count()))
        os.environ['MKL_NUM_THREADS'] = str(min(8, os.cpu_count()))
        
        logger.info("Applied inference optimizations")
    
    def optimize_memory(self):
        """优化内存使用"""
        # 禁用梯度计算（推理时不需要）
        torch.set_grad_enabled(False)
        
        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 触发垃圾回收
        gc.collect()
        
        logger.info("Applied memory optimizations")
    
    def _get_model_size(self):
        """获取模型大小（MB）"""
        total_size = 0
        for param in self.model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size / (1024 * 1024)  # 转换为MB
    
    def _prepare_past_key_values(self, sequence_ids: List[Optional[str]]) -> Optional[Tuple]:
        """准备KV缓存"""
        cached_kvs = []
        has_cache = False
        
        for seq_id in sequence_ids:
            if seq_id and seq_id in self.kv_cache:
                cached_kv = self.kv_cache.get(seq_id)
                cached_kvs.append(cached_kv)
                has_cache = True
            else:
                cached_kvs.append(None)
        
        return tuple(cached_kvs) if has_cache else None
    
    def _update_kv_cache(self, sequence_ids: List[Optional[str]], past_key_values: Tuple):
        """更新KV缓存"""
        for i, seq_id in enumerate(sequence_ids):
            if seq_id and i < len(past_key_values):
                self.kv_cache.set(seq_id, past_key_values[i])
    
    async def forward(self, batch_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """优化的前向传播 - 支持异步并行处理"""
        import asyncio
        loop = asyncio.get_event_loop()
        
        # 使用线程池执行同步推理，释放事件循环
        result = await loop.run_in_executor(
            self.executor,
            self._forward_sync,
            batch_inputs
        )
        
        return result
    
    def _forward_sync(self, batch_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """同步执行的模型推理"""
        start_time = time.time()
        
        # 转换输入
        input_ids = torch.tensor(
            batch_inputs["input_ids"],
            dtype=torch.long,
            device=self.device
        )
        batch_size = batch_inputs["batch_size"]
        sequence_ids = batch_inputs.get("sequence_ids", [None] * batch_size)
        
        # KV缓存处理
        past_key_values = self._prepare_past_key_values(sequence_ids)
        
        # 运行模型推理
        with torch.no_grad():
            if past_key_values:
                outputs = self.model(input_ids, past_key_values=past_key_values)
            else:
                outputs = self.model(input_ids)
            
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            # 更新KV缓存
            if hasattr(outputs, 'past_key_values'):
                self._update_kv_cache(sequence_ids, outputs.past_key_values)
        
        # 处理输出
        if logits.dim() > 2:
            logits = logits.squeeze(0)
        
        duration = time.time() - start_time
        logger.debug(f"Forward pass completed in {duration*1000:.2f} ms")
        
        return {
            "logits": logits,
            "request_positions": batch_inputs["request_positions"]
        }

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 10, 
                temperature: float = 0.7, do_sample: bool = True) -> torch.Tensor:
        """生成方法 - 支持实际模型推理
        
        Args:
            input_ids: 输入的token列表
            max_new_tokens: 要生成的新token数量
            temperature: 温度参数
            do_sample: 是否使用采样
            
        Returns:
            只包含新生成的token的tensor
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # 确保输入在正确的设备上
        input_ids = input_ids.to(self.device)
        
        # 保存原始输入长度
        original_length = input_ids.shape[1]
        
        # 只生成请求数量的token
        for i in range(max_new_tokens):
            # 获取模型输出
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[:, -1, :]  # 获取最后一个位置的logits
            
            # 应用温度并采样
            if do_sample:
                adjusted_logits = logits / temperature
                probs = torch.softmax(adjusted_logits, dim=-1)
                token = torch.multinomial(probs, 1)[0, 0].item()
            else:
                token = torch.argmax(logits, dim=-1)[0, 0].item()
            
            # 检查是否是结束token
            if hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'eos_token_id'):
                if token == self.tokenizer.eos_token_id:
                    break
            
            # 添加生成的token
            token_tensor = torch.tensor([[token]], device=self.device, dtype=torch.long)
            input_ids = torch.cat([input_ids, token_tensor], dim=1)
        
        # 只返回新生成的token，而不是完整的序列
        return input_ids[:, original_length:]
    
    def generate_stream(self, input_ids: torch.Tensor, max_new_tokens: int = 10, 
                      temperature: float = 0.7, do_sample: bool = True):
        """流式生成方法 - 逐token生成并返回
        
        Args:
            input_ids: 输入的token列表
            max_new_tokens: 要生成的新token数量
            temperature: 温度参数
            do_sample: 是否使用采样
            
        Yields:
            每个生成的token的tensor
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # 确保输入在正确的设备上
        input_ids = input_ids.to(self.device)
        
        # 只生成请求数量的token
        for i in range(max_new_tokens):
            # 获取模型输出
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[:, -1, :]  # 获取最后一个位置的logits
            
            # 应用温度并采样
            if do_sample:
                adjusted_logits = logits / temperature
                probs = torch.softmax(adjusted_logits, dim=-1)
                token = torch.multinomial(probs, 1)[0, 0]
            else:
                token = torch.argmax(logits, dim=-1)[0, 0]
            
            # 检查是否是结束token
            if hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'eos_token_id'):
                if token.item() == self.tokenizer.eos_token_id:
                    break
            
            # 返回生成的token
            yield token
            
            # 添加生成的token
            token_tensor = token.unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, token_tensor], dim=1)

    def generate_batch(self, requests: List[Dict]) -> List[List[int]]:
        """批量生成 - 为OptimizedModelExecutor添加此方法"""
        results = []
        for req in requests:
            input_ids = req.get("input_ids", [])
            max_new_tokens = req.get("max_new_tokens", 10)
            temperature = req.get("temperature", 0.7)
            do_sample = req.get("do_sample", True)
            
            # 使用单个生成方法处理每个请求
            if isinstance(input_ids, torch.Tensor):
                input_tensor = input_ids
            else:
                input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)
            
            try:
                # 调用模型生成
                output = self.generate(input_tensor, max_new_tokens=max_new_tokens, 
                                     temperature=temperature, do_sample=do_sample)
                if isinstance(output, torch.Tensor):
                    results.append(output[0].tolist())
                else:
                    # 如果生成失败，返回输入ID加上一些默认token
                    results.append(input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids + [0] * max_new_tokens)
            except Exception as e:
                logger.error(f"Error in generate_batch: {e}")
                # 如果生成失败，返回输入ID加上一些默认token
                results.append(input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids + [0] * max_new_tokens)
        
        return results


class BatchOptimizedExecutor(OptimizedModelExecutor):
    """支持批处理的优化执行器"""
    
    def __init__(self, model_path: str, batch_size: int = 8, **kwargs):
        super().__init__(model_path, **kwargs)
        self.batch_size = batch_size
        self.request_queue = []
        logger.info(f"✅ Batch processing enabled (batch_size={batch_size})")
    
    def add_request(self, input_ids: List[int], max_tokens: int = 100):
        """添加请求到队列"""
        self.request_queue.append({
            "input_ids": input_ids,
            "max_tokens": max_tokens,
            "generated_tokens": []
        })
    
    def process_batch(self):
        """批量处理请求"""
        if not self.request_queue:
            return []
        
        # 准备批次
        batch_size = min(self.batch_size, len(self.request_queue))
        batch_requests = self.request_queue[:batch_size]
        
        results = []
        for i, req in enumerate(batch_requests):
            input_ids = req.get("input_ids", [])
            max_new_tokens = req.get("max_new_tokens", 10)
            temperature = req.get("temperature", 0.7)
            do_sample = req.get("do_sample", True)
            
            # 使用单个生成方法处理每个请求
            if isinstance(input_ids, torch.Tensor):
                input_tensor = input_ids
            else:
                input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)
            
            try:
                # 调用模型生成
                output = self.generate(input_tensor, max_new_tokens=max_new_tokens, 
                                     temperature=temperature, do_sample=do_sample)
                if isinstance(output, torch.Tensor):
                    results.append(output[0].tolist())
                else:
                    # 如果生成失败，返回输入ID加上一些默认token
                    results.append(input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids + [0] * max_new_tokens)
            except Exception as e:
                logger.error(f"Error in process_batch: {e}")
                # 如果生成失败，返回输入ID加上一些默认token
                results.append(input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids + [0] * max_new_tokens)
        
        # 从队列中移除已处理的请求
        self.request_queue = self.request_queue[batch_size:]
        
        return results
    
    def generate_batch(self, requests: List[Dict]) -> List[List[int]]:
        """批量生成 - 在BatchOptimizedExecutor中重写此方法"""
        # 清空队列并添加新请求
        self.request_queue = []
        for req in requests:
            self.add_request(req.get("input_ids", []), req.get("max_new_tokens", 10))
        
        # 处理所有请求
        results = []
        while self.request_queue:
            batch_results = self.process_batch()
            results.extend(batch_results)
        
        return results


# 便捷函数
def create_optimized_executor(
    model_path: str,
    quantization: str = "int8",
    enable_compile: bool = True,
    enable_batching: bool = False,
    batch_size: int = 8
) -> OptimizedModelExecutor:
    """
    创建优化后的模型执行器
    
    Args:
        model_path: 模型路径
        quantization: 量化方式 ("int8", "int4", None)
        enable_compile: 是否启用torch.compile
        enable_batching: 是否启用批处理
        batch_size: 批次大小
    
    Returns:
        优化后的模型执行器
    """
    if enable_batching:
        return BatchOptimizedExecutor(
            model_path=model_path,
            quantization=quantization,
            enable_compile=enable_compile,
            batch_size=batch_size
        )
    else:
        return OptimizedModelExecutor(
            model_path=model_path,
            quantization=quantization,
            enable_compile=enable_compile
        )


if __name__ == "__main__":
    # 测试代码
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 创建优化执行器
    executor = create_optimized_executor(
        model_path="./model/Qwen/Qwen3-0.6B",
        quantization="int8",
        enable_compile=True
    )
    
    # 测试推理
    test_input = torch.randint(0, 1000, (1, 1))
    with torch.no_grad():
        output = executor.model(test_input)
        print(f"Output shape: {output.logits.shape}")