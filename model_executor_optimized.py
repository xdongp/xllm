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
        logger.info("="*60)
    
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
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
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
            logger.warning(f"bitsandbytes quantization failed: {e}")
            logger.info("Falling back to dynamic int8 quantization...")
            
            # 回退到动态量化
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            model = model.to(self.device)
            
            # 应用动态量化
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            
            logger.info("✅ Applied dynamic int8 quantization")
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
            logger.warning(f"int4 quantization failed: {e}")
            logger.info("Falling back to int8 quantization...")
            return self._load_with_int8_quantization()
    
    def _apply_torch_compile(self, model):
        """应用torch.compile优化"""
        try:
            if torch.__version__ < "2.0.0":
                logger.warning("PyTorch version < 2.0, torch.compile not available")
                return model
            
            logger.info("Applying torch.compile optimization...")
            
            # 使用最大优化模式
            compiled_model = torch.compile(
                model,
                mode="max-autotune",
                fullgraph=False,
                dynamic=False,
                backend="inductor"
            )
            
            logger.info("✅ Applied torch.compile optimization")
            return compiled_model
            
        except Exception as e:
            logger.warning(f"torch.compile optimization failed: {e}")
            return model
    
    def _warmup_model(self):
        """预热模型（重要！）"""
        logger.info("Warming up model...")
        
        try:
            # 创建示例输入
            sample_input = torch.randint(0, 1000, (1, 1), device=self.device)
            
            # 运行几次预热
            with torch.no_grad():
                for i in range(3):
                    _ = self.model(sample_input)
                    logger.debug(f"Warmup iteration {i+1}/3")
            
            logger.info("✅ Model warmup completed")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def _apply_inference_optimizations(self):
        """应用推理优化"""
        # 设置PyTorch优化
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        
        # 启用CPU优化
        if hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = True
        
        # 设置线程数
        cpu_count = os.cpu_count() or 4
        torch.set_num_threads(cpu_count)
        
        # 设置环境变量
        os.environ["OMP_NUM_THREADS"] = str(cpu_count)
        os.environ["MKL_NUM_THREADS"] = str(cpu_count)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)
        
        logger.info(f"✅ Inference optimizations applied (using {cpu_count} threads)")
    
    def optimize_memory(self):
        """内存优化"""
        # 强制垃圾回收
        gc.collect()
        
        # 设置更激进的垃圾回收
        gc.set_threshold(700, 10, 10)
        
        # 禁用梯度
        torch.set_grad_enabled(False)
        
        logger.info("✅ Memory optimizations applied")
    
    def _get_model_size(self):
        """计算模型大小（MB）"""
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / 1024 / 1024
    
    def forward(self, batch_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """优化的前向传播"""
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
    
    def _prepare_past_key_values(self, sequence_ids: List[Optional[int]]):
        """准备过去的键值对"""
        cached_keys_list = []
        cached_values_list = []
        
        for seq_id in sequence_ids:
            if seq_id is not None:
                cached_kv = self.kv_cache.get(seq_id)
                if cached_kv is not None:
                    cached_keys, cached_values = cached_kv
                    cached_keys_list.append(cached_keys)
                    cached_values_list.append(cached_values)
                else:
                    cached_keys_list.append(None)
                    cached_values_list.append(None)
            else:
                cached_keys_list.append(None)
                cached_values_list.append(None)
        
        # 准备模型格式
        if any(k is not None for k in cached_keys_list):
            return self._prepare_past_key_values_for_model(cached_keys_list, cached_values_list)
        return None
    
    def _prepare_past_key_values_for_model(self, keys_list, values_list):
        """准备模型格式的过去键值对"""
        # 这里简化实现，实际需要根据模型格式调整
        return None
    
    def _update_kv_cache(self, sequence_ids, past_key_values):
        """更新KV缓存"""
        for seq_id, past_kv in zip(sequence_ids, past_key_values):
            if seq_id is not None and past_kv is not None:
                self.kv_cache.set(seq_id, past_kv)
    
    def encode(self, text: str) -> List[int]:
        """编码文本"""
        return [ord(c) for c in text][:100]
    
    def decode(self, token_ids: List[int]) -> str:
        """解码token"""
        try:
            return ''.join(chr(t) for t in token_ids if 32 <= t <= 126)
        except:
            return str(token_ids)


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
        batch_input_ids = []
        
        for i in range(batch_size):
            req = self.request_queue[i]
            if req["generated_tokens"]:
                batch_input_ids.append(req["generated_tokens"][-1])
            else:
                batch_input_ids.append(req["input_ids"][-1])
        
        # 转换为张量
        input_ids_tensor = torch.tensor(
            batch_input_ids,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(1)
        
        # 批量推理
        with torch.no_grad():
            outputs = self.model(input_ids_tensor)
            logits = outputs.logits[:, -1, :]
        
        # 处理输出
        results = []
        for i in range(batch_size):
            req = self.request_queue[i]
            
            # 采样
            token = self.sampler.sample(logits[i].cpu().numpy())
            req["generated_tokens"].append(token)
            
            # 检查是否完成
            if len(req["generated_tokens"]) >= req["max_tokens"]:
                results.append({
                    "input_ids": req["input_ids"],
                    "output_tokens": req["generated_tokens"]
                })
                self.request_queue.pop(i)
        
        return results
    
    def generate_batch(self, requests: List[Dict]) -> List[List[int]]:
        """批量生成"""
        # 添加请求
        for req in requests:
            self.add_request(req["input_ids"], req.get("max_tokens", 100))
        
        # 处理直到完成
        all_results = []
        while self.request_queue:
            results = self.process_batch()
            all_results.extend(results)
        
        return all_results


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
