"""
xLLM的模型执行器 - CPU优化实现（整合优化版本）
"""
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import torch
import torch.nn as nn
import math
import sys
import os
import time
import logging
import threading
from queue import Queue

# 将父目录添加到路径中，以便我们可以从xllm包导入
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

# 导入量化支持
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


class TransformerLayer(nn.Module):
    """单个变压器层实现"""
    
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.intermediate_size = intermediate_size
        
        # 自注意力
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # 层归一化
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        
        # 前馈网络
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传递"""
        # 自注意力
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # 简化的注意力计算（仅用于演示）
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # 简化的注意力分数计算
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        hidden_states = self.o_proj(attn_output)
        hidden_states = residual + hidden_states
        
        # 前馈网络
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        hidden_states = self.down_proj(gate * torch.sigmoid(gate) * up)
        
        hidden_states = residual + hidden_states
        
        return hidden_states


class ModelExecutor:
    """CPU优化的模型执行器（整合优化版本）"""
    
    def __init__(self, 
                 model_path: str, 
                 quantization: str = None, 
                 use_c_sampler: bool = False,
                 enable_compile: bool = True,
                 warmup_iterations: int = 20):
        """
        初始化模型执行器
        
        Args:
            model_path: 模型路径
            quantization: 量化方法 ('int8', 'int4', None)
            use_c_sampler: 是否使用C采样器
            enable_compile: 是否启用torch.compile优化
            warmup_iterations: 预热迭代次数
        """
        self.model_path = model_path
        self.quantization = quantization
        self.use_c_sampler = use_c_sampler
        self.enable_compile = enable_compile
        self.warmup_iterations = warmup_iterations
        self.device = torch.device("cpu")  # 强制CPU执行
        self.model = None
        self.config = None
        
        # 选择采样器实现
        if self.use_c_sampler and C_SAMPLER_AVAILABLE:
            logger.info("Using C language sampler implementation")
            self.sampler = CSampler()
            self.sampler_type = "C"
        else:
            if self.use_c_sampler and not C_SAMPLER_AVAILABLE:
                logger.warning("C sampler requested but not available, falling back to Python sampler")
            logger.info("Using Python sampler implementation")
            self.sampler = Sampler()
            self.sampler_type = "Python"
        
        # 设置CPU线程数以更好地利用多核
        torch.set_num_threads(torch.get_num_threads())
        
        # 初始化KV缓存
        from kv_cache import get_global_kv_cache
        self.kv_cache = get_global_kv_cache()
        
        # 应用推理优化
        self._apply_inference_optimizations()
        
        # 加载模型
        self._load_model()
        
        # 应用torch.compile优化
        if self.enable_compile:
            self._apply_torch_compile()
        
        # 预热模型
        self._warmup_model()
    
    def _load_model(self):
        """从路径加载模型，可选择量化"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            logger.info(f"Quantization method: {self.quantization}")
            
            # 首先加载模型配置并打印基本信息
            logger.info("Loading model configuration...")
            self.config = AutoConfig.from_pretrained(self.model_path)
            logger.info(f"Model type: {getattr(self.config, 'model_type', 'Unknown')}")
            logger.info(f"Model architecture: {getattr(self.config, 'architectures', 'Unknown')}")
            
            # 打印模型加载进度
            logger.info("Starting model loading process...")
            logger.info("这可能需要几分钟时间，具体取决于模型大小...")
            
            # 根据量化选项加载模型
            if self.quantization == "int8":
                self._load_with_int8_quantization()
            elif self.quantization == "int4":
                self._load_with_int4_quantization()
            else:
                self._load_full_precision()
            
            # 打印模型信息
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            logger.info(f"Model parameters (trainable): {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
            
            logger.info(f"Successfully loaded model from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {str(e)}")
            logger.error(f"Traceback: {e}", exc_info=True)
            
            # 如果模型加载失败，使用占位符模型
            logger.warning("Using placeholder model")
            self.model = self._create_placeholder_model()
    
    def _load_full_precision(self):
        """加载全精度模型"""
        logger.info("Loading model with full precision (float32)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.model = self.model.to(self.device)
        self._optimize_model_for_speed(self.model)
        self.optimize_memory()
    
    def _load_with_int8_quantization(self):
        """使用Int8量化加载模型"""
        if HAS_BNB:
            try:
                logger.info("Loading model with int8 quantization...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                self.model = self.model.to(self.device)
                self._optimize_model_for_speed(self.model)
                self.optimize_memory()
            except Exception as e:
                logger.warning(f"Failed to load model with int8 quantization: {e}")
                logger.warning("Falling back to full precision")
                self.quantization = None
                self._load_full_precision()
        else:
            logger.warning("bitsandbytes not available, falling back to full precision")
            self.quantization = None
            self._load_full_precision()
    
    def _load_with_int4_quantization(self):
        """使用Int4量化加载模型"""
        if HAS_BNB:
            try:
                logger.info("Loading model with int4 quantization...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = self.model.to(self.device)
                self._optimize_model_for_speed(self.model)
                self.optimize_memory()
            except Exception as e:
                logger.warning(f"Failed to load model with int4 quantization: {e}")
                logger.warning("Falling back to full precision")
                self.quantization = None
                self._load_full_precision()
        else:
            logger.warning("bitsandbytes not available, falling back to full precision")
            self.quantization = None
            self._load_full_precision()
    
    def _apply_torch_compile(self):
        """应用torch.compile优化"""
        try:
            if hasattr(torch, 'compile'):
                logger.info("Applying torch.compile optimization...")
                # 使用最激进的优化模式和inductor后端
                self.model = torch.compile(
                    self.model,
                    mode="max-autotune",
                    backend="inductor"
                )
                logger.info("✅ torch.compile优化已启用 (max-autotune mode)")
            else:
                logger.warning("torch.compile not available (requires PyTorch 2.0+)")
        except Exception as e:
            logger.warning(f"⚠️ torch.compile优化失败: {e}")
    
    def _warmup_model(self):
        """预热模型以获得最佳性能"""
        if self.warmup_iterations <= 0:
            logger.info("Skipping model warmup")
            return
        
        logger.info(f"Warming up model with {self.warmup_iterations} iterations...")
        warmup_start = time.time()
        
        try:
            self.model.eval()
            with torch.no_grad():
                for i in range(self.warmup_iterations):
                    # 创建虚拟输入
                    dummy_input = torch.randint(
                        0, 
                        self.config.vocab_size, 
                        (1, 1), 
                        dtype=torch.long,
                        device=self.device
                    )
                    
                    # 运行前向传播
                    _ = self.model(dummy_input)
                    
                    if (i + 1) % 5 == 0:
                        logger.debug(f"Warmup progress: {i + 1}/{self.warmup_iterations}")
            
            warmup_duration = time.time() - warmup_start
            logger.info(f"✅ 模型预热完成，耗时: {warmup_duration:.2f}秒")
        except Exception as e:
            logger.warning(f"⚠️ 模型预热失败: {e}")
    
    def _create_placeholder_model(self):
        """创建占位符模型用于测试"""
        class PlaceholderModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(32000, 512)
                self.layers = nn.ModuleList([
                    TransformerLayer(512, 8, 1024) for _ in range(4)
                ])
                self.norm = nn.LayerNorm(512)
                self.lm_head = nn.Linear(512, 32000, bias=False)
            
            def forward(self, input_ids, **kwargs):
                hidden_states = self.embed_tokens(input_ids)
                for layer in self.layers:
                    hidden_states = layer(hidden_states)
                hidden_states = self.norm(hidden_states)
                logits = self.lm_head(hidden_states)
                if 'past_key_values' in kwargs and kwargs['past_key_values'] is not None:
                    return type('obj', (object,), {'logits': logits, 'past_key_values': kwargs['past_key_values']})()
                return type('obj', (object,), {'logits': logits})()
        
        return PlaceholderModel()
    
    def _extract_past_key_values(self, outputs) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """从模型输出中提取past_key_values"""
        if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
            past_kv = outputs.past_key_values
            if isinstance(past_kv, (list, tuple)) and len(past_kv) > 0:
                first_element = past_kv[0]
                if isinstance(first_element, (list, tuple)) and len(first_element) >= 2:
                    return past_kv
        elif isinstance(outputs, (list, tuple)) and len(outputs) > 1:
            for item in outputs:
                if hasattr(item, 'past_key_values') and item.past_key_values is not None:
                    past_kv = item.past_key_values
                    if isinstance(past_kv, (list, tuple)) and len(past_kv) > 0:
                        first_element = past_kv[0]
                        if isinstance(first_element, (list, tuple)) and len(first_element) >= 2:
                            return past_kv
        return None
    
    def _update_kv_cache(self, sequence_ids: List[str], past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]):
        """更新KV缓存"""
        if not past_key_values or not sequence_ids:
            return
            
        if not isinstance(past_key_values, (list, tuple)):
            logger.warning(f"Unexpected past_key_values type: {type(past_key_values)}")
            return
            
        for i, seq_id in enumerate(sequence_ids):
            if seq_id is not None and i < len(past_key_values):
                layer_key_values = past_key_values[i]
                if isinstance(layer_key_values, (tuple, list)) and len(layer_key_values) >= 2:
                    keys, values = layer_key_values[0], layer_key_values[1]
                    if isinstance(keys, torch.Tensor) and isinstance(values, torch.Tensor):
                        self.kv_cache.put(seq_id, keys, values)
                        logger.debug(f"Updated KV cache for sequence {seq_id}")
                    else:
                        logger.warning(f"Invalid key/value types for sequence {seq_id}: {type(keys)}, {type(values)}")
                else:
                    logger.debug(f"Skipping layer {i} for sequence {seq_id} due to invalid format")

    def _prepare_past_key_values_for_model(self, cached_keys_list: List[Optional[torch.Tensor]], 
                                          cached_values_list: List[Optional[torch.Tensor]]) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """准备past_key_values参数供模型使用"""
        if not any(k is not None for k in cached_keys_list):
            return None
            
        past_key_values = []
        for keys, values in zip(cached_keys_list, cached_values_list):
            if keys is not None and values is not None:
                keys = keys.to(self.device)
                values = values.to(self.device)
                past_key_values.append((keys, values))
            else:
                past_key_values.append(None)
                
        past_key_values = [item for item in past_key_values if item is not None]
        return past_key_values if past_key_values else None
    
    async def forward(self, batch_inputs: Dict) -> Dict:
        """
        一批输入的前向传递
        
        Args:
            batch_inputs: Dictionary containing input data
                - input_ids: List of token IDs
                - request_positions: Positions of each request in the batch
                - batch_size: Size of the batch
                - sequence_ids: List of sequence IDs for KV cache
                
        Returns:
            Dictionary containing output logits and metadata
        """
        logger.debug(f"ModelExecutor.forward called with batch_size={batch_inputs['batch_size']}")
        start_time = time.time()
        
        input_ids = torch.tensor(batch_inputs["input_ids"], dtype=torch.long, device=self.device)
        request_positions = batch_inputs["request_positions"]
        batch_size = batch_inputs["batch_size"]
        sequence_ids = batch_inputs.get("sequence_ids", [None] * batch_size)
        
        logger.debug(f"Input tensor shape: {input_ids.shape}")
        logger.debug(f"Request positions: {request_positions}")
        logger.debug(f"Sequence IDs: {sequence_ids}")
        
        # 尝试从KV缓存中获取缓存的键值对
        cached_keys_list = []
        cached_values_list = []
        cache_hits = 0
        cache_misses = 0
        
        for seq_id in sequence_ids:
            if seq_id is not None:
                cached_kv = self.kv_cache.get(seq_id)
                if cached_kv is not None:
                    cached_keys, cached_values = cached_kv
                    cached_keys_list.append(cached_keys)
                    cached_values_list.append(cached_values)
                    cache_hits += 1
                    logger.debug(f"KV cache hit for sequence {seq_id}")
                else:
                    cached_keys_list.append(None)
                    cached_values_list.append(None)
                    cache_misses += 1
                    logger.debug(f"KV cache miss for sequence {seq_id}")
            else:
                cached_keys_list.append(None)
                cached_values_list.append(None)
        
        if cache_hits + cache_misses > 0:
            hit_rate = cache_hits / (cache_hits + cache_misses)
            logger.debug(f"Batch cache hit rate: {hit_rate:.2%} ({cache_hits}/{cache_hits + cache_misses})")
        
        model_kwargs = {}
        past_key_values = self._prepare_past_key_values_for_model(cached_keys_list, cached_values_list)
        if past_key_values:
            model_kwargs["past_key_values"] = past_key_values
            logger.debug(f"Prepared past_key_values with {len(past_key_values)} layers")
        
        with torch.no_grad():
            logger.debug("Running model inference...")
            inference_start = time.time()
            
            if hasattr(self.model, 'forward'):
                input_ids = input_ids.to(self.device)
                logger.debug(f"Input tensor device: {input_ids.device}")
                
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                    logger.debug(f"Added batch dimension, new shape: {input_ids.shape}")
                
                supported_kwargs = {}
                if "past_key_values" in model_kwargs:
                    supported_kwargs["past_key_values"] = model_kwargs["past_key_values"]
                
                outputs = self.model(input_ids, **supported_kwargs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                
                past_key_values = self._extract_past_key_values(outputs)
                if past_key_values:
                    self._update_kv_cache(sequence_ids, past_key_values)
            else:
                input_ids = input_ids.to(self.device)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                logits = self.model(input_ids)
            
            inference_duration = time.time() - inference_start
            logger.debug(f"Model inference completed in {inference_duration:.2f} seconds")
            logger.debug(f"Output logits shape: {logits.shape}")
        
        if logits.dim() > 2:
            logits = logits.squeeze(0)
        
        logger.debug(f"Final logits shape: {logits.shape}")
        
        duration = time.time() - start_time
        logger.debug(f"ModelExecutor.forward completed in {duration:.2f} seconds")
        
        cache_stats = self.kv_cache.get_cache_stats()
        logger.debug(f"KV Cache Stats: {cache_stats}")
        
        return {
            "logits": logits,
            "request_positions": request_positions
        }
    
    def encode(self, text: str) -> List[int]:
        """将文本编码为token ID"""
        return [ord(c) for c in text][:100]
    
    def decode(self, token_ids: List[int]) -> str:
        """将token ID解码为文本"""
        try:
            return ''.join(chr(t) for t in token_ids if 32 <= t <= 126)
        except:
            return str(token_ids)
    
    def _apply_inference_optimizations(self):
        """应用推理优化"""
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        
        if hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = True
        
        torch.set_num_threads(torch.get_num_threads())
        
        cpu_count = os.cpu_count() or 4
        os.environ["OMP_NUM_THREADS"] = str(cpu_count)
        os.environ["MKL_NUM_THREADS"] = str(cpu_count)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)
        
        logger.info("✅ 推理优化设置完成")
    
    def _optimize_model_for_speed(self, model):
        """为速度优化模型"""
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        if hasattr(model, 'config'):
            model.config.use_cache = True
            model.config.output_attentions = False
            model.config.output_hidden_states = False
        
        return model
    
    def optimize_memory(self):
        """内存优化"""
        import gc
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.set_threshold(700, 10, 10)
        
        torch.set_grad_enabled(False)
        
        logger.info("✅ 内存优化完成")


class BatchOptimizedExecutor:
    """批处理优化的执行器"""
    
    def __init__(self, 
                 model_path: str,
                 quantization: str = None,
                 enable_compile: bool = True,
                 warmup_iterations: int = 20,
                 max_batch_size: int = 8,
                 batch_timeout: float = 0.05):
        """
        初始化批处理执行器
        
        Args:
            model_path: 模型路径
            quantization: 量化方法
            enable_compile: 是否启用编译优化
            warmup_iterations: 预热迭代次数
            max_batch_size: 最大批处理大小
            batch_timeout: 批处理超时时间（秒）
        """
        self.executor = ModelExecutor(
            model_path=model_path,
            quantization=quantization,
            use_c_sampler=False,
            enable_compile=enable_compile,
            warmup_iterations=warmup_iterations
        )
        
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.request_queue = Queue()
        self.is_running = False
        self.batch_thread = None
        
        logger.info(f"BatchOptimizedExecutor initialized with max_batch_size={max_batch_size}")
    
    def add_request(self, request_id: str, input_ids: List[int], max_tokens: int = 100):
        """
        添加请求到批处理队列
        
        Args:
            request_id: 请求ID
            input_ids: 输入token IDs
            max_tokens: 最大生成token数
        """
        self.request_queue.put({
            'request_id': request_id,
            'input_ids': input_ids,
            'max_tokens': max_tokens,
            'timestamp': time.time()
        })
        logger.debug(f"Added request {request_id} to batch queue")
    
    async def process_batch(self, batch: List[Dict]) -> List[Dict]:
        """
        处理一批请求
        
        Args:
            batch: 请求列表
            
        Returns:
            处理结果列表
        """
        if not batch:
            return []
        
        logger.info(f"Processing batch of {len(batch)} requests")
        
        # 合并输入
        all_input_ids = []
        request_positions = []
        sequence_ids = []
        
        for i, req in enumerate(batch):
            all_input_ids.extend(req['input_ids'])
            request_positions.append(len(all_input_ids))
            sequence_ids.append(req['request_id'])
        
        # 准备批处理输入
        batch_inputs = {
            'input_ids': all_input_ids,
            'request_positions': request_positions,
            'batch_size': len(batch),
            'sequence_ids': sequence_ids
        }
        
        # 执行前向传播
        outputs = await self.executor.forward(batch_inputs)
        
        # 分割输出
        results = []
        logits = outputs['logits']
        
        for i, req in enumerate(batch):
            # 简化的结果分割（实际实现需要更复杂的逻辑）
            results.append({
                'request_id': req['request_id'],
                'logits': logits,
                'status': 'completed'
            })
        
        return results
    
    def generate_batch(self, batch: List[Dict]) -> Dict[str, List[int]]:
        """
        为一批请求生成tokens
        
        Args:
            batch: 请求列表
            
        Returns:
            生成结果的字典
        """
        results = {}
        processed = self.process_batch(batch)
        
        for result in processed:
            results[result['request_id']] = result['logits']
        
        return results
    
    def start(self):
        """启动批处理线程"""
        if self.is_running:
            logger.warning("Batch thread is already running")
            return
        
        self.is_running = True
        self.batch_thread = threading.Thread(target=self._batch_loop, daemon=True)
        self.batch_thread.start()
        logger.info("Batch processing thread started")
    
    def stop(self):
        """停止批处理线程"""
        self.is_running = False
        if self.batch_thread:
            self.batch_thread.join(timeout=5.0)
        logger.info("Batch processing thread stopped")
    
    def _batch_loop(self):
        """批处理循环"""
        while self.is_running:
            batch = []
            start_time = time.time()
            
            # 收集请求直到达到批大小或超时
            while len(batch) < self.max_batch_size and (time.time() - start_time) < self.batch_timeout:
                try:
                    request = self.request_queue.get(timeout=0.01)
                    batch.append(request)
                except:
                    continue
            
            if batch:
                try:
                    self.process_batch(batch)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")


def create_optimized_executor(model_path: str,
                              quantization: str = None,
                              use_c_sampler: bool = False,
                              enable_compile: bool = True,
                              warmup_iterations: int = 20,
                              enable_batch: bool = False,
                              max_batch_size: int = 8) -> ModelExecutor:
    """
    创建优化执行器的工厂函数
    
    Args:
        model_path: 模型路径
        quantization: 量化方法 ('int8', 'int4', None)
        use_c_sampler: 是否使用C采样器
        enable_compile: 是否启用torch.compile优化
        warmup_iterations: 预热迭代次数
        enable_batch: 是否启用批处理
        max_batch_size: 最大批处理大小
        
    Returns:
        优化执行器实例
    """
    logger.info(f"Creating optimized executor with:")
    logger.info(f"  - Model path: {model_path}")
    logger.info(f"  - Quantization: {quantization}")
    logger.info(f"  - C sampler: {use_c_sampler}")
    logger.info(f"  - Compile: {enable_compile}")
    logger.info(f"  - Warmup iterations: {warmup_iterations}")
    logger.info(f"  - Batch: {enable_batch}")
    
    if enable_batch:
        return BatchOptimizedExecutor(
            model_path=model_path,
            quantization=quantization,
            enable_compile=enable_compile,
            warmup_iterations=warmup_iterations,
            max_batch_size=max_batch_size
        )
    else:
        return ModelExecutor(
            model_path=model_path,
            quantization=quantization,
            use_c_sampler=use_c_sampler,
            enable_compile=enable_compile,
            warmup_iterations=warmup_iterations
        )


if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description="Test optimized model executor")
    parser.add_argument("--model-path", type=str, default="./model/Qwen/Qwen3-0.6B",
                       help="Path to the model")
    parser.add_argument("--quantization", type=str, choices=["int8", "int4", "none"], default="none",
                       help="Quantization method")
    parser.add_argument("--enable-compile", action="store_true", default=True,
                       help="Enable torch.compile optimization")
    parser.add_argument("--warmup-iterations", type=int, default=20,
                       help="Number of warmup iterations")
    
    args = parser.parse_args()
    
    # 创建执行器
    quantization = None if args.quantization == "none" else args.quantization
    executor = create_optimized_executor(
        model_path=args.model_path,
        quantization=quantization,
        enable_compile=args.enable_compile,
        warmup_iterations=args.warmup_iterations
    )
    
    print("✅ 优化执行器创建成功！")
    print(f"配置:")
    print(f"  - 模型路径: {args.model_path}")
    print(f"  - 量化方法: {args.quantization}")
    print(f"  - 编译优化: {'启用' if args.enable_compile else '禁用'}")
    print(f"  - 预热迭代: {args.warmup_iterations}")
