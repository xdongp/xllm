"""
优化的采样器实现，包含多种优化策略以提高采样速度
"""
import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from scipy.special import softmax
import threading

logger = logging.getLogger(__name__)

class OptimizedSampler:
    """优化的采样器实现"""
    
    def __init__(self):
        # 优化参数
        self.greedy_threshold = 0.15  # 贪婪采样的温度阈值
        self.fast_topk_threshold = 3  # 快速Top-K采样的阈值
        self.vocab_limit = 50  # 限制词汇表采样的阈值
        
        # 性能统计
        self.stats = {
            'total_samples': 0,
            'greedy_samples': 0,
            'fast_topk_samples': 0,
            'temperature_samples': 0,
            'topp_samples': 0,
            'limited_vocab_samples': 0
        }
        
        # 优化配置
        self.optimization_config = {
            'enable_fast_greedy': True,
            'enable_fast_topk': True,
            'enable_vocab_limiting': True,
            'enable_batch_optimization': True
        }
        
        # 线程锁
        self.lock = threading.Lock()
        
        logger.info("✅ 优化采样器初始化完成")
    
    def sample(self, logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0, **kwargs) -> torch.Tensor:
        """
        优化的采样方法，根据参数自动选择最快策略
        
        Args:
            logits: 模型输出的logits
            temperature: 温度参数
            top_k: Top-K采样参数
            top_p: Top-P采样参数
            
        Returns:
            采样结果
        """
        with self.lock:
            self.stats['total_samples'] += 1
            
            # 根据参数选择最优采样策略
            if temperature < self.greedy_threshold:
                # 激进贪婪采样
                self.stats['greedy_samples'] += 1
                return self._sample_greedy_ultra_fast(logits)
            elif top_k > 0 and top_k <= self.fast_topk_threshold:
                # 快速Top-K采样
                self.stats['fast_topk_samples'] += 1
                return self._sample_topk_ultra_fast(logits, top_k)
            elif top_p > 0 and top_p < 1.0:
                # Top-P采样
                self.stats['topp_samples'] += 1
                return self._sample_topp_optimized(logits, top_p, temperature)
            elif temperature != 1.0:
                # 温度采样
                self.stats['temperature_samples'] += 1
                return self._sample_temperature_optimized(logits, temperature)
            else:
                # 标准采样
                return self._sample_standard(logits)
    
    def _sample_greedy_ultra_fast(self, logits: torch.Tensor) -> torch.Tensor:
        """超快贪婪采样"""
        return torch.argmax(logits, dim=-1)
    
    def _sample_topk_ultra_fast(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """超快Top-K采样"""
        # 获取Top-K的索引
        top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
        
        # 在Top-K中进行采样
        probs = torch.softmax(top_k_values, dim=-1)
        sampled_indices = torch.multinomial(probs, 1)
        
        # 映射回原始词汇表索引
        return top_k_indices.gather(-1, sampled_indices).squeeze(-1)
    
    def _sample_temperature_optimized(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """优化的温度采样"""
        # 应用温度
        scaled_logits = logits / temperature
        
        # 使用Gumbel-max技巧进行采样（更快）
        noise = torch.zeros_like(scaled_logits).uniform_(1e-6, 1.0 - 1e-6)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(scaled_logits + gumbel_noise, dim=-1)
    
    def _sample_topp_optimized(self, logits: torch.Tensor, top_p: float, temperature: float = 1.0) -> torch.Tensor:
        """优化的Top-P采样"""
        # 应用温度
        if temperature != 1.0:
            logits = logits / temperature
        
        # 计算softmax概率
        probs = torch.softmax(logits, dim=-1)
        
        # 按概率排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找到累积概率超过top_p的索引
        mask = cumulative_probs <= top_p
        
        # 确保至少有一个元素被选中
        mask[..., 0] = True
        
        # 创建掩码
        filtered_probs = sorted_probs * mask.float()
        
        # 重新归一化
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        
        # 采样
        sampled_indices = torch.multinomial(filtered_probs, 1)
        
        # 映射回原始索引
        return sorted_indices.gather(-1, sampled_indices).squeeze(-1)
    
    def _sample_standard(self, logits: torch.Tensor) -> torch.Tensor:
        """标准采样"""
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)
    
    def sample_batch_optimized(self, logits_batch: torch.Tensor, temperatures: List[float], 
                              top_ks: Optional[List[int]] = None, 
                              top_ps: Optional[List[float]] = None) -> torch.Tensor:
        """
        优化的批量采样，对相同参数的序列进行分组处理
        
        Args:
            logits_batch: 批量logits (batch_size, vocab_size)
            temperatures: 每个序列的温度参数
            top_ks: 每个序列的Top-K参数
            top_ps: 每个序列的Top-P参数
            
        Returns:
            批量采样结果
        """
        batch_size = logits_batch.shape[0]
        
        # 如果所有参数都相同，使用向量化操作
        if (len(set(temperatures)) == 1 and 
            (top_ks is None or len(set(top_ks)) == 1) and
            (top_ps is None or len(set(top_ps)) == 1)):
            
            # 使用统一参数进行批量采样
            temp = temperatures[0]
            top_k = top_ks[0] if top_ks else 0
            top_p = top_ps[0] if top_ps else 0.0
            
            return self._sample_batch_uniform(logits_batch, temp, top_k, top_p)
        
        # 否则按参数分组进行优化采样
        results = torch.zeros(batch_size, dtype=torch.long, device=logits_batch.device)
        
        # 按参数分组
        param_groups = {}
        for i, (temp, tk, tp) in enumerate(zip(temperatures, 
                                             top_ks or [0]*batch_size, 
                                             top_ps or [0.0]*batch_size)):
            key = (temp, tk, tp)
            if key not in param_groups:
                param_groups[key] = []
            param_groups[key].append(i)
        
        # 对每组使用相同的采样参数
        for (temp, top_k, top_p), indices in param_groups.items():
            group_logits = logits_batch[indices]
            
            # 根据参数选择采样方法
            if temp < self.greedy_threshold:
                group_results = self._sample_greedy_ultra_fast(group_logits)
            elif top_k > 0 and top_k <= self.fast_topk_threshold:
                group_results = self._sample_topk_ultra_fast(group_logits, top_k)
            elif top_p > 0 and top_p < 1.0:
                group_results = self._sample_topp_optimized(group_logits, top_p, temp)
            elif temp != 1.0:
                group_results = self._sample_temperature_optimized(group_logits, temp)
            else:
                group_results = self._sample_standard(group_logits)
            
            # 将结果放回对应位置
            for idx, result in zip(indices, group_results):
                results[idx] = result
        
        return results
    
    def _sample_batch_uniform(self, logits_batch: torch.Tensor, temperature: float, 
                            top_k: int, top_p: float) -> torch.Tensor:
        """统一参数的批量采样"""
        if temperature < self.greedy_threshold:
            return torch.argmax(logits_batch, dim=-1)
        elif top_k > 0 and top_k <= self.fast_topk_threshold:
            return self._sample_topk_ultra_fast(logits_batch, top_k)
        elif top_p > 0 and top_p < 1.0:
            return self._sample_topp_optimized(logits_batch, top_p, temperature)
        elif temperature != 1.0:
            return self._sample_temperature_optimized(logits_batch, temperature)
        else:
            return self._sample_standard(logits_batch)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        total = self.stats['total_samples']
        if total == 0:
            return self.stats.copy()
        
        stats_with_percentages = self.stats.copy()
        stats_with_percentages['greedy_percentage'] = (self.stats['greedy_samples'] / total) * 100
        stats_with_percentages['fast_topk_percentage'] = (self.stats['fast_topk_samples'] / total) * 100
        stats_with_percentages['temperature_percentage'] = (self.stats['temperature_samples'] / total) * 100
        stats_with_percentages['topp_percentage'] = (self.stats['topp_samples'] / total) * 100
        
        return stats_with_percentages
    
    def update_optimization_config(self, **kwargs):
        """更新优化配置"""
        for key, value in kwargs.items():
            if key in self.optimization_config:
                self.optimization_config[key] = value
                logger.info(f"Updated optimization config: {key} = {value}")
    
    def reset_stats(self):
        """重置性能统计"""
        for key in self.stats:
            self.stats[key] = 0
        logger.info("Performance stats reset")


class SamplingOptimizer:
    """采样优化器，提供不同的优化策略"""
    
    def __init__(self):
        self.strategies = {
            'speed': {
                'greedy_threshold': 0.15,
                'fast_topk_threshold': 3,
                'vocab_limit': 50
            },
            'quality': {
                'greedy_threshold': 0.05,
                'fast_topk_threshold': 5,
                'vocab_limit': 100
            },
            'balanced': {
                'greedy_threshold': 0.1,
                'fast_topk_threshold': 4,
                'vocab_limit': 75
            }
        }
    
    def apply_strategy(self, sampler: OptimizedSampler, strategy: str = 'speed'):
        """应用优化策略"""
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        config = self.strategies[strategy]
        sampler.greedy_threshold = config['greedy_threshold']
        sampler.fast_topk_threshold = config['fast_topk_threshold']
        sampler.vocab_limit = config['vocab_limit']
        
        logger.info(f"Applied {strategy} optimization strategy")


def benchmark_optimized_sampler():
    """基准测试优化采样器"""
    import time
    
    logger.info("Benchmarking optimized sampler...")
    
    # 创建采样器
    sampler = OptimizedSampler()
    optimizer = SamplingOptimizer()
    
    # 测试数据
    batch_size = 32
    vocab_size = 32000
    logits = torch.randn(batch_size, vocab_size)
    
    # 测试不同策略
    strategies = ['speed', 'balanced', 'quality']
    
    for strategy in strategies:
        logger.info(f"Testing {strategy} strategy...")
        optimizer.apply_strategy(sampler, strategy)
        
        # 预热
        for _ in range(10):
            _ = sampler.sample(logits[0], temperature=0.7)
        
        # 基准测试
        start_time = time.time()
        for _ in range(100):
            _ = sampler.sample(logits[0], temperature=0.7)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        logger.info(f"{strategy} strategy avg time: {avg_time*1000:.2f}ms per sample")
    
    # 批量采样测试
    logger.info("Testing batch sampling...")
    temperatures = [0.7] * batch_size
    top_ks = [0] * batch_size
    top_ps = [0.9] * batch_size
    
    # 预热
    for _ in range(10):
        _ = sampler.sample_batch_optimized(logits, temperatures, top_ks, top_ps)
    
    # 基准测试
    start_time = time.time()
    for _ in range(50):
        _ = sampler.sample_batch_optimized(logits, temperatures, top_ks, top_ps)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 50
    logger.info(f"Batch sampling avg time: {avg_time*1000:.2f}ms per batch")
    
    # 显示性能统计
    stats = sampler.get_performance_stats()
    logger.info(f"Performance stats: {stats}")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 运行基准测试
    benchmark_optimized_sampler()