#!/usr/bin/env python3
"""
基准测试脚本 - 测试xLLM优化后的性能
测试不同token长度下的推理性能，特别关注超过200个token的情况
"""

import torch
import time
import argparse
import logging
from typing import List
from model_executor_optimized import OptimizedModelExecutor
from sampler import Sampler

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """性能基准测试类 - 与统一基准测试使用相同策略"""
    
    def __init__(self, model_path: str, quantization: str = "fp16"):
        logger.info(f"Initializing benchmark with model: {model_path}, quantization: {quantization}")
        self.executor = OptimizedModelExecutor(
            model_path=model_path,
            quantization=quantization,
            use_c_sampler=True,
            enable_compile=True
        )
        self.sampler = Sampler()
        
        # 预热模型
        logger.info("Warming up model...")
        self._warmup()
    
    def _warmup(self):
        """预热模型"""
        for _ in range(3):
            dummy_input = torch.tensor([[1, 2, 3, 4, 5]])
            self.executor._forward_sync(dummy_input, 0)
    
    def test_token_lengths(self, token_lengths: List[int], max_new_tokens: int = 10, iterations: int = 3):
        """测试不同token长度的性能 - 与统一基准测试相同的策略"""
        logger.info(f"Testing token lengths: {token_lengths}, max_new_tokens={max_new_tokens}")
        
        results = {}
        
        for length in token_lengths:
            logger.info(f"\nTesting input length: {length}")
            
            # 生成测试输入
            test_input = [100 + (i % 1000) for i in range(length)]
            input_tensor = torch.tensor([test_input])
            
            # 预热
            logger.info("Warming up...")
            for _ in range(2):
                self.executor._forward_sync(torch.tensor([[1, 2, 3]]), 0)
            
            # 测试推理
            times = []
            tokens_per_iteration = []
            
            for i in range(iterations):
                logger.info(f"Iteration {i+1}/{iterations}")
                start_time = time.time()
                
                # 使用generate方法生成指定数量的新token
                outputs = self.executor.generate(
                    input_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True
                )
                
                end_time = time.time()
                if isinstance(outputs, torch.Tensor):
                    generated_tokens = len(outputs[0]) - len(test_input)  # 计算新生成的token数
                else:
                    generated_tokens = max_new_tokens  # 假设成功生成了所有token
                
                duration = end_time - start_time
                times.append(duration)
                tokens_per_iteration.append(generated_tokens)
                
                logger.info(f"Generated {generated_tokens} tokens in {duration:.4f} seconds")
            
            # 计算统计数据
            if times:
                avg_time = sum(times) / len(times)
                avg_tokens = sum(tokens_per_iteration) / len(tokens_per_iteration)
                tokens_per_second = sum(tokens_per_iteration) / sum(times) if sum(times) > 0 else 0
            else:
                avg_time = 0
                avg_tokens = 0
                tokens_per_second = 0
            
            results[length] = {
                "avg_time": avg_time,
                "avg_tokens": avg_tokens,
                "tokens_per_second": tokens_per_second,
                "individual_times": times
            }
            
            logger.info(f"\nInput length {length} - Results:")
            logger.info(f"  Average time per iteration: {avg_time:.4f} seconds")
            logger.info(f"  Average tokens generated: {avg_tokens:.1f}")
            logger.info(f"  Tokens per second: {tokens_per_second:.2f}")
        
        return results
    
    def _generate_tokens(self, input_ids: List[int], max_new_tokens: int, temperature: float = 0.7):
        """生成token的辅助方法"""
        current_ids = input_ids[:]
        generated_count = 0
        
        while generated_count < max_new_tokens:
            input_tensor = torch.tensor([current_ids])
            
            # 前向传播获取logits
            logits = self.executor._forward_sync(input_tensor, len(current_ids) - 1)
            
            # 采样下一个token
            next_token = self.sampler.sample(logits[0, -1, :].unsqueeze(0), temperature=temperature)
            next_token_id = next_token.item()
            
            current_ids.append(next_token_id)
            generated_count += 1
        
        return current_ids[len(input_ids):]  # 返回新生成的token
    
    def print_summary(self, results: dict):
        """打印性能摘要 - 与统一基准测试格式一致"""
        logger.info("\nPerformance Summary:")
        logger.info("="*80)
        logger.info(f"{'Input Length':<15} {'Avg Time (s)':<15} {'Tokens/s':<15} {'Avg Gen Tokens':<15}")
        logger.info("-"*80)
        
        for length, data in results.items():
            logger.info(f"{length:<15} {data['avg_time']:<15.4f} {data['tokens_per_second']:<15.2f} {data['avg_tokens']:<15.1f}")
        
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description="xLLM Performance Benchmark (Optimized)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--quantization", type=str, default="fp16", help="Quantization type (fp16, int8, int4)")
    parser.add_argument("--max-new-tokens", type=int, default=10, help="Max new tokens to generate")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations per test")
    
    args = parser.parse_args()
    
    # 创建基准测试器
    benchmark = PerformanceBenchmark(args.model_path, args.quantization)
    
    # 测试不同token长度 - 与统一基准测试相同的长度
    token_lengths = [50, 100, 200, 300, 400, 500]
    
    # 运行基准测试
    results = benchmark.test_token_lengths(
        token_lengths, 
        max_new_tokens=args.max_new_tokens, 
        iterations=args.iterations
    )
    
    # 打印摘要
    benchmark.print_summary(results)


if __name__ == "__main__":
    main()