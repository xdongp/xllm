"""
C语言与Python采样器性能对比测试
对比sampler.c和sampler.py的性能差异
"""
import numpy as np
import torch
import time
import sys
import os
from typing import Dict, List, Tuple
import json

# 导入C采样器
try:
    from sampler_c import CSampler, create_sampler
    C_AVAILABLE = True
except Exception as e:
    print(f"警告: 无法导入C采样器: {e}")
    C_AVAILABLE = False

# 导入Python采样器
try:
    from sampler import Sampler
    PYTHON_AVAILABLE = True
except Exception as e:
    print(f"警告: 无法导入Python采样器: {e}")
    PYTHON_AVAILABLE = False


class PerformanceBenchmark:
    """性能基准测试类"""
    
    def __init__(self, vocab_size: int = 100, num_iterations: int = 1000):
        """
        初始化基准测试
        
        Args:
            vocab_size: 词汇表大小
            num_iterations: 每个测试的迭代次数
        """
        self.vocab_size = vocab_size
        self.num_iterations = num_iterations
        self.results = {}
        
        # 生成测试数据（固定随机种子以确保公平）
        np.random.seed(42)
        self.test_logits = []
        for _ in range(num_iterations):
            logits = np.random.randn(vocab_size).astype(np.float32)
            self.test_logits.append(logits)
    
    def run_test(self, sampler, method_name: str, **kwargs) -> Dict:
        """
        运行单个测试
        
        Args:
            sampler: 采样器实例
            method_name: 方法名（如'sample_greedy', 'sample_temperature'等）
            **kwargs: 方法参数
            
        Returns:
            测试结果字典
        """
        # 重置统计
        if hasattr(sampler, 'reset_stats'):
            sampler.reset_stats()
        
        # 预热
        for _ in range(10):
            logits = np.random.randn(self.vocab_size).astype(np.float32)
            method = getattr(sampler, method_name)
            method(logits, **kwargs)
        
        # 重置统计
        if hasattr(sampler, 'reset_stats'):
            sampler.reset_stats()
        
        # 开始计时
        start_time = time.time()
        
        # 运行测试
        for i in range(self.num_iterations):
            logits = self.test_logits[i]
            method = getattr(sampler, method_name)
            method(logits, **kwargs)
        
        # 结束计时
        end_time = time.time()
        
        # 获取统计
        if hasattr(sampler, 'get_stats'):
            stats = sampler.get_stats()
        else:
            stats = {
                'total_samples': self.num_iterations,
                'total_time': end_time - start_time,
                'average_sample_time': (end_time - start_time) / self.num_iterations,
                'samples_per_second': self.num_iterations / (end_time - start_time),
            }
        
        return {
            'method': method_name,
            'total_time': end_time - start_time,
            'average_time': stats['average_sample_time'],
            'throughput': stats['samples_per_second'],
            'num_samples': self.num_iterations,
            'params': kwargs,
        }
    
    def compare_implementations(self) -> Dict:
        """
        对比C和Python实现
        
        Returns:
            对比结果
        """
        results = {
            'vocab_size': self.vocab_size,
            'num_iterations': self.num_iterations,
            'c_implementation': {},
            'python_implementation': {},
            'comparison': {},
        }
        
        # 测试配置（C和Python使用不同的接口）
        test_configs = [
            ('C', 'sample_greedy', {}),
            ('C', 'sample_temperature', {'temperature': 0.8}),
            ('C', 'sample_temperature', {'temperature': 1.2}),
            ('C', 'sample_topk', {'top_k': 10, 'temperature': 1.0}),
            ('C', 'sample_topk', {'top_k': 50, 'temperature': 1.0}),
            ('C', 'sample_topp', {'top_p': 0.9, 'temperature': 1.0}),
            ('C', 'sample_topp', {'top_p': 0.95, 'temperature': 1.0}),
            ('C', 'sample', {'temperature': 0.5, 'top_k': 10, 'top_p': 0.9}),  # 自适应采样
            ('Python', 'sample', {'temperature': 0.01, 'top_k': 50, 'top_p': 0.9}),  # 近似贪婪
            ('Python', 'sample', {'temperature': 0.8, 'top_k': 50, 'top_p': 0.9}),
            ('Python', 'sample', {'temperature': 1.2, 'top_k': 50, 'top_p': 0.9}),
            ('Python', 'sample', {'temperature': 1.0, 'top_k': 10, 'top_p': 0.9}),
            ('Python', 'sample', {'temperature': 1.0, 'top_k': 50, 'top_p': 0.9}),
            ('Python', 'sample', {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}),
        ]
        
        # 定义对比映射（C方法 -> Python方法）
        comparison_mapping = [
            ('sample_greedy', 'sample', {'temperature': 0.01, 'top_k': 50, 'top_p': 0.9}),
            ('sample_temperature', 'sample', {'temperature': 0.8, 'top_k': 50, 'top_p': 0.9}),
            ('sample_temperature', 'sample', {'temperature': 1.2, 'top_k': 50, 'top_p': 0.9}),
            ('sample_topk', 'sample', {'temperature': 1.0, 'top_k': 10, 'top_p': 0.9}),
            ('sample_topk', 'sample', {'temperature': 1.0, 'top_k': 50, 'top_p': 0.9}),
            ('sample_topp', 'sample', {'temperature': 1.0, 'top_k': 50, 'top_p': 0.9}),
            ('sample_topp', 'sample', {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}),
            ('sample', 'sample', {'temperature': 0.5, 'top_k': 10, 'top_p': 0.9}),
        ]
        
        # 测试C实现
        if C_AVAILABLE:
            print("\n" + "="*60)
            print("测试C语言实现...")
            print("="*60)
            
            c_sampler = create_sampler(self.vocab_size)
            
            for impl_type, method_name, params in test_configs:
                if impl_type == 'C':
                    print(f"\n测试 {method_name} (参数: {params})...")
                    result = self.run_test(c_sampler, method_name, **params)
                    results['c_implementation'][method_name] = result
                    print(f"  平均时间: {result['average_time']*1000:.4f} ms")
                    print(f"  吞吐量: {result['throughput']:.2f} tokens/秒")
            
            del c_sampler
        
        # 测试Python实现
        if PYTHON_AVAILABLE:
            print("\n" + "="*60)
            print("测试Python实现...")
            print("="*60)
            
            py_sampler = Sampler(self.vocab_size)
            
            for impl_type, method_name, params in test_configs:
                if impl_type == 'Python':
                    # Python使用torch.Tensor
                    print(f"\n测试 {method_name} (参数: {params})...")
                    
                    # 重置统计
                    if hasattr(py_sampler, 'reset_stats'):
                        py_sampler.reset_stats()
                    
                    # 预热
                    for _ in range(10):
                        logits = torch.from_numpy(np.random.randn(self.vocab_size).astype(np.float32))
                        method = getattr(py_sampler, method_name)
                        method(logits, **params)
                    
                    # 重置统计
                    if hasattr(py_sampler, 'reset_stats'):
                        py_sampler.reset_stats()
                    
                    # 开始计时
                    start_time = time.time()
                    
                    # 运行测试
                    for i in range(self.num_iterations):
                        logits = torch.from_numpy(self.test_logits[i])
                        method = getattr(py_sampler, method_name)
                        method(logits, **params)
                    
                    # 结束计时
                    end_time = time.time()
                    
                    # 获取统计
                    stats = py_sampler.stats
                    # 使用参数作为键的一部分，确保每个参数组合都有独立的结果
                    result_key = f"{method_name}_{hash(frozenset(params.items()))}"
                    result = {
                        'method': method_name,
                        'total_time': end_time - start_time,
                        'average_time': (end_time - start_time) / self.num_iterations,
                        'throughput': self.num_iterations / (end_time - start_time),
                        'num_samples': self.num_iterations,
                        'params': params,
                    }
                    
                    results['python_implementation'][result_key] = result
                    print(f"  平均时间: {result['average_time']*1000:.4f} ms")
                    print(f"  吞吐量: {result['throughput']:.2f} tokens/秒")
            
            del py_sampler
        
        # 计算对比（根据采样类型进行对比）
        if C_AVAILABLE and PYTHON_AVAILABLE:
            print("\n" + "="*60)
            print("性能对比分析...")
            print("="*60)
            
            for c_method, py_method, params in comparison_mapping:
                # 查找对应的Python结果
                py_key = None
                for key, result in results['python_implementation'].items():
                    if result['params'] == params:
                        py_key = key
                        break
                
                if c_method in results['c_implementation'] and py_key:
                    c_result = results['c_implementation'][c_method]
                    py_result = results['python_implementation'][py_key]
                    
                    speedup = py_result['average_time'] / c_result['average_time']
                    improvement = ((py_result['average_time'] - c_result['average_time']) / py_result['average_time']) * 100
                    
                    comparison_key = f"{c_method}_vs_{py_method}"
                    results['comparison'][comparison_key] = {
                        'c_method': c_method,
                        'py_method': py_method,
                        'params': params,
                        'speedup': speedup,
                        'improvement_percent': improvement,
                        'c_time_ms': c_result['average_time'] * 1000,
                        'py_time_ms': py_result['average_time'] * 1000,
                    }
                    
                    print(f"\n{c_method} vs {py_method} (参数: {params}):")
                    print(f"  C实现: {c_result['average_time']*1000:.4f} ms")
                    print(f"  Python实现: {py_result['average_time']*1000:.4f} ms")
                    print(f"  加速比: {speedup:.2f}x")
                    print(f"  性能提升: {improvement:.2f}%")
        
        return results
    
    def print_summary(self, results: Dict):
        """打印测试摘要"""
        print("\n" + "="*60)
        print("性能对比摘要")
        print("="*60)
        
        print(f"\n测试配置:")
        print(f"  词汇表大小: {results['vocab_size']}")
        print(f"  迭代次数: {results['num_iterations']}")
        
        if 'comparison' in results and results['comparison']:
            print(f"\n性能对比:")
            
            avg_speedup = 0
            avg_improvement = 0
            count = 0
            
            for method_name, comparison in results['comparison'].items():
                print(f"\n{method_name}:")
                print(f"  加速比: {comparison['speedup']:.2f}x")
                print(f"  性能提升: {comparison['improvement_percent']:.2f}%")
                
                avg_speedup += comparison['speedup']
                avg_improvement += comparison['improvement_percent']
                count += 1
            
            if count > 0:
                avg_speedup /= count
                avg_improvement /= count
                
                print(f"\n{'='*60}")
                print(f"平均加速比: {avg_speedup:.2f}x")
                print(f"平均性能提升: {avg_improvement:.2f}%")
                print(f"{'='*60}")
        
        # 结论
        print(f"\n结论:")
        if C_AVAILABLE and PYTHON_AVAILABLE:
            avg_speedup = np.mean([c['speedup'] for c in results['comparison'].values()])
            if avg_speedup > 1.2:
                print(f"  ✓ C语言实现显著优于Python实现（平均加速{avg_speedup:.2f}x）")
            elif avg_speedup > 1.0:
                print(f"  ✓ C语言实现略优于Python实现（平均加速{avg_speedup:.2f}x）")
            else:
                print(f"  ✗ C语言实现未达到预期性能（平均加速{avg_speedup:.2f}x）")
        elif C_AVAILABLE:
            print(f"  ✓ 仅测试了C语言实现")
        elif PYTHON_AVAILABLE:
            print(f"  ✓ 仅测试了Python实现")
        else:
            print(f"  ✗ 无可用的实现进行测试")
    
    def save_results(self, results: Dict, filename: str = 'benchmark_results.json'):
        """保存结果到JSON文件"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n结果已保存到: {filename}")


def main():
    """主函数"""
    print("="*60)
    print("C语言 vs Python采样器性能对比测试")
    print("="*60)
    
    # 检查可用性
    if not C_AVAILABLE and not PYTHON_AVAILABLE:
        print("\n错误: 没有可用的采样器实现！")
        print("请确保:")
        print("  1. C库已编译: gcc -shared -fPIC -o libsampler.dylib sampler.c -lm")
        print("  2. Python采样器文件存在: sampler.py")
        return
    
    # 创建基准测试
    benchmark = PerformanceBenchmark(
        vocab_size=100,
        num_iterations=10000
    )
    
    # 运行对比测试
    results = benchmark.compare_implementations()
    
    # 打印摘要
    benchmark.print_summary(results)
    
    # 保存结果
    benchmark.save_results(results, 'sampler_benchmark_results.json')
    
    print("\n测试完成!")


if __name__ == "__main__":
    main()
