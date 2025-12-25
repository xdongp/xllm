#!/usr/bin/env python3
"""
xLLM 基准测试工具
用于测试xLLM在不同并发数和token数量下的性能表现
"""

import requests
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any


class BenchmarkTester:
    """基准测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.generate_url = f"{base_url}/generate"
        self.health_url = f"{base_url}/health"
    
    def check_server_health(self) -> bool:
        """检查服务器健康状态"""
        try:
            response = requests.get(self.health_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def send_request(self, prompt: str, max_tokens: int, temperature: float = 0.7) -> Dict[str, Any]:
        """发送单个生成请求"""
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        start_time = time.time()
        try:
            response = requests.post(
                self.generate_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                # 计算实际生成的token数
                generated_text = result["generated_text"]
                # 简单估算token数（实际应该使用tokenizer计算）
                estimated_tokens = len(generated_text.split())
                
                return {
                    "success": True,
                    "response_time": end_time - start_time,
                    "prompt_tokens": len(prompt.split()),
                    "generated_tokens": estimated_tokens,
                    "total_tokens": len(prompt.split()) + estimated_tokens,
                    "throughput": estimated_tokens / (end_time - start_time) if end_time > start_time else 0,
                    "finish_reason": result["finish_reason"]
                }
            else:
                return {
                    "success": False,
                    "response_time": end_time - start_time,
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "response_time": end_time - start_time,
                "error": str(e)
            }
    
    def run_sequential_test(self, num_requests: int, max_tokens: int, prompts: List[str]) -> List[Dict[str, Any]]:
        """运行顺序性能测试"""
        print(f"运行顺序测试: {num_requests}个请求, 每个请求生成{max_tokens}个token...")
        
        results = []
        start_time = time.time()
        
        for i in range(num_requests):
            prompt = prompts[i % len(prompts)]
            result = self.send_request(prompt, max_tokens)
            results.append(result)
            status = "✓" if result["success"] else "✗"
            print(f"  请求 {i+1}/{num_requests}: {status} {result['response_time']:.2f}秒")
        
        total_time = time.time() - start_time
        
        return results
    
    def run_concurrent_test(self, num_requests: int, max_tokens: int, concurrency: int, 
                          prompts: List[str]) -> List[Dict[str, Any]]:
        """运行并发性能测试"""
        print(f"运行并发测试: {num_requests}个请求, {concurrency}个并发, 每个请求生成{max_tokens}个token...")
        
        results = []
        start_time = time.time()
        
        # 增加线程池大小以支持更高并发
        max_workers = max(concurrency, 10)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有请求
            future_to_index = {
                executor.submit(self.send_request, prompts[i % len(prompts)], max_tokens): i 
                for i in range(num_requests)
            }
            
            # 收集完成的结果
            for future in as_completed(future_to_index):
                result = future.result()
                results.append(result)
                index = future_to_index[future]
                status = "✓" if result["success"] else "✗"
                print(f"  请求 {index+1}/{num_requests}: {status} {result['response_time']:.2f}秒")
        
        total_time = time.time() - start_time
        
        return results
    
    def run_token_count_test(self, max_tokens_list: List[int], concurrency: int, 
                           prompts: List[str]) -> Dict[int, List[Dict[str, Any]]]:
        """运行不同token数量的性能测试"""
        print(f"运行token数量测试: 并发数{concurrency}...")
        
        results = {}
        
        for max_tokens in max_tokens_list:
            print(f"\n测试生成{max_tokens}个token的性能...")
            test_results = self.run_concurrent_test(
                num_requests=5, 
                max_tokens=max_tokens, 
                concurrency=min(concurrency, 5),
                prompts=prompts
            )
            results[max_tokens] = test_results
        
        return results
    
    def calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算统计数据"""
        if not results:
            return {}
        
        successful_results = [r for r in results if r["success"]]
        failed_requests = len(results) - len(successful_results)
        
        if not successful_results:
            return {"failed_requests": failed_requests}
        
        response_times = [r["response_time"] for r in successful_results]
        throughputs = [r["throughput"] for r in successful_results]
        total_tokens = [r["total_tokens"] for r in successful_results]
        generated_tokens = [r["generated_tokens"] for r in successful_results]
        
        return {
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "failed_requests": failed_requests,
            "avg_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "avg_throughput": sum(throughputs) / len(throughputs),
            "total_tokens_processed": sum(total_tokens),
            "avg_generated_tokens": sum(generated_tokens) / len(generated_tokens)
        }
    
    def print_statistics(self, stats: Dict[str, Any], test_name: str):
        """打印统计结果"""
        print(f"\n{test_name}统计结果:")
        print("-" * 50)
        
        if not stats:
            print("  无结果")
            return
        
        if stats.get("failed_requests", 0) == stats.get("total_requests", 0):
            print(f"  所有请求失败: {stats['failed_requests']}个请求")
            return
        
        print(f"  总请求数: {stats.get('total_requests', 0)}")
        print(f"  成功请求数: {stats.get('successful_requests', 0)}")
        print(f"  失败请求数: {stats.get('failed_requests', 0)}")
        print(f"  平均响应时间: {stats.get('avg_response_time', 0):.2f}秒")
        print(f"  最小响应时间: {stats.get('min_response_time', 0):.2f}秒")
        print(f"  最大响应时间: {stats.get('max_response_time', 0):.2f}秒")
        print(f"  平均吞吐量: {stats.get('avg_throughput', 0):.2f} tokens/秒")
        print(f"  总处理token数: {stats.get('total_tokens_processed', 0)}")
        print(f"  平均生成token数: {stats.get('avg_generated_tokens', 0):.2f}")


def main():
    parser = argparse.ArgumentParser(description="xLLM 基准测试工具")
    parser.add_argument("--url", default="http://localhost:8000", help="xLLM服务器地址")
    parser.add_argument("--test-type", choices=["sequential", "concurrent", "token-count", "all"], 
                       default="all", help="测试类型")
    parser.add_argument("--requests", type=int, default=20, help="请求数量")
    parser.add_argument("--concurrency", type=int, default=10, help="并发数")
    parser.add_argument("--max-tokens", type=int, default=50, help="最大生成token数")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = BenchmarkTester(args.url)
    
    # 检查服务器健康状态
    if not tester.check_server_health():
        print("错误: 无法连接到xLLM服务器，请确保服务器正在运行")
        return
    
    # 测试提示词
    prompts = [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "机器学习是人工智能的一个分支，它使计算机能够在不被明确编程的情况下从数据中学习。",
        "深度学习是机器学习的一个子集，它模仿人脑的工作方式来学习数据中的模式。",
        "自然语言处理是人工智能领域中的一个重要方向，它致力于让计算机理解和生成人类语言。",
        "计算机视觉是人工智能的一个重要应用领域，旨在让计算机能够像人类一样理解和解释图像和视频。"
    ]
    
    print("xLLM 基准测试工具")
    print("=" * 50)
    print(f"服务器地址: {args.url}")
    print()
    
    if args.test_type in ["sequential", "all"]:
        # 顺序测试
        seq_results = tester.run_sequential_test(args.requests, args.max_tokens, prompts)
        seq_stats = tester.calculate_statistics(seq_results)
        tester.print_statistics(seq_stats, "顺序测试")
    
    if args.test_type in ["concurrent", "all"]:
        # 并发测试
        conc_results = tester.run_concurrent_test(args.requests, args.max_tokens, args.concurrency, prompts)
        conc_stats = tester.calculate_statistics(conc_results)
        tester.print_statistics(conc_stats, "并发测试")
    
    if args.test_type in ["token-count", "all"]:
        # 不同token数量测试
        token_counts = [10, 25, 50, 100, 200]
        token_results = tester.run_token_count_test(token_counts, args.concurrency, prompts)
        
        print("\nToken数量测试汇总:")
        print("-" * 50)
        for token_count, results in token_results.items():
            stats = tester.calculate_statistics(results)
            if stats.get("successful_requests", 0) > 0:
                print(f"  生成{token_count:3d}个token: 平均响应时间 {stats.get('avg_response_time', 0):.2f}秒, "
                      f"平均吞吐量 {stats.get('avg_throughput', 0):.2f} tokens/秒")
            else:
                print(f"  生成{token_count:3d}个token: 测试失败")


if __name__ == "__main__":
    main()
