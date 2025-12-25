#!/usr/bin/env python3
"""
xLLM 测试脚本，包含性能分析功能
"""

import cProfile
import pstats
import io
import sys
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def test_single_request():
    """测试单个请求"""
    url = "http://localhost:8000/generate"
    headers = {"Content-Type": "application/json"}
    
    # 测试数据
    test_cases = [
        {
            "prompt": "Hello, how are you?",
            "max_tokens": 50,
            "temperature": 0.7
        },
        {
            "prompt": "Explain machine learning in simple terms",
            "max_tokens": 50,
            "temperature": 0.7
        },
        {
            "prompt": "Write a short poem about technology",
            "max_tokens": 30,
            "temperature": 0.8
        }
    ]
    
    results = []
    
    for i, payload in enumerate(test_cases):
        print(f"测试请求 {i+1}: {payload['prompt'][:30]}...")
        start_time = time.time()
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                results.append({
                    "success": True,
                    "response_time": end_time - start_time,
                    "prompt": payload["prompt"],
                    "generated_text_length": len(result.get("generated_text", "")),
                    "request_id": result.get("request_id", "")
                })
                print(f"  ✓ 成功: {end_time - start_time:.2f}秒")
            else:
                results.append({
                    "success": False,
                    "response_time": end_time - start_time,
                    "error": f"HTTP {response.status_code}",
                    "prompt": payload["prompt"]
                })
                print(f"  ✗ 失败: HTTP {response.status_code}")
        except Exception as e:
            end_time = time.time()
            results.append({
                "success": False,
                "response_time": end_time - start_time,
                "error": str(e),
                "prompt": payload["prompt"]
            })
            print(f"  ✗ 错误: {str(e)}")
    
    return results

def test_concurrent_requests(num_requests=5, concurrency=3):
    """测试并发请求"""
    print(f"\n并发测试: {num_requests}个请求, {concurrency}个并发")
    
    url = "http://localhost:8000/generate"
    headers = {"Content-Type": "application/json"}
    
    # 测试数据
    payloads = [
        {"prompt": "Hello, how are you?", "max_tokens": 30, "temperature": 0.7},
        {"prompt": "Explain machine learning in simple terms", "max_tokens": 30, "temperature": 0.7},
        {"prompt": "Write a short poem about technology", "max_tokens": 20, "temperature": 0.8},
        {"prompt": "What is the weather like today?", "max_tokens": 25, "temperature": 0.7},
        {"prompt": "Tell me a joke", "max_tokens": 35, "temperature": 0.9}
    ]
    
    def send_request(payload):
        start_time = time.time()
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response_time": end_time - start_time,
                    "prompt": payload["prompt"],
                    "generated_text_length": len(result.get("generated_text", ""))
                }
            else:
                return {
                    "success": False,
                    "response_time": end_time - start_time,
                    "error": f"HTTP {response.status_code}",
                    "prompt": payload["prompt"]
                }
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "response_time": end_time - start_time,
                "error": str(e),
                "prompt": payload["prompt"]
            }
    
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # 提交所有请求
        future_to_index = {
            executor.submit(send_request, payloads[i % len(payloads)]): i 
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
    print(f"并发测试完成，总耗时: {total_time:.2f}秒")
    
    return results

def print_statistics(results):
    """打印统计结果"""
    if not results:
        print("无结果")
        return
    
    successful_results = [r for r in results if r["success"]]
    failed_requests = len(results) - len(successful_results)
    
    print(f"\n统计结果:")
    print("-" * 50)
    print(f"  总请求数: {len(results)}")
    print(f"  成功请求数: {len(successful_results)}")
    print(f"  失败请求数: {failed_requests}")
    
    if successful_results:
        response_times = [r["response_time"] for r in successful_results]
        avg_response_time = sum(response_times) / len(response_times)
        print(f"  平均响应时间: {avg_response_time:.2f}秒")
        print(f"  最小响应时间: {min(response_times):.2f}秒")
        print(f"  最大响应时间: {max(response_times):.2f}秒")
        
        total_text_length = sum(r["generated_text_length"] for r in successful_results)
        avg_text_length = total_text_length / len(successful_results)
        print(f"  平均生成文本长度: {avg_text_length:.1f} 字符")

def main():
    """主函数"""
    print("xLLM 测试与性能分析工具")
    print("=" * 50)
    
    # 检查服务器是否运行
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code != 200:
            print("错误: xLLM 服务器未运行或不可访问")
            return
    except Exception as e:
        print(f"错误: 无法连接到 xLLM 服务器: {e}")
        return
    
    print("✓ 已连接到 xLLM 服务器")
    
    # 创建性能分析器
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 运行测试
    print("\n1. 运行单请求测试...")
    single_results = test_single_request()
    
    print("\n2. 运行并发测试...")
    concurrent_results = test_concurrent_requests(5, 3)
    
    # 停止性能分析
    profiler.disable()
    
    # 保存分析结果
    profiler.dump_stats('xllm_test_profile.prof')
    
    # 打印统计信息
    print("\n3. 性能分析结果:")
    print("-" * 50)
    
    # 显示性能分析摘要
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(20)
    
    # 读取并显示前几行
    stats_output = s.getvalue()
    lines = stats_output.split('\n')
    for line in lines[:30]:  # 显示前30行
        print(line)
    
    print(f"\n性能分析结果已保存到 xllm_test_profile.prof")
    print("使用以下命令查看火焰图:")
    print("snakeviz xllm_test_profile.prof")
    
    # 打印测试统计
    print("\n4. 测试统计:")
    all_results = single_results + concurrent_results
    print_statistics(all_results)

if __name__ == "__main__":
    main()