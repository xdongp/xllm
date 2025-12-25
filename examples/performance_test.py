#!/usr/bin/env python3
"""
xLLM的性能测试示例
"""
import requests
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


def send_request(prompt, max_tokens=50):
    """向xLLM服务器发送单个请求"""
    url = "http://localhost:8080/generate"
    
    payload = {
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    start_time = time.time()
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "response_time": end_time - start_time,
                "generated_tokens": len(result["generated_text"].split()),
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


def sequential_test(num_requests=10):
    """运行顺序性能测试"""
    print(f"运行顺序测试，共 {num_requests} 个请求...")
    
    prompts = [
        "人工智能是",
        "机器学习是",
        "深度学习是",
        "自然语言处理是",
        "计算机视觉是",
        "神经网络是",
        "数据科学是",
        "大数据是",
        "云计算是",
        "区块链是"
    ]
    
    results = []
    start_time = time.time()
    
    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        result = send_request(prompt)
        results.append(result)
        print(f"请求 {i+1}/{num_requests}: {result['response_time']:.2f}秒")
    
    total_time = time.time() - start_time
    
    # 计算统计信息
    successful_requests = sum(1 for r in results if r["success"])
    total_response_time = sum(r["response_time"] for r in results)
    avg_response_time = total_response_time / num_requests if num_requests > 0 else 0
    
    print(f"\n顺序测试结果:")
    print(f"  总时间: {total_time:.2f}秒")
    print(f"  成功请求数: {successful_requests}/{num_requests}")
    print(f"  平均响应时间: {avg_response_time:.2f}秒")
    print(f"  每秒请求数: {num_requests/total_time:.2f}")
    
    return results


def concurrent_test(num_requests=10, max_workers=5):
    """运行并发性能测试"""
    print(f"\n运行并发测试，共 {num_requests} 个请求 ({max_workers} 个工作线程)...")
    
    prompts = [
        "人工智能是",
        "机器学习是",
        "深度学习是",
        "自然语言处理是",
        "计算机视觉是",
        "神经网络是",
        "数据科学是",
        "大数据是",
        "云计算是",
        "区块链是"
    ]
    
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有请求
        future_to_index = {
            executor.submit(send_request, prompts[i % len(prompts)]): i 
            for i in range(num_requests)
        }
        
        # 收集完成的结果
        for future in as_completed(future_to_index):
            result = future.result()
            results.append(result)
            index = future_to_index[future]
            print(f"请求 {index+1}/{num_requests}: {result['response_time']:.2f}秒")
    
    total_time = time.time() - start_time
    
    # 计算统计信息
    successful_requests = sum(1 for r in results if r["success"])
    total_response_time = sum(r["response_time"] for r in results)
    avg_response_time = total_response_time / num_requests if num_requests > 0 else 0
    
    print(f"\n并发测试结果:")
    print(f"  总时间: {total_time:.2f}秒")
    print(f"  成功请求数: {successful_requests}/{num_requests}")
    print(f"  平均响应时间: {avg_response_time:.2f}秒")
    print(f"  每秒请求数: {num_requests/total_time:.2f}")
    
    return results


def stress_test():
    """运行压力测试以评估系统限制"""
    print("\n运行压力测试...")
    
    # 使用递增的并发数进行测试
    concurrency_levels = [1, 2, 4, 8, 16]
    results = []
    
    for concurrency in concurrency_levels:
        print(f"\n测试 {concurrency} 个并发请求...")
        test_results = concurrent_test(num_requests=concurrency, max_workers=concurrency)
        successful = sum(1 for r in test_results if r["success"])
        avg_time = sum(r["response_time"] for r in test_results) / len(test_results) if test_results else 0
        
        results.append({
            "concurrency": concurrency,
            "successful": successful,
            "avg_response_time": avg_time
        })
    
    print("\n压力测试摘要:")
    print("并发数 | 成功请求数 | 平均响应时间")
    print("-------|------------|--------------")
    for result in results:
        print(f"{result['concurrency']:>6} | {result['successful']:>10} | {result['avg_response_time']:>12.2f}秒")


if __name__ == "__main__":
    print("xLLM性能测试")
    print("=" * 30)
    
    # 运行测试
    sequential_results = sequential_test(num_requests=5)
    concurrent_results = concurrent_test(num_requests=5, max_workers=3)
    stress_test()
    
    print("\n性能测试完成!")