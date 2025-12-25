#!/usr/bin/env python3
"""
KV缓存分析工具
用于分析和优化KV缓存性能
"""

import requests
import json
import time
import argparse
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np

class CacheAnalyzer:
    """KV缓存分析器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.cache_stats_url = f"{base_url}/cache-stats"
        self.generate_url = f"{base_url}/generate"
        self.health_url = f"{base_url}/health"
    
    def check_server_health(self) -> bool:
        """检查服务器健康状态"""
        try:
            response = requests.get(self.health_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            response = requests.get(self.cache_stats_url, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def send_request(self, prompt: str, max_tokens: int = 50) -> Dict[str, Any]:
        """发送生成请求"""
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7
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
                return {
                    "success": True,
                    "response_time": end_time - start_time,
                    "generated_text": result.get("generated_text", ""),
                    "request_id": result.get("request_id", "")
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
    
    def run_cache_efficiency_test(self, num_requests: int = 10) -> List[Dict[str, Any]]:
        """运行缓存效率测试"""
        print(f"运行缓存效率测试: {num_requests}个请求...")
        
        # 测试提示词（包含一些重复前缀以测试缓存效果）
        prompts = [
            "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
            "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。机器学习是实现人工智能的一种方法。",
            "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。深度学习是机器学习的一个子集。",
            "机器学习是人工智能的一个分支，它使计算机能够在不被明确编程的情况下从数据中学习。",
            "深度学习是机器学习的一个子集，它模仿人脑的工作方式来学习数据中的模式。",
            "自然语言处理是人工智能领域中的一个重要方向，它致力于让计算机理解和生成人类语言。",
            "计算机视觉是人工智能的一个重要应用领域，旨在让计算机能够像人类一样理解和解释图像和视频。",
            "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。这是人工智能的定义。",
            "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。人工智能有很多应用。",
            "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。未来人工智能会如何发展？"
        ]
        
        results = []
        
        # 获取初始缓存统计
        initial_stats = self.get_cache_stats()
        print(f"初始缓存统计: {initial_stats}")
        
        # 发送请求
        for i in range(num_requests):
            prompt = prompts[i % len(prompts)]
            print(f"发送请求 {i+1}/{num_requests}: {prompt[:50]}...")
            
            result = self.send_request(prompt, 30)
            results.append(result)
            
            # 获取缓存统计
            stats = self.get_cache_stats()
            print(f"  缓存统计: {stats}")
            
            if "hit_rate" in stats:
                print(f"  缓存命中率: {stats['hit_rate']:.2%}")
            
            status = "✓" if result["success"] else "✗"
            print(f"  请求结果: {status} {result['response_time']:.2f}秒")
        
        # 获取最终缓存统计
        final_stats = self.get_cache_stats()
        print(f"最终缓存统计: {final_stats}")
        
        return results, initial_stats, final_stats
    
    def analyze_cache_performance(self, initial_stats: Dict, final_stats: Dict):
        """分析缓存性能"""
        print("\n缓存性能分析:")
        print("=" * 50)
        
        if "error" in initial_stats or "error" in final_stats:
            print("无法获取缓存统计信息")
            return
        
        # 计算缓存命中率改善
        if "hit_rate" in initial_stats and "hit_rate" in final_stats:
            initial_hit_rate = initial_stats["hit_rate"]
            final_hit_rate = final_stats["hit_rate"]
            improvement = final_hit_rate - initial_hit_rate
            
            print(f"初始缓存命中率: {initial_hit_rate:.2%}")
            print(f"最终缓存命中率: {final_hit_rate:.2%}")
            print(f"命中率改善: {improvement:+.2%}")
        
        # 显示缓存使用情况
        if "current_size" in final_stats and "max_size" in final_stats:
            current_size = final_stats["current_size"]
            max_size = final_stats["max_size"]
            usage = current_size / max_size if max_size > 0 else 0
            
            print(f"缓存使用量: {current_size}/{max_size} ({usage:.1%})")
        
        # 显示请求统计
        if "total_hits" in final_stats and "total_misses" in final_stats:
            total_hits = final_stats["total_hits"]
            total_misses = final_stats["total_misses"]
            total_requests = total_hits + total_misses
            
            print(f"总缓存请求: {total_requests}")
            print(f"缓存命中: {total_hits}")
            print(f"缓存未命中: {total_misses}")
    
    def plot_cache_performance(self, initial_stats: Dict, final_stats: Dict):
        """绘制缓存性能图表"""
        if "error" in initial_stats or "error" in final_stats:
            print("无法绘制图表：缺少缓存统计信息")
            return
        
        # 准备数据
        labels = ['初始', '最终']
        hit_rates = [
            initial_stats.get("hit_rate", 0),
            final_stats.get("hit_rate", 0)
        ]
        cache_sizes = [
            initial_stats.get("current_size", 0),
            final_stats.get("current_size", 0)
        ]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 命中率图表
        bars1 = ax1.bar(labels, hit_rates, color=['skyblue', 'lightgreen'])
        ax1.set_ylabel('命中率')
        ax1.set_title('缓存命中率变化')
        ax1.set_ylim(0, 1)
        
        # 在柱状图上添加数值标签
        for bar, rate in zip(bars1, hit_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.2%}', ha='center', va='bottom')
        
        # 缓存大小图表
        bars2 = ax2.bar(labels, cache_sizes, color=['skyblue', 'lightgreen'])
        ax2.set_ylabel('缓存条目数')
        ax2.set_title('缓存使用量变化')
        
        # 在柱状图上添加数值标签
        for bar, size in zip(bars2, cache_sizes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{size}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('cache_performance.png')
        print("缓存性能图表已保存为 cache_performance.png")
        plt.show()
    
    def run_comprehensive_analysis(self, num_requests: int = 10):
        """运行综合分析"""
        print("KV缓存综合分析工具")
        print("=" * 50)
        
        # 检查服务器健康状态
        if not self.check_server_health():
            print("错误: 无法连接到xLLM服务器，请确保服务器正在运行")
            return
        
        print("✓ 已连接到 xLLM 服务器")
        
        # 运行缓存效率测试
        results, initial_stats, final_stats = self.run_cache_efficiency_test(num_requests)
        
        # 分析缓存性能
        self.analyze_cache_performance(initial_stats, final_stats)
        
        # 绘制性能图表
        try:
            self.plot_cache_performance(initial_stats, final_stats)
        except Exception as e:
            print(f"绘制图表时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description="KV缓存分析工具")
    parser.add_argument("--url", default="http://localhost:8000", help="xLLM服务器地址")
    parser.add_argument("--requests", type=int, default=10, help="请求数量")
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = CacheAnalyzer(args.url)
    
    # 运行综合分析
    analyzer.run_comprehensive_analysis(args.requests)

if __name__ == "__main__":
    main()