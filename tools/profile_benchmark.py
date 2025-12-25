#!/usr/bin/env python3
"""
xLLM 基准测试性能分析工具
使用 cProfile 和 py-spy 对 benchmark.py 进行性能分析
"""

import cProfile
import pstats
import subprocess
import time
import os
import signal
import sys
from pathlib import Path

def profile_with_cprofile():
    """使用 cProfile 进行性能分析"""
    print("使用 cProfile 进行性能分析...")
    
    # 创建性能分析器
    profiler = cProfile.Profile()
    
    # 导入并运行基准测试
    profiler.enable()
    
    # 运行基准测试（简化版本，减少请求数量以缩短分析时间）
    from benchmark import BenchmarkTester
    
    # 创建测试器
    tester = BenchmarkTester()
    
    # 检查服务器健康状态
    if not tester.check_server_health():
        print("错误: 无法连接到xLLM服务器，请确保服务器正在运行")
        return
    
    # 测试提示词
    prompts = [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "机器学习是人工智能的一个分支，它使计算机能够在不被明确编程的情况下从数据中学习。"
    ]
    
    # 运行简化的并发测试
    print("运行简化的并发测试进行性能分析...")
    conc_results = tester.run_concurrent_test(5, 30, 3, prompts)
    
    profiler.disable()
    
    # 保存分析结果
    profiler.dump_stats('benchmark_cprofile.prof')
    
    # 打印统计信息
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    print("\n性能分析结果已保存到 benchmark_cprofile.prof")
    print("使用以下命令查看火焰图:")
    print("snakeviz benchmark_cprofile.prof")

def profile_with_pyspy():
    """使用 py-spy 进行性能分析"""
    print("使用 py-spy 进行性能分析...")
    
    # 首先启动基准测试作为一个子进程
    print("启动基准测试进程...")
    
    # 创建一个简化版本的基准测试脚本用于分析
    test_script = """
import sys
sys.path.append('.')
from benchmark import BenchmarkTester

tester = BenchmarkTester()
if tester.check_server_health():
    prompts = [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "机器学习是人工智能的一个分支，它使计算机能够在不被明确编程的情况下从数据中学习。"
    ]
    
    # 运行较长的测试以便 py-spy 采样
    for i in range(10):
        conc_results = tester.run_concurrent_test(3, 50, 3, prompts)
        print(f"完成第 {i+1} 轮测试")
"""
    
    # 将测试脚本写入临时文件
    with open('temp_benchmark.py', 'w') as f:
        f.write(test_script)
    
    # 启动基准测试进程
    benchmark_process = subprocess.Popen([sys.executable, 'temp_benchmark.py'])
    
    try:
        # 等待进程启动
        time.sleep(2)
        
        # 使用 py-spy 采样
        print(f"使用 py-spy 采样进程 {benchmark_process.pid}...")
        pyspy_cmd = [
            'py-spy', 'record', 
            '-o', 'benchmark_flame.svg',
            '--pid', str(benchmark_process.pid)
        ]
        
        # 运行 py-spy 采样 30 秒
        pyspy_process = subprocess.Popen(pyspy_cmd)
        
        # 等待采样完成
        time.sleep(30)
        
        # 终止 py-spy 进程
        pyspy_process.terminate()
        pyspy_process.wait()
        
        print("py-spy 采样完成")
        print("火焰图已保存到 benchmark_flame.svg")
        
    finally:
        # 终止基准测试进程
        benchmark_process.terminate()
        benchmark_process.wait()
        
        # 删除临时文件
        if os.path.exists('temp_benchmark.py'):
            os.remove('temp_benchmark.py')

def main():
    """主函数"""
    print("xLLM 基准测试性能分析工具")
    print("=" * 50)
    
    # 检查所需工具是否可用
    try:
        subprocess.run(['snakeviz', '--help'], capture_output=True, check=True)
        print("✓ snakeviz 可用")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ snakeviz 不可用，请安装: pip install snakeviz")
        return
    
    try:
        subprocess.run(['py-spy', '--help'], capture_output=True, check=True)
        print("✓ py-spy 可用")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ py-spy 不可用，请安装: pip install py-spy")
        return
    
    print("\n选择性能分析方法:")
    print("1. 使用 cProfile 进行性能分析")
    print("2. 使用 py-spy 进行性能分析（需要服务器运行）")
    print("3. 运行两种分析方法")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        profile_with_cprofile()
    elif choice == "2":
        profile_with_pyspy()
    elif choice == "3":
        profile_with_cprofile()
        profile_with_pyspy()
    else:
        print("无效选择")

if __name__ == "__main__":
    main()