#!/usr/bin/env python3
"""
简单的CPU利用率监控工具
类似于top命令，但更简洁，专门用于监控xLLM服务器的CPU使用情况
"""

import time
import psutil
import sys

def monitor_cpu(interval=1, duration=None):
    """
    监控CPU利用率
    
    Args:
        interval: 采样间隔（秒）
        duration: 监控持续时间（秒），None表示无限监控
    """
    print("CPU利用率监控工具")
    print("按 Ctrl+C 停止监控")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        while True:
            # 获取CPU使用率
            cpu_percent = psutil.cpu_percent(interval=interval, percpu=True)
            total_cpu = psutil.cpu_percent(interval=0.1)
            
            # 获取内存使用情况
            mem = psutil.virtual_memory()
            mem_percent = mem.percent
            mem_used = mem.used / (1024 ** 3)  # GB
            mem_total = mem.total / (1024 ** 3)  # GB
            
            # 获取当前进程信息
            current_pid = psutil.Process()
            proc_cpu = current_pid.cpu_percent(interval=0)
            proc_mem = current_pid.memory_percent()
            
            # 打印时间
            print(f"监控时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 50)
            
            # 打印总CPU和内存使用情况
            print(f"总CPU使用率: {total_cpu:.1f}%")
            print(f"内存使用率: {mem_percent:.1f}% ({mem_used:.1f}GB / {mem_total:.1f}GB)")
            print(f"当前进程: {proc_cpu:.1f}% CPU, {proc_mem:.1f}% 内存")
            print("-" * 50)
            
            # 打印每个CPU核心的使用率
            print("各核心CPU使用率:")
            for i, cpu in enumerate(cpu_percent):
                print(f"核心 {i:2d}: {cpu:.1f}%")
            print("-" * 50)
            print()  # 增加空行分隔不同采样点
            
            # 检查是否达到监控时长
            if duration is not None and (time.time() - start_time) >= duration:
                break
                
    except KeyboardInterrupt:
        print("\n监控已停止")

def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="简单的CPU利用率监控工具")
    parser.add_argument("-i", "--interval", type=float, default=1, help="采样间隔（秒）")
    parser.add_argument("-d", "--duration", type=int, help="监控持续时间（秒）")
    parser.add_argument("-p", "--pid", type=int, help="只监控指定PID的进程")
    
    args = parser.parse_args()
    
    if args.pid:
        monitor_process(args.pid, args.interval, args.duration)
    else:
        monitor_cpu(args.interval, args.duration)

def monitor_process(pid, interval=1, duration=None):
    """
    监控指定PID的进程CPU利用率
    
    Args:
        pid: 进程ID
        interval: 采样间隔（秒）
        duration: 监控持续时间（秒）
    """
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"错误: 进程 {pid} 不存在")
        sys.exit(1)
    
    print(f"监控进程 {pid} ({process.name()})")
    print("按 Ctrl+C 停止监控")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        while True:
            # 获取进程CPU使用率
            cpu_percent = process.cpu_percent(interval=interval)
            
            # 获取进程内存使用情况
            mem_info = process.memory_info()
            mem_percent = process.memory_percent()
            mem_rss = mem_info.rss / (1024 ** 2)  # MB
            
            # 获取进程创建时间
            create_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                      time.localtime(process.create_time()))
            
            # 打印进程信息
            print(f"进程ID: {pid}")
            print(f"进程名称: {process.name()}")
            print(f"创建时间: {create_time}")
            print(f"监控时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 50)
            
            # 打印资源使用情况
            print(f"CPU使用率: {cpu_percent:.1f}%")
            print(f"内存使用率: {mem_percent:.1f}% ({mem_rss:.1f} MB)")
            print("-" * 50)
            print()  # 增加空行分隔不同采样点
            
            # 检查是否达到监控时长
            if duration is not None and (time.time() - start_time) >= duration:
                break
                
    except KeyboardInterrupt:
        print("\n监控已停止")
    except psutil.NoSuchProcess:
        print(f"错误: 进程 {pid} 已终止")
    except Exception as e:
        print(f"监控出错: {e}")

if __name__ == "__main__":
    main()