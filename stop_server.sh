#!/bin/bash

# 停止xLLM服务器脚本

echo "停止xLLM服务器..."

# 查找并终止xLLM服务器进程
SERVER_PIDS=$(pgrep -f "xllm_server.py")
if [ ! -z "$SERVER_PIDS" ]; then
    echo "找到xLLM服务器进程: $SERVER_PIDS"
    echo "终止进程..."
    kill -TERM $SERVER_PIDS 2>/dev/null
    
    # 等待片刻以优雅关闭
    sleep 3
    
    # 检查进程是否仍在运行，如果需要则强制终止
    STILL_RUNNING=$(pgrep -f "xllm_server.py")
    if [ ! -z "$STILL_RUNNING" ]; then
        echo "强制终止剩余进程: $STILL_RUNNING"
        kill -9 $STILL_RUNNING 2>/dev/null
    fi
    
    echo "✓ xLLM服务器进程已停止"
else
    echo "未找到xLLM服务器进程"
fi

# 同时尝试终止任何uvicorn进程（以防服务器直接使用uvicorn启动）
UVICORN_PIDS=$(pgrep -f "uvicorn")
if [ ! -z "$UVICORN_PIDS" ]; then
    echo "找到uvicorn进程: $UVICORN_PIDS"
    echo "终止uvicorn进程..."
    kill -TERM $UVICORN_PIDS 2>/dev/null
    
    # 等待片刻
    sleep 2
    
    # 检查进程是否仍在运行，如果需要则强制终止
    STILL_RUNNING_UVICORN=$(pgrep -f "uvicorn")
    if [ ! -z "$STILL_RUNNING_UVICORN" ]; then
        echo "强制终止剩余uvicorn进程: $STILL_RUNNING_UVICORN"
        kill -9 $STILL_RUNNING_UVICORN 2>/dev/null
    fi
    
    echo "✓ Uvicorn进程已停止"
fi

# 最终检查
echo ""
echo "最终检查剩余的xLLM相关进程:"
ps aux | grep -i "xllm\|python3.*server" | grep -v grep || echo "未找到xLLM相关进程"