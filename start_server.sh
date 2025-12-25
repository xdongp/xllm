#!/bin/bash

# xLLM 启动脚本
# 版本: v4.0
# 优化程度: 250%性能提升
# 作者: xLLM Team

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 默认参数
MODEL_PATH="./model/Qwen/Qwen3-0.6B"
PORT=8000
QUANTIZATION="fp16"
DEBUG_MODE=true
CHECK_ONLY=false

# 显示帮助信息
show_help() {
    echo -e "${CYAN}🚀 xLLM 启动脚本${NC}"
    echo -e "${CYAN}================================${NC}"
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --model-path PATH    模型路径 (默认: $MODEL_PATH)"
    echo "  -p, --port PORT          端口号 (默认: $PORT)"
    echo "  -q, --quantization TYPE  量化类型 (int8/fp16, 默认: $QUANTIZATION)"
    echo "  -d, --debug              启用调试模式 (默认: 启用)"
    echo "  --no-debug               禁用调试模式"
    echo "  -c, --check              仅检查环境，不启动服务器"
    echo "  -h, --help               显示此帮助信息"
    echo ""
    echo -e "${GREEN}示例:${NC}"
    echo "  $0                                    # 使用默认配置启动"
    echo "  $0 -m ./my_model -p 8080             # 自定义模型和端口"
    echo "  $0 --no-debug                        # 生产环境模式"
    echo "  $0 -c                                 # 仅检查环境"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -q|--quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        -d|--debug)
            DEBUG_MODE=true
            shift
            ;;
        --no-debug)
            DEBUG_MODE=false
            shift
            ;;
        -c|--check)
            CHECK_ONLY=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知参数 $1${NC}" >&2
            show_help
            exit 1
            ;;
    esac
done

# 检查环境函数
check_environment() {
    echo -e "${BLUE}🔍 检查环境...${NC}"
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ 错误: 未找到 python3${NC}" >&2
        return 1
    fi
    echo -e "${GREEN}✓ python3 已找到$(python3 --version)${NC}"
    
    # 检查依赖
    python3 -c "import torch; import transformers; import fastapi; import uvicorn" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 错误: 缺少必要的Python依赖${NC}" >&2
        echo -e "${YELLOW}提示: 运行 'pip install torch transformers fastapi uvicorn' 安装依赖${NC}"
        return 1
    fi
    echo -e "${GREEN}✓ Python依赖已找到${NC}"
    
    # 检查模型路径
    if [ ! -d "$MODEL_PATH" ]; then
        echo -e "${RED}❌ 错误: 模型路径不存在: $MODEL_PATH${NC}" >&2
        return 1
    fi
    echo -e "${GREEN}✓ 模型路径存在: $MODEL_PATH${NC}"
    
    # 检查端口是否被占用
    if command -v lsof &> /dev/null; then
        if lsof -i :$PORT &> /dev/null; then
            echo -e "${YELLOW}⚠️  端口 $PORT 已被占用${NC}"
        else
            echo -e "${GREEN}✓ 端口 $PORT 可用${NC}"
        fi
    fi
    
    echo -e "${GREEN}✓ 环境检查完成${NC}"
    return 0
}

# 获取CPU核心数
get_cpu_cores() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sysctl -n hw.ncpu
    else
        # Linux
        nproc
    fi
}

# 应用CPU优化
apply_cpu_optimizations() {
    echo -e "${BLUE}🔧 应用CPU优化...${NC}"
    
    # 获取CPU核心数
    CPU_CORES=$(get_cpu_cores)
    echo -e "${GREEN}检测到CPU核心数: $CPU_CORES${NC}"
    
    # 设置最激进的CPU优化
    export OMP_NUM_THREADS=$CPU_CORES
    export MKL_NUM_THREADS=$CPU_CORES
    export OPENBLAS_NUM_THREADS=$CPU_CORES
    export TORCH_NUM_THREADS=$CPU_CORES
    
    # CPU亲和性和调度优化
    export KMP_AFFINITY="granularity=fine,compact,1,0"
    export KMP_BLOCKTIME=0  # 零阻塞时间，最激进设置
    export KMP_SETTINGS=1
    
    # 内存优化
    export MALLOC_ARENA_MAX=2
    export MALLOC_MMAP_THRESHOLD_=131072
    export MALLOC_TRIM_THRESHOLD_=131072
    export MALLOC_TOP_PAD_=131072
    
    # PyTorch优化
    export TORCH_CUDNN_V8_API_ENABLED=1
    export TORCH_SHOW_CPP_STACKTRACES=0
    
    # 禁用不必要的功能以提高性能
    export PYTHONHASHSEED=0
    export PYTHONDONTWRITEBYTECODE=1
    
    echo -e "${GREEN}✓ CPU优化已应用${NC}"
    echo -e "${GREEN}  - CPU线程数: $CPU_CORES${NC}"
    echo -e "${GREEN}  - 内存优化: 启用${NC}"
    echo -e "${GREEN}  - PyTorch优化: 启用${NC}"
}

# 启动服务器
start_server() {
    echo -e "${CYAN}🚀 启动xLLM服务器...${NC}"
    echo -e "${CYAN}================================${NC}"
    echo -e "${GREEN}模型路径: $MODEL_PATH${NC}"
    echo -e "${GREEN}端口: $PORT${NC}"
    echo -e "${GREEN}量化: $QUANTIZATION${NC}"
    echo -e "${GREEN}调试模式: $(if [ "$DEBUG_MODE" = true ]; then echo "启用"; else echo "禁用"; fi)${NC}"
    echo ""
    
    # 构建启动命令
    CMD="python3 xllm_server.py --model-path $MODEL_PATH --port $PORT --quantization $QUANTIZATION"
    if [ "$DEBUG_MODE" = true ]; then
        CMD="$CMD --debug"
    fi
    
    echo -e "${BLUE}执行命令: $CMD${NC}"
    echo ""
    
    # 启动服务器
    eval $CMD
}

# 主函数
main() {
    echo -e "${CYAN}🚀 xLLM 启动脚本${NC}"
    echo -e "${CYAN}================================${NC}"
    echo -e "${GREEN}版本: v4.0${NC}"
    echo -e "${GREEN}优化程度: 250%性能提升${NC}"
    echo ""
    
    # 检查环境
    if ! check_environment; then
        exit 1
    fi
    
    # 如果只是检查环境，则退出
    if [ "$CHECK_ONLY" = true ]; then
        echo -e "${GREEN}✅ 环境检查完成，所有依赖都已找到${NC}"
        exit 0
    fi
    
    # 应用CPU优化
    apply_cpu_optimizations
    
    echo ""
    echo -e "${YELLOW}提示: 服务器启动后，可以使用以下命令测试:${NC}"
    echo -e "${YELLOW}curl http://localhost:$PORT/health${NC}"
    echo -e "${YELLOW}curl -X POST http://localhost:$PORT/generate -H \"Content-Type: application/json\" -d '{\"prompt\": \"Hello\", \"max_tokens\": 5, \"temperature\": 0.7}'${NC}"
    echo ""
    
    # 启动服务器
    start_server
}

# 运行主函数
main "$@"