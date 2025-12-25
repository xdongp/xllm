"""
xLLM HTTP服务接口 - FastAPI实现
"""
import argparse
import asyncio
import json
import logging
import sys
import os
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# 将父目录添加到路径中，以便我们可以从xllm包导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xllm.tokenizer_manager import TokenizerManager
from xllm.scheduler import Scheduler

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="xLLM API", description="CPU优化的大语言模型推理引擎")

# 全局变量
tokenizer_manager = None
scheduler = None

class GenerationRequest(BaseModel):
    """生成请求模型"""
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 100
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None  # 修改为使用Optional和Union

class EncodeRequest(BaseModel):
    """编码请求模型"""
    text: str

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str

class ModelInfoResponse(BaseModel):
    """模型信息响应模型"""
    model_name: str
    model_type: str

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化组件"""
    global tokenizer_manager, scheduler
    
    # 从环境变量或命令行参数获取配置
    model_path = os.getenv("MODEL_PATH", "/data/models/Qwen3-0.6B")
    quantization = os.getenv("QUANTIZATION", None)
    
    logger.info(f"Initializing xLLM with model: {model_path}")
    logger.info(f"Quantization: {quantization}")
    
    try:
        # 初始化调度器
        scheduler = Scheduler(model_path, quantization=quantization)
        
        # 初始化分词器管理器
        tokenizer_manager = TokenizerManager(model_path, quantization=quantization)
        
        # 启动调度器循环
        scheduler.start_scheduler_loop()
        
        logger.info("xLLM initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize xLLM: {str(e)}")
        raise

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    """生成文本"""
    logger.info(f"Received generation request: {request.prompt[:50]}...")
    
    try:
        if request.stream:
            # 流式响应
            async def stream_generator():
                async for chunk in tokenizer_manager.generate_stream(
                    request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stop=request.stop
                ):
                    yield chunk
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                }
            )
        else:
            # 非流式响应
            result = await tokenizer_manager.generate(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=request.stop
            )
            return {"text": result}
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/encode")
async def encode_text(request: EncodeRequest):
    """编码文本为token IDs"""
    logger.info(f"Encoding text: {request.text[:50]}...")
    
    try:
        token_ids = tokenizer_manager.encode(request.text)
        return {"token_ids": token_ids}
    except Exception as e:
        logger.error(f"Encoding failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查"""
    return HealthResponse(status="healthy")

@app.get("/models")
async def model_info():
    """获取模型信息"""
    # 这里应该返回实际的模型信息
    return ModelInfoResponse(
        model_name="Qwen3-0.6B-placeholder",
        model_type="placeholder"
    )

@app.get("/cache-stats")
async def get_cache_stats():
    """获取KV缓存统计信息"""
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")
    
    stats = scheduler.get_cache_stats()
    return stats

@app.get("/cache-stats/detailed")
async def get_detailed_cache_stats():
    """获取详细的KV缓存统计信息"""
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")
    
    stats = scheduler.get_detailed_cache_stats()
    return stats

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="xLLM Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-path", type=str, default="/data/models/Qwen3-0.6B", help="Path to the model")
    parser.add_argument("--quantization", type=str, choices=["int8", "fp16"], help="Quantization method")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # 设置环境变量供应用使用
    os.environ["MODEL_PATH"] = args.model_path
    if args.quantization:
        os.environ["QUANTIZATION"] = args.quantization
    
    # 根据--debug参数设置日志级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    logger.info(f"Starting xLLM server on {args.host}:{args.port}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
    
    # 运行服务器
    uvicorn.run(
        "xllm_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="debug" if args.debug else "info"
    )

if __name__ == "__main__":
    main()