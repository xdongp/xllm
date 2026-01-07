import argparse
import asyncio
import json
import time
import os
import sys
import random
import logging
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from tokenizer_manager import TokenizerManager
from scheduler import Scheduler

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置调度器日志级别为INFO以查看详细日志
import scheduler as scheduler_module
scheduler_module.logger.setLevel(logging.INFO)

# 全局变量
scheduler: Optional[Scheduler] = None
tokenizer_manager: Optional[TokenizerManager] = None


# 创建应用
def get_app():
    return create_app()

# 创建应用实例
def create_app():
    app = FastAPI(
        title="xLLM API",
        description="高效的大语言模型推理服务器",
        version="1.0.0"
    )
    
    # 从环境变量获取DEBUG设置
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    
    # 设置日志级别
    if debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    # 设置调度器的日志级别为DEBUG，以便查看详细的调度信息
    logging.getLogger("scheduler").setLevel(logging.DEBUG)
    
    @app.on_event('startup')
    async def startup_event():
        """启动事件 - 初始化调度器和分词器管理器"""
        global scheduler, tokenizer_manager
        
        try:
            model_path = os.getenv("MODEL_PATH", "./model/Qwen/Qwen3-0.6B")
            quantization = os.getenv("QUANTIZATION", "fp16")
            
            # 从环境变量或默认值获取调度器参数
            max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "8"))
            max_context_length = int(os.getenv("MAX_CONTEXT_LENGTH", "2048"))
            
            logger.info(f"初始化调度器，模型路径: {model_path}, 量化: {quantization}")
            logger.info(f"调度器配置: max_batch_size={max_batch_size}, max_context_length={max_context_length}")
            
            scheduler = Scheduler(
                model_path=model_path,
                quantization=quantization,
                max_batch_size=max_batch_size,
                max_context_length=max_context_length
            )
            scheduler.start_scheduler_loop()
            
            logger.info("初始化分词器管理器")
            tokenizer_manager = TokenizerManager(model_path)
            
            logger.info("服务器启动完成")
        except Exception as e:
            logger.error(f"启动失败: {e}")
            raise
    
    @app.on_event('shutdown')
    async def shutdown_event():
        """关闭事件 - 清理资源"""
        global scheduler
        if scheduler:
            scheduler.stop_scheduler_loop()
            logger.info("调度器已停止")
    
    @app.get("/health")
    async def health():
        """健康检查端点"""
        return {"status": "healthy", "model_loaded": scheduler is not None}
    
    @app.post("/generate")
    async def generate(request: Dict[str, Any]):
        """生成端点 - 支持流式和非流式响应"""
        # 参数验证
        prompt = request.get('prompt', '')
        messages = request.get('messages', [])
        
        if not prompt and not messages:
            raise HTTPException(status_code=400, detail="必须提供prompt或messages参数")
        
        # 处理输入
        if messages:
            # 如果提供了messages，将其转换为prompt
            texts = []
            for msg in messages:
                if isinstance(msg, dict) and 'content' in msg:
                    texts.append(str(msg['content']))
            input_text = '\n'.join(texts)
        else:
            input_text = str(prompt)
        
        max_tokens = request.get('max_tokens', 100)
        temperature = request.get('temperature', 0.7)
        stream = request.get('stream', False)
        
        # 生成唯一请求ID
        import time
        import random
        request_id = f"req_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        try:
            if stream:
                # 流式响应
                async def generate_stream():
                    try:
                        async for token_data in tokenizer_manager.generate_stream(
                            request_id=request_id,
                            prompt=input_text,
                            max_tokens=max_tokens,
                            temperature=temperature
                        ):
                            if token_data.get('finished'):
                                # 发送最后一个token
                                yield f"data: {json.dumps({'token': token_data['text'], 'done': True})}\n\n"
                                yield "data: [DONE]\n\n"
                            else:
                                # 发送单个token
                                yield f"data: {json.dumps({'token': token_data['text']})}\n\n"
                    except Exception as e:
                        logger.error(f"流式生成错误: {e}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                
                return StreamingResponse(
                    generate_stream(),
                    media_type="text/event-stream"
                )
            else:
                # 非流式响应 - 直接使用tokenizer_manager（暂时禁用调度器以恢复性能）
                start_time = asyncio.get_event_loop().time()
                
                try:
                    result_text = await tokenizer_manager.generate(
                        request_id=request_id,
                        prompt=input_text,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                except Exception as e:
                    logger.error(f"生成失败: {e}")
                    raise HTTPException(status_code=500, detail=str(e))


                
                end_time = asyncio.get_event_loop().time()
                
                # 计算token数量
                tokens = tokenizer_manager.tokenizer.encode(result_text)
                
                response_time = end_time - start_time
                tokens_per_second = len(tokens) / response_time if response_time > 0 else 0
                
                return {
                    "id": request_id,
                    "text": result_text,
                    "tokens": tokens,
                    "response_time": round(response_time, 3),
                    "tokens_per_second": round(tokens_per_second, 2)
                }
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"生成请求错误: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/encode")
    async def encode(request: Dict[str, Any]):
        """编码端点 - 将文本编码为token IDs"""
        text = request.get('text', '')
        if not isinstance(text, str):
            raise HTTPException(status_code=400, detail="必须提供text参数")
        
        try:
            tokens = tokenizer_manager.encode(text)
            return {
                "tokens": tokens,
                "length": len(tokens)
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"编码请求错误: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app

# 创建应用实例
app = get_app()

# 启动服务器
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="xLLM 推理服务器")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--model-path", type=str, required=True, help="模型路径")
    parser.add_argument("--quantization", type=str, choices=["int8", "fp16"], default="fp16", help="量化方法")
    parser.add_argument("--reload", action="store_true", help="启用热重载")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--max-batch-size", type=int, default=8, help="最大批处理大小")
    parser.add_argument("--max-context-length", type=int, default=2048, help="最大上下文长度")
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["QUANTIZATION"] = args.quantization
    os.environ["MAX_BATCH_SIZE"] = str(args.max_batch_size)
    os.environ["MAX_CONTEXT_LENGTH"] = str(args.max_context_length)
    
    # 根据debug参数设置日志级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    # 启动服务器
    logger.info(f"启动服务器在 {args.host}:{args.port}")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"量化: {args.quantization}")
    logger.info(f"最大批处理大小: {args.max_batch_size}")
    logger.info(f"最大上下文长度: {args.max_context_length}")
    
    uvicorn.run(
        "xllm_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info" if not args.debug else "debug",
        workers=1
    )