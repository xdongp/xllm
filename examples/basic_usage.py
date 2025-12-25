#!/usr/bin/env python3
"""
xLLM的基本使用示例
"""
import requests
import json
import time


def example_text_generation():
    """使用xLLM API生成文本的示例"""
    # 定义API端点
    url = "http://localhost:8080/generate"
    
    # 定义请求负载
    payload = {
        "prompt": "人工智能是",
        "temperature": 0.7,
        "max_tokens": 50,
        "stream": False
    }
    
    # 发送请求
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            result = response.json()
            print("生成的文本:")
            print(result["generated_text"])
            print(f"完成原因: {result['finish_reason']}")
        else:
            print(f"错误: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"连接xLLM服务器时出错: {e}")


def example_text_generation_streaming():
    """使用xLLM API流式生成文本的示例"""
    # 定义API端点
    url = "http://localhost:8080/generate"
    
    # 定义请求负载
    payload = {
        "prompt": "人工智能是",
        "temperature": 0.7,
        "max_tokens": 50,
        "stream": True
    }
    
    # 发送请求
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            stream=True
        )
        
        if response.status_code == 200:
            print("流式生成的文本:")
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        data = decoded_line[6:]  # 移除 'data: ' 前缀
                        if data != '[DONE]':
                            try:
                                json_data = json.loads(data)
                                print(json_data.get('token', ''), end='', flush=True)
                            except json.JSONDecodeError:
                                pass
            print()  # 结尾换行
        else:
            print(f"错误: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"连接xLLM服务器时出错: {e}")


def example_text_encoding():
    """使用xLLM API编码文本的示例"""
    # 定义API端点
    url = "http://localhost:8080/encode"
    
    # 定义请求负载
    payload = {
        "text": "人工智能是"
    }
    
    # 发送请求
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            result = response.json()
            print("编码的令牌:")
            print(result["token_ids"])
        else:
            print(f"错误: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"连接xLLM服务器时出错: {e}")


def example_health_check():
    """使用xLLM API进行健康检查的示例"""
    # 定义API端点
    url = "http://localhost:8080/health"
    
    # 发送请求
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            result = response.json()
            print("健康检查结果:")
            print(result)
        else:
            print(f"错误: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"连接xLLM服务器时出错: {e}")


if __name__ == "__main__":
    print("xLLM基本使用示例")
    print("=" * 30)
    
    # 等待服务器启动片刻
    time.sleep(1)
    
    print("\n1. 健康检查:")
    example_health_check()
    
    print("\n2. 文本编码:")
    example_text_encoding()
    
    print("\n3. 文本生成:")
    example_text_generation()
    
    print("\n4. 流式文本生成:")
    example_text_generation_streaming()