import requests
import json
import time
import argparse
import sys
from typing import Dict, Any, Optional


class XLLMClient:
    """xLLM 客户端类"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.session = requests.Session()
        self.default_params = {
            "max_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50
        }
        self.interrupt_requested = False  # 中断请求标志
        
    def check_server_health(self) -> bool:
        """检查服务器健康状态"""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def request_interrupt(self):
        """请求中断当前生成"""
        self.interrupt_requested = True
    
    def reset_interrupt(self):
        """重置中断标志"""
        self.interrupt_requested = False
    
    def generate_text(self, prompt: str, **kwargs) -> Optional[Dict[str, Any]]:
        """生成文本"""
        # 重置中断标志
        self.reset_interrupt()
        
        # 合并参数
        params = self.default_params.copy()
        params.update(kwargs)
        params["prompt"] = prompt
        
        # 检查是否请求流式输出
        if params.get("stream", False):
            return self._generate_streaming(prompt, **{k: v for k, v in params.items() if k != "prompt"})
        else:
            try:
                response = self.session.post(
                    f"{self.server_url}/generate",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(params),
                    timeout=60  # 60秒超时
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"❌ 服务器返回错误: {response.status_code}")
                    print(f"错误信息: {response.text}")
                    return None
                    
            except requests.exceptions.Timeout:
                print("❌ 请求超时")
                return None
            except requests.exceptions.ConnectionError:
                print("❌ 无法连接到服务器，请确保服务器正在运行")
                return None
            except Exception as e:
                print(f"❌ 请求失败: {e}")
                return None

    def _generate_streaming(self, prompt: str, **kwargs) -> Optional[Dict[str, Any]]:
        """生成文本（流式输出）"""
        params = kwargs.copy()
        params["prompt"] = prompt
        params["stream"] = True  # 确保启用流式
        
        try:
            # 发送流式请求
            response = self.session.post(
                f"{self.server_url}/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps(params),
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                print("⏳ 服务器正在生成回答... (输入 Ctrl+C 中断) ", end='', flush=True)
                
                # 处理SSE流
                full_text = ""
                
                for line in response.iter_lines(decode_unicode=True):
                    # 检查是否请求中断
                    if self.interrupt_requested:
                        print("\n✅ 生成已中断")
                        return {"generated_text": full_text, "finish_reason": "interrupted"}
                    
                    if line.startswith('data: '):
                        data_str = line[6:]  # 移除 'data: ' 前缀
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if "token" in data:
                                token_text = data["token"]
                                full_text += token_text
                                
                                # 直接输出token，实现逐字追加效果
                                #print(f"\r{chr(32) * 40}\r", end='')  # 清除进度信息
                                print(token_text, end='', flush=True)
                                
                                # 添加轻微延迟，避免显示过快造成视觉混乱
                                #time.sleep(0.01)
                            elif "generated_text" in data and data.get("done"):
                                # 完成消息
                                print()  # 换行
                                return {"generated_text": data["generated_text"], "finish_reason": data.get("finish_reason", "stop")}
                        except json.JSONDecodeError:
                            continue
                
                print()  # 换行
                return {"generated_text": full_text}
            else:
                print(f"❌ 服务器返回错误: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("❌ 请求超时")
            return None
        except requests.exceptions.ConnectionError:
            print("❌ 无法连接到服务器，请确保服务器正在运行")
            return None
        except Exception as e:
            print(f"❌ 流式请求失败: {e}")
            return None
    
    def chat_loop(self):
        """交互式聊天循环"""
        print("🚀 xLLM 交互式客户端")
        print("=" * 50)
        print("提示:")
        print("- 输入问题后按回车发送")
        print("- 输入 'quit' 或 'exit' 退出")
        print("- 输入 'clear' 清屏")
        print("- 输入 'help' 显示帮助")
        print("- 输入 'config' 查看/修改配置")
        print("- 输入 'stream on/off' 开启/关闭流式输出")
        print("- 输入 'tokens <数字>' 设置本次生成的最大token数")
        print("=" * 50)
        
        # 检查服务器状态
        if not self.check_server_health():
            print("❌ 无法连接到服务器，请确保 xLLM 服务器正在运行")
            print(f"服务器地址: {self.server_url}")
            return
        
        print("✅ 已连接到服务器，开始对话...")
        print()
        
        # 默认启用流式输出
        streaming_enabled = True
        
        while True:
            try:
                user_input = input("💬 您: ").strip()
                
                if not user_input:
                    continue
                
                # 处理特殊命令
                if user_input.lower() in ['quit', 'exit', 'q']:
                    # 如果当前有请求在进行，中断它
                    self.request_interrupt()
                    print("👋 再见！")
                    break
                elif user_input.lower() == 'clear':
                    import os
                    os.system('clear' if os.name != 'nt' else 'cls')
                    continue
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif user_input.lower() == 'config':
                    self.show_config()
                    continue
                elif user_input.lower().startswith('set '):
                    self.set_config(user_input[4:])  # 去掉 'set ' 前缀
                    continue
                elif user_input.lower() == 'stream on':
                    streaming_enabled = True
                    print("✅ 已启用流式输出")
                    continue
                elif user_input.lower() == 'stream off':
                    streaming_enabled = False
                    print("✅ 已禁用流式输出")
                    continue
                
                # 检查是否为设置token数的命令
                if user_input.lower().startswith('tokens '):
                    try:
                        token_value = int(user_input[7:].strip())  # 去掉 'tokens ' 前缀
                        if token_value > 0:
                            # 为本次请求临时设置max_tokens
                            result = self.generate_text(user_input, stream=streaming_enabled, max_tokens=token_value)
                        else:
                            print("❌ 无效的token数，请输入大于0的数字")
                            continue
                    except ValueError:
                        print("❌ 无效的token数，请输入数字")
                        continue
                else:
                    # 发送请求到服务器
                    print("⏳ 服务器正在生成回答... (输入 Ctrl+C 中断) \n", end='', flush=True)
                    
                    start_time = time.time()
                    # 传递stream参数
                    result = self.generate_text(user_input, stream=streaming_enabled)
                    end_time = time.time()
                
                if result:
                    print("\r" + " " * 40 + "\r", end='')  # 清除 "服务器正在生成回答..." 文本
                    
                    # 提取生成的文本
                    if isinstance(result, dict):
                        if "generated_text" in result:
                            generated_text = result["generated_text"]
                        elif "text" in result:
                            generated_text = result["text"]
                        else:
                            generated_text = str(result)
                    else:
                        generated_text = str(result)
                    
                    # 如果不是流式输出，显示完整结果
                    if not streaming_enabled:
                        print(f"🤖 服务器: {generated_text}")
                    
                    # 显示统计信息
                    response_time = end_time - start_time
                    token_count = len(generated_text.split())
                    speed = token_count / response_time if response_time > 0 else 0
                    
                    print(f"📈 统计: {token_count} tokens, {response_time:.2f}s, {speed:.2f} tokens/s")
                else:
                    print("\r" + " " * 40 + "\r", end='')  # 清除 "服务器正在生成回答..." 文本
                    print("❌ 生成失败，请重试")
                
                print()  # 空行分隔
                
            except KeyboardInterrupt:
                print("\n✅ 生成已中断")
                # 重置中断标志
                self.reset_interrupt()
                continue  # 继续对话循环
            except EOFError:
                print("\n👋 对话结束")
                break
    
    def show_help(self):
        """显示帮助信息"""
        print("\n📖 帮助信息:")
        print("  - 直接输入问题开始对话")
        print("  - quit/exit/q: 退出程序")
        print("  - clear: 清屏")
        print("  - help: 显示此帮助")
        print("  - config: 显示当前配置")
        print("  - stream on/off: 开启/关闭流式输出")
        print("  - tokens <数字>: 设置本次生成的最大token数")
        print("  - set <param> <value>: 设置参数")
        print("    例如: set temperature 0.8")
        print("    例如: set max_tokens 256")
        print()
    
    def show_config(self):
        """显示当前配置"""
        print("\n⚙️  当前配置:")
        for key, value in self.default_params.items():
            print(f"  {key}: {value}")
        print()
    
    def set_config(self, config_str: str):
        """设置配置参数"""
        try:
            parts = config_str.strip().split()
            if len(parts) >= 2:
                param_name = parts[0]
                param_value = " ".join(parts[1:])
                
                # 尝试转换值的类型
                try:
                    if param_name in ['max_tokens', 'top_k']:
                        param_value = int(param_value)
                    elif param_name in ['temperature', 'top_p']:
                        param_value = float(param_value)
                except ValueError:
                    print(f"❌ 无效的参数值: {param_value}")
                    return
                
                self.default_params[param_name] = param_value
                print(f"✅ 已设置 {param_name} = {param_value}")
            else:
                print("❌ 用法: set <参数名> <参数值>")
                print("   例如: set temperature 0.8")
        except Exception as e:
            print(f"❌ 设置配置失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="xLLM 交互式客户端")
    parser.add_argument("--server-url", default="http://localhost:8000",
                       help="xLLM 服务器地址 (默认: http://localhost:8000)")
    parser.add_argument("--max-tokens", type=int, default=50,
                       help="最大生成token数 (默认: 50)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="温度参数，控制随机性 (默认: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p 采样参数 (默认: 0.9)")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Top-k 采样参数 (默认: 50)")
    
    args = parser.parse_args()
    
    # 创建客户端
    client = XLLMClient(args.server_url)
    
    # 设置默认参数
    client.default_params.update({
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k
    })
    
    # 开始聊天循环
    client.chat_loop()


if __name__ == "__main__":
    main()
