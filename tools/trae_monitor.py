#!/usr/bin/env python3
"""
TRAE 监控工具
用于监控TRAE的执行结果，支持预设命令和TODO功能
"""

import os
import sys
import time
import json
import subprocess
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional


class CommandPreset:
    """命令预设类"""
    
    def __init__(self, name: str, command: str, description: str = ""):
        self.name = name
        self.command = command
        self.description = description
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "command": self.command,
            "description": self.description,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """从字典创建对象"""
        preset = cls(
            name=data["name"],
            command=data["command"],
            description=data.get("description", "")
        )
        preset.created_at = data.get("created_at", datetime.now().isoformat())
        return preset


class TodoItem:
    """TODO项类"""
    
    def __init__(self, content: str, priority: str = "medium", status: str = "pending"):
        self.content = content
        self.priority = priority  # high, medium, low
        self.status = status  # pending, in_progress, completed
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "content": self.content,
            "priority": self.priority,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """从字典创建对象"""
        todo = cls(
            content=data["content"],
            priority=data.get("priority", "medium"),
            status=data.get("status", "pending")
        )
        todo.created_at = data.get("created_at", datetime.now().isoformat())
        todo.updated_at = data.get("updated_at", datetime.now().isoformat())
        return todo
    
    def update_status(self, status: str):
        """更新状态"""
        self.status = status
        self.updated_at = datetime.now().isoformat()


class TraeMonitor:
    """TRAE监控器"""
    
    def __init__(self, data_dir: str = ".trae_monitor"):
        self.data_dir = data_dir
        self.presets_file = os.path.join(data_dir, "presets.json")
        self.todos_file = os.path.join(data_dir, "todos.json")
        self.log_file = os.path.join(data_dir, "monitor.log")
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 加载数据
        self.presets = self._load_presets()
        self.todos = self._load_todos()
    
    def _load_presets(self) -> List[CommandPreset]:
        """加载命令预设"""
        if not os.path.exists(self.presets_file):
            return []
        
        try:
            with open(self.presets_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [CommandPreset.from_dict(preset_data) for preset_data in data]
        except Exception as e:
            print(f"加载命令预设失败: {e}")
            return []
    
    def _save_presets(self):
        """保存命令预设"""
        try:
            print(f"尝试保存预设到文件: {self.presets_file}")
            print(f"预设数量: {len(self.presets)}")
            with open(self.presets_file, "w", encoding="utf-8") as f:
                json.dump([preset.to_dict() for preset in self.presets], f, indent=2, ensure_ascii=False)
            print("预设保存成功")
        except Exception as e:
            print(f"保存命令预设失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_todos(self) -> List[TodoItem]:
        """加载TODO列表"""
        if not os.path.exists(self.todos_file):
            return []
        
        try:
            with open(self.todos_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [TodoItem.from_dict(todo_data) for todo_data in data]
        except Exception as e:
            print(f"加载TODO列表失败: {e}")
            return []
    
    def _save_todos(self):
        """保存TODO列表"""
        try:
            with open(self.todos_file, "w", encoding="utf-8") as f:
                json.dump([todo.to_dict() for todo in self.todos], f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存TODO列表失败: {e}")
    
    def _log(self, message: str, level: str = "INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        # 打印到控制台
        print(log_entry.strip())
        
        # 保存到文件
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"写入日志失败: {e}")
    
    def add_preset(self, name: str, command: List[str], description: str = ""):
        """添加命令预设"""
        # 检查名称是否已存在
        for preset in self.presets:
            if preset.name == name:
                print(f"错误: 预设命令 '{name}' 已存在")
                return False
        
        # 将命令列表连接成字符串
        command_str = " ".join(command)
        preset = CommandPreset(name, command_str, description)
        self.presets.append(preset)
        self._save_presets()
        self._log(f"已添加命令预设: {name}")
        return True
    
    def list_presets(self):
        """列出所有命令预设"""
        if not self.presets:
            print("没有命令预设")
            return
        
        print("命令预设列表:")
        print("-" * 80)
        for i, preset in enumerate(self.presets, 1):
            print(f"{i}. {preset.name}")
            print(f"   命令: {preset.command}")
            if preset.description:
                print(f"   描述: {preset.description}")
            print(f"   创建时间: {preset.created_at}")
            print("-" * 80)
    
    def remove_preset(self, name: str):
        """删除命令预设"""
        for i, preset in enumerate(self.presets):
            if preset.name == name:
                del self.presets[i]
                self._save_presets()
                self._log(f"已删除命令预设: {name}")
                return True
        
        print(f"错误: 预设命令 '{name}' 不存在")
        return False
    
    def add_todo(self, content: str, priority: str = "medium"):
        """添加TODO项"""
        todo = TodoItem(content, priority)
        self.todos.append(todo)
        self._save_todos()
        self._log(f"已添加TODO: {content}")
        return True
    
    def list_todos(self, status: Optional[str] = None):
        """列出TODO项"""
        filtered_todos = self.todos
        if status:
            filtered_todos = [todo for todo in self.todos if todo.status == status]
        
        if not filtered_todos:
            print(f"没有{' ' + status if status else ''} TODO项")
            return
        
        print(f"{' ' + status if status else ''} TODO列表:")
        print("-" * 80)
        for i, todo in enumerate(filtered_todos, 1):
            print(f"{i}. [{todo.status}] [{todo.priority}] {todo.content}")
            print(f"   创建时间: {todo.created_at}")
            print(f"   更新时间: {todo.updated_at}")
            print("-" * 80)
    
    def update_todo_status(self, index: int, status: str):
        """更新TODO项状态"""
        if index < 1 or index > len(self.todos):
            print(f"错误: TODO项索引 {index} 无效")
            return False
        
        todo = self.todos[index - 1]
        todo.update_status(status)
        self._save_todos()
        self._log(f"已更新TODO状态: {todo.content} -> {status}")
        return True
    
    def remove_todo(self, index: int):
        """删除TODO项"""
        if index < 1 or index > len(self.todos):
            print(f"错误: TODO项索引 {index} 无效")
            return False
        
        todo = self.todos.pop(index - 1)
        self._save_todos()
        self._log(f"已删除TODO: {todo.content}")
        return True
    
    def run_command(self, command: str, cwd: Optional[str] = None):
        """执行命令并监控结果"""
        self._log(f"开始执行命令: {command}", "INFO")
        
        try:
            # 执行命令
            process = subprocess.Popen(
                command, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                cwd=cwd,
                text=True
            )
            
            # 实时读取输出
            output_lines = []
            while process.poll() is None:
                line = process.stdout.readline()
                if line:
                    line = line.strip()
                    output_lines.append(line)
                    self._log(line, "OUTPUT")
                    time.sleep(0.1)  # 避免过度占用CPU
            
            # 读取剩余输出
            for line in process.stdout.readlines():
                line = line.strip()
                output_lines.append(line)
                self._log(line, "OUTPUT")
            
            # 检查退出码
            exit_code = process.returncode
            if exit_code == 0:
                self._log(f"命令执行成功，退出码: {exit_code}", "SUCCESS")
            else:
                self._log(f"命令执行失败，退出码: {exit_code}", "ERROR")
            
            # 保存输出到文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.data_dir, f"command_output_{timestamp}.log")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(output_lines))
            
            return {
                "success": exit_code == 0,
                "exit_code": exit_code,
                "output": output_lines,
                "output_file": output_file
            }
            
        except Exception as e:
            self._log(f"执行命令时出错: {e}", "ERROR")
            return {
                "success": False,
                "exit_code": -1,
                "error": str(e)
            }
    
    def run_preset(self, preset_name: str, cwd: Optional[str] = None):
        """执行预设命令"""
        for preset in self.presets:
            if preset.name == preset_name:
                self._log(f"执行预设命令: {preset_name}")
                return self.run_command(preset.command, cwd)
        
        print(f"错误: 预设命令 '{preset_name}' 不存在")
        return None
    
    def kill_process(self, process_name: str):
        """终止指定名称的进程"""
        self._log(f"尝试终止进程: {process_name}")
        
        try:
            # 在macOS/Linux上使用pkill命令
            command = f"pkill -f {process_name}"
            result = subprocess.run(
                command, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True
            )
            
            if result.returncode == 0:
                self._log(f"成功终止进程: {process_name}", "SUCCESS")
                return True
            else:
                self._log(f"终止进程失败，退出码: {result.returncode}", "ERROR")
                self._log(f"错误信息: {result.stdout}", "ERROR")
                return False
        except Exception as e:
            self._log(f"终止进程时出错: {e}", "ERROR")
            return False


def main():
    parser = argparse.ArgumentParser(description="TRAE监控工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 监控命令执行
    monitor_parser = subparsers.add_parser("monitor", help="监控命令执行")
    monitor_parser.add_argument("cmd", nargs="*", help="要执行的命令")
    monitor_parser.add_argument("--cwd", help="命令执行的工作目录")
    
    # 终止进程
    kill_parser = subparsers.add_parser("kill", help="终止指定进程")
    kill_parser.add_argument("process_name", help="进程名称或关键字")
    
    # 命令预设管理
    preset_parser = subparsers.add_parser("preset", help="命令预设管理")
    preset_subparsers = preset_parser.add_subparsers(dest="preset_command", help="预设管理命令")
    
    # 添加预设
    add_preset_parser = preset_subparsers.add_parser("add", help="添加命令预设")
    add_preset_parser.add_argument("name", help="预设名称")
    add_preset_parser.add_argument("cmd", help="命令内容（使用引号括起来）")
    add_preset_parser.add_argument("--desc", help="命令描述")
    
    # 列出预设
    list_preset_parser = preset_subparsers.add_parser("list", help="列出命令预设")
    
    # 删除预设
    remove_preset_parser = preset_subparsers.add_parser("remove", help="删除命令预设")
    remove_preset_parser.add_argument("name", help="预设名称")
    
    # 执行预设
    run_preset_parser = preset_subparsers.add_parser("run", help="执行命令预设")
    run_preset_parser.add_argument("name", help="预设名称")
    run_preset_parser.add_argument("--cwd", help="命令执行的工作目录")
    
    # TODO管理
    todo_parser = subparsers.add_parser("todo", help="TODO列表管理")
    todo_subparsers = todo_parser.add_subparsers(dest="todo_command", help="TODO管理命令")
    
    # 添加TODO
    add_todo_parser = todo_subparsers.add_parser("add", help="添加TODO项")
    add_todo_parser.add_argument("content", help="TODO内容")
    add_todo_parser.add_argument("--priority", choices=["high", "medium", "low"], default="medium", help="优先级")
    
    # 列出TODO
    list_todo_parser = todo_subparsers.add_parser("list", help="列出TODO项")
    list_todo_parser.add_argument("--status", choices=["pending", "in_progress", "completed"], help="状态筛选")
    
    # 更新TODO状态
    update_todo_parser = todo_subparsers.add_parser("update", help="更新TODO项状态")
    update_todo_parser.add_argument("index", type=int, help="TODO项索引")
    update_todo_parser.add_argument("status", choices=["pending", "in_progress", "completed"], help="新状态")
    
    # 删除TODO
    remove_todo_parser = todo_subparsers.add_parser("remove", help="删除TODO项")
    remove_todo_parser.add_argument("index", type=int, help="TODO项索引")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # 创建监控器
    monitor = TraeMonitor()
    
    # 处理命令
    if args.command == "monitor":
        # 检查cmd是否存在
        if not args.cmd:
            print("错误: 必须提供要执行的命令")
            sys.exit(1)
        
        # 将命令列表连接成字符串
        command_str = " ".join(args.cmd)
        monitor.run_command(command_str, args.cwd)
    
    elif args.command == "kill":
        monitor.kill_process(args.process_name)
    
    elif args.command == "preset":
        if args.preset_command == "add":
            monitor.add_preset(args.name, [args.cmd], args.desc)
        elif args.preset_command == "list":
            monitor.list_presets()
        elif args.preset_command == "remove":
            monitor.remove_preset(args.name)
        elif args.preset_command == "run":
            monitor.run_preset(args.name, args.cwd)
        else:
            preset_parser.print_help()
    
    elif args.command == "todo":
        if args.todo_command == "add":
            monitor.add_todo(args.content, args.priority)
        elif args.todo_command == "list":
            monitor.list_todos(args.status)
        elif args.todo_command == "update":
            monitor.update_todo_status(args.index - 1, args.status)
        elif args.todo_command == "remove":
            monitor.remove_todo(args.index - 1)
        else:
            todo_parser.print_help()

if __name__ == "__main__":
    main()