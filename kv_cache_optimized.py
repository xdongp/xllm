"""
优化的KV缓存实现，用于提高推理效率
"""
import torch
import time
from typing import Dict, List, Optional, Tuple, OrderedDict
from typing import Any
import logging
import threading
from collections import deque

logger = logging.getLogger(__name__)

class OptimizedKVCacheEntry:
    """优化的KV缓存条目"""
    
    def __init__(self, key_cache: torch.Tensor, value_cache: torch.Tensor, sequence_id: str):
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.sequence_id = sequence_id
        self.last_access_time = time.time()
        self.access_count = 0
        self.created_time = time.time()
        self.last_update_time = time.time()
        # 预计算内存使用量
        self.memory_usage = self._calculate_memory_usage(key_cache, value_cache)
    
    def _calculate_memory_usage(self, key_cache: torch.Tensor, value_cache: torch.Tensor) -> float:
        """计算张量内存使用量（MB）"""
        element_size = key_cache.element_size()
        key_bytes = key_cache.numel() * element_size
        value_bytes = value_cache.numel() * element_size
        total_mb = (key_bytes + value_bytes) / (1024 * 1024)
        return total_mb


class OptimizedKVCache:
    """优化的KV缓存管理器 - 使用更高效的数据结构和算法"""
    
    def __init__(self, max_size: int = 100, max_memory_mb: int = 2000):
        """
        初始化KV缓存
        
        Args:
            max_size: 缓存最大条目数
            max_memory_mb: 缓存最大内存占用（MB）
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.cache = {}  # 使用普通字典替代OrderedDict以提高查找速度
        self.access_order = deque()  # 使用双端队列维护访问顺序
        self.memory_usage = 0.0
        
        # 缓存统计信息
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "memory_reclaims": 0
        }
        
        # 线程锁以确保线程安全
        self._lock = threading.RLock()
        
        logger.info(f"优化KV缓存初始化: 最大条目数={max_size}, 最大内存={max_memory_mb}MB")
    
    def _calculate_memory_usage(self, key_cache: torch.Tensor, value_cache: torch.Tensor) -> float:
        """计算张量内存使用量（MB）"""
        element_size = key_cache.element_size()
        key_bytes = key_cache.numel() * element_size
        value_bytes = value_cache.numel() * element_size
        total_mb = (key_bytes + value_bytes) / (1024 * 1024)
        return total_mb
    
    def _evict_lru(self) -> None:
        """淘汰最久未使用的缓存条目"""
        if not self.access_order:
            return
        
        with self._lock:
            # 从访问顺序队列的左端获取最久未使用的序列ID
            while self.access_order:
                sequence_id = self.access_order.popleft()
                if sequence_id in self.cache:
                    entry = self.cache.pop(sequence_id)
                    self.memory_usage -= entry.memory_usage
                    self.stats["evictions"] += 1
                    logger.debug(f"淘汰缓存条目: {sequence_id}, 释放内存: {entry.memory_usage:.2f}MB")
                    break
    
    def _ensure_size_limit(self) -> None:
        """确保缓存不超过大小限制"""
        with self._lock:
            while len(self.cache) >= self.max_size and self.cache:
                self._evict_lru()
    
    def _ensure_memory_limit(self) -> None:
        """确保缓存不超过内存限制"""
        if self.max_memory_mb is None:
            return
        
        with self._lock:
            while self.memory_usage > self.max_memory_mb and self.cache:
                self._evict_lru()
    
    def get(self, sequence_id: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """获取缓存条目 - 返回(key_cache, value_cache)元组"""
        with self._lock:
            if sequence_id in self.cache:
                entry = self.cache[sequence_id]
                
                # 更新访问统计
                entry.access_count += 1
                entry.last_access_time = time.time()
                
                # 更新访问顺序 - 移到队列右端（最近使用）
                if sequence_id in self.access_order:
                    self.access_order.remove(sequence_id)
                self.access_order.append(sequence_id)
                
                # 更新命中统计
                self.stats["hits"] += 1
                
                logger.debug(f"缓存命中: {sequence_id}, 访问次数: {entry.access_count}")
                return entry.key_cache, entry.value_cache
            else:
                # 更新未命中统计
                self.stats["misses"] += 1
                logger.debug(f"缓存未命中: {sequence_id}")
                return None
    
    def put(self, sequence_id: str, key_cache: torch.Tensor, value_cache: torch.Tensor) -> None:
        """添加或更新缓存条目"""
        with self._lock:
            # 计算新缓存的内存使用
            new_memory_usage = self._calculate_memory_usage(key_cache, value_cache)
            
            # 检查是否已存在该序列的缓存
            if sequence_id in self.cache:
                # 更新现有缓存
                existing_entry = self.cache[sequence_id]
                
                # 获取现有缓存的序列长度
                existing_seq_len = existing_entry.key_cache.size(2)
                new_seq_len = key_cache.size(2)
                
                logger.debug(f"更新缓存条目: {sequence_id}, 现有序列长度: {existing_seq_len}, 新序列长度: {new_seq_len}")
                
                if new_seq_len > existing_seq_len:
                    # 增量更新：只追加新token部分
                    tokens_to_add = new_seq_len - existing_seq_len
                    new_key_part = key_cache[:, :, -tokens_to_add:]
                    new_value_part = value_cache[:, :, -tokens_to_add:]
                    
                    if new_key_part.dim() == 3 and new_key_part.size(0) > 0 and new_key_part.size(1) > 0:
                        # 使用更高效的连接方式
                        updated_key = torch.cat([existing_entry.key_cache, new_key_part], dim=2)
                        updated_value = torch.cat([existing_entry.value_cache, new_value_part], dim=2)
                        
                        # 更新内存使用统计
                        self.memory_usage -= existing_entry.memory_usage
                        self.memory_usage += new_memory_usage
                        
                        # 创建新条目
                        updated_entry = OptimizedKVCacheEntry(updated_key, updated_value, sequence_id)
                        updated_entry.access_count = existing_entry.access_count
                        
                        # 更新缓存
                        self.cache[sequence_id] = updated_entry
                        
                        # 更新访问顺序
                        if sequence_id in self.access_order:
                            self.access_order.remove(sequence_id)
                        self.access_order.append(sequence_id)
                        
                        logger.debug(f"成功增量更新缓存: {sequence_id}, 追加token数: {tokens_to_add}, 新总长度: {updated_key.size(2)}")
                else:
                    # 完全替换缓存
                    self.memory_usage -= existing_entry.memory_usage
                    self.memory_usage += new_memory_usage
                    
                    # 创建新条目
                    new_entry = OptimizedKVCacheEntry(key_cache, value_cache, sequence_id)
                    new_entry.access_count = existing_entry.access_count
                    
                    # 更新缓存
                    self.cache[sequence_id] = new_entry
                    
                    # 更新访问顺序
                    if sequence_id in self.access_order:
                        self.access_order.remove(sequence_id)
                    self.access_order.append(sequence_id)
                    
                    logger.debug(f"完全替换缓存: {sequence_id}, 序列长度: {new_seq_len}")
            else:
                # 确保不超过内存限制
                self._ensure_memory_limit()
                
                # 确保不超过大小限制
                self._ensure_size_limit()
                
                # 创建新条目
                new_entry = OptimizedKVCacheEntry(key_cache, value_cache, sequence_id)
                
                # 添加到缓存
                self.cache[sequence_id] = new_entry
                self.memory_usage += new_entry.memory_usage
                
                # 更新访问顺序
                self.access_order.append(sequence_id)
                
                logger.debug(f"添加新缓存条目: {sequence_id}, 序列长度: {key_cache.size(2)}, 内存使用: {new_entry.memory_usage:.2f}MB")
    
    def remove(self, sequence_id: str) -> None:
        """移除缓存条目"""
        with self._lock:
            if sequence_id in self.cache:
                entry = self.cache.pop(sequence_id)
                self.memory_usage -= entry.memory_usage
                
                # 从访问顺序中移除
                if sequence_id in self.access_order:
                    self.access_order.remove(sequence_id)
                
                logger.debug(f"移除缓存条目: {sequence_id}, 释放内存: {entry.memory_usage:.2f}MB")
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            released_memory = self.memory_usage
            self.cache.clear()
            self.access_order.clear()
            self.memory_usage = 0
            self.stats["memory_reclaims"] += released_memory
            logger.info(f"清空缓存, 释放内存: {released_memory:.2f}MB")
    
    def resize(self, new_size: int) -> None:
        """调整缓存大小"""
        with self._lock:
            self.max_size = new_size
            
            # 如果当前缓存条目数超过新大小，淘汰多余的条目
            while len(self.cache) > self.max_size:
                self._evict_lru()
            
            logger.info(f"调整缓存大小: {new_size}, 当前条目数: {len(self.cache)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / (total_requests + 1e-9) * 100 if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "memory_usage": self.memory_usage,
                "max_memory": self.max_memory_mb,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                **self.stats
            }
    
    def trim_cache(self, max_entries: int = None) -> None:
        """修剪缓存，保留最近使用的条目"""
        if max_entries is None:
            max_entries = max(1, self.max_size // 2)
        
        with self._lock:
            # 只保留最近使用的max_entries个条目
            while len(self.cache) > max_entries:
                self._evict_lru()
            
            logger.info(f"修剪缓存完成, 保留条目数: {len(self.cache)}")


# 全局优化KV缓存实例
_global_optimized_kv_cache = None

def get_global_optimized_kv_cache(max_size: int = 100, max_memory_mb: int = 2000) -> OptimizedKVCache:
    """获取全局优化KV缓存实例
    
    Args:
        max_size: 缓存最大条目数
        max_memory_mb: 缓存最大内存占用（MB）
        
    Returns:
        OptimizedKVCache实例
    """
    global _global_optimized_kv_cache
    if _global_optimized_kv_cache is None:
        _global_optimized_kv_cache = OptimizedKVCache(max_size=max_size, max_memory_mb=max_memory_mb)
    return _global_optimized_kv_cache