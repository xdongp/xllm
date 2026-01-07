"""
KV缓存实现，用于存储和复用注意力键值对以提高推理效率
"""

import torch
import time  # 添加time模块导入
from typing import Dict, List, Optional, Tuple, OrderedDict
from typing import Any  # 添加Any类型的导入
import logging

logger = logging.getLogger(__name__)


class KVCacheEntry:
    """KV缓存条目"""
    
    def __init__(self, key_cache: torch.Tensor, value_cache: torch.Tensor, sequence_id: str):
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.sequence_id = sequence_id
        self.last_access_time = 0  # 最后访问时间，用于LRU淘汰
        self.hit_count = 0  # 缓存命中次数统计
        self.created_time = time.time()  # 条目创建时间
        self.last_update_time = time.time()  # 最后更新时间


class KVCache:
    """KV缓存管理器 - 支持LRU淘汰策略和内存优化"""
    
    def __init__(self, max_size: int = 10, max_memory_mb: int = None):
        """
        初始化KV缓存
        
        Args:
            max_size: 缓存最大条目数
            max_memory_mb: 缓存最大内存占用（MB）
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb  # 新增：最大内存限制
        self.cache = OrderedDict()
        self.memory_usage = 0  # 内存使用量（MB）
        
        # 缓存统计信息
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "memory_reclaims": 0
        }
        
        logger.info(f"KV缓存初始化: 最大条目数={max_size}, 最大内存={max_memory_mb}MB")
    
    def _calculate_memory_usage(self, key_cache: torch.Tensor, value_cache: torch.Tensor) -> float:
        """计算张量内存使用量（MB）"""
        # 获取张量元素大小（字节）
        element_size = key_cache.element_size()
        
        # 计算总字节数
        key_bytes = key_cache.numel() * element_size
        value_bytes = value_cache.numel() * element_size
        
        # 转换为MB
        total_mb = (key_bytes + value_bytes) / (1024 * 1024)
        
        return total_mb
    
    def _evict_oldest(self) -> None:
        """淘汰最旧的缓存条目"""
        if not self.cache:
            return
        
        # 淘汰最旧的条目
        sequence_id, entry = self.cache.popitem(last=False)
        
        # 更新内存使用统计
        self.memory_usage -= entry.memory_usage
        self.stats["evictions"] += 1
        
        logger.debug(f"淘汰缓存条目: {sequence_id}, 释放内存: {entry.memory_usage:.2f}MB")
    
    def _ensure_memory_limit(self) -> None:
        """确保缓存不超过内存限制"""
        if self.max_memory_mb is None:
            return
        
        # 检查内存使用
        while self.memory_usage > self.max_memory_mb and self.cache:
            self._evict_oldest()
    
    def get(self, sequence_id: str) -> Optional[KVCacheEntry]:
        """获取缓存条目"""
        if sequence_id in self.cache:
            # 更新访问时间（移到OrderedDict末尾）
            entry = self.cache.pop(sequence_id)
            self.cache[sequence_id] = entry
            
            # 更新命中统计
            self.stats["hits"] += 1
            entry.access_count += 1
            
            logger.debug(f"缓存命中: {sequence_id}, 访问次数: {entry.access_count}")
            return entry
        else:
            # 更新未命中统计
            self.stats["misses"] += 1
            logger.debug(f"缓存未命中: {sequence_id}")
            return None
    
    def put(self, sequence_id: str, key_cache: torch.Tensor, value_cache: torch.Tensor) -> None:
        """添加或更新缓存条目，支持增量更新
        
        Args:
            sequence_id: 序列ID
            key_cache: 更新的key缓存张量
            value_cache: 更新的value缓存张量
        """
        # 计算新缓存的内存使用
        new_memory_usage = self._calculate_memory_usage(key_cache, value_cache)
        
        # 检查是否已存在该序列的缓存
        if sequence_id in self.cache:
            # 增量更新缓存
            existing_entry = self.cache[sequence_id]
            
            # 获取现有缓存的序列长度
            existing_seq_len = existing_entry.key_cache.size(2)
            new_seq_len = key_cache.size(2)
            
            logger.debug(f"更新缓存条目: {sequence_id}, 现有序列长度: {existing_seq_len}, 新序列长度: {new_seq_len}")
            
            if new_seq_len > existing_seq_len:
                # 增量更新：只追加新token部分
                # 计算需要追加的token数量
                tokens_to_add = new_seq_len - existing_seq_len
                
                # 获取新的token部分
                new_key_part = key_cache[:, :, -tokens_to_add:]
                new_value_part = value_cache[:, :, -tokens_to_add:]
                
                # 检查是否有空维度
                if new_key_part.dim() == 3 and new_key_part.size(0) > 0 and new_key_part.size(1) > 0:
                    # 执行增量更新
                    updated_key = torch.cat([existing_entry.key_cache, new_key_part], dim=2)
                    updated_value = torch.cat([existing_entry.value_cache, new_value_part], dim=2)
                    
                    # 更新内存使用统计
                    self.memory_usage -= existing_entry.memory_usage
                    self.memory_usage += new_memory_usage
                    
                    # 创建新条目
                    updated_entry = KVCacheEntry(updated_key, updated_value, sequence_id)
                    updated_entry.memory_usage = new_memory_usage
                    updated_entry.access_count = existing_entry.access_count
                    updated_entry.last_accessed = time.time()
                    
                    # 更新缓存
                    self.cache[sequence_id] = updated_entry
                    
                    logger.debug(f"成功增量更新缓存: {sequence_id}, 追加token数: {tokens_to_add}, 新总长度: {updated_key.size(2)}")
            else:
                # 完全替换缓存
                self.memory_usage -= existing_entry.memory_usage
                self.memory_usage += new_memory_usage
                
                # 创建新条目
                new_entry = KVCacheEntry(key_cache, value_cache, sequence_id)
                new_entry.memory_usage = new_memory_usage
                new_entry.access_count = existing_entry.access_count
                new_entry.last_accessed = time.time()
                
                # 更新缓存
                self.cache[sequence_id] = new_entry
                
                logger.debug(f"完全替换缓存: {sequence_id}, 序列长度: {new_seq_len}")
        else:
            # 确保不超过最大内存限制
            self._ensure_memory_limit()
            
            # 确保不超过最大条目数
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            # 创建新条目
            new_entry = KVCacheEntry(key_cache, value_cache, sequence_id)
            new_entry.memory_usage = new_memory_usage
            
            # 添加到缓存
            self.cache[sequence_id] = new_entry
            self.memory_usage += new_memory_usage
            
            logger.debug(f"添加新缓存条目: {sequence_id}, 序列长度: {key_cache.size(2)}, 内存使用: {new_memory_usage:.2f}MB")
    
    def remove(self, sequence_id: str) -> None:
        """移除缓存条目"""
        if sequence_id in self.cache:
            entry = self.cache.pop(sequence_id)
            self.memory_usage -= entry.memory_usage
            logger.debug(f"移除缓存条目: {sequence_id}, 释放内存: {entry.memory_usage:.2f}MB")
    
    def clear(self) -> None:
        """清空缓存"""
        released_memory = self.memory_usage
        self.cache.clear()
        self.memory_usage = 0
        self.stats["memory_reclaims"] += released_memory
        logger.info(f"清空缓存, 释放内存: {released_memory:.2f}MB")
    
    def resize(self, new_size: int) -> None:
        """调整缓存大小"""
        self.max_size = new_size
        
        # 如果当前缓存条目数超过新大小，淘汰多余的条目
        while len(self.cache) > self.max_size:
            self._evict_oldest()
        
        logger.info(f"调整缓存大小: {new_size}, 当前条目数: {len(self.cache)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        hit_rate = self.stats["hits"] / (self.stats["hits"] + self.stats["misses"] + 1e-9) * 100
        
        return {
            "size": len(self.cache),
            "memory_usage": self.memory_usage,
            "max_memory": self.max_memory_mb,
            "hit_rate": hit_rate,
            **self.stats
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息（与get_stats方法相同）"""
        return self.get_stats()
    
    def trim_cache(self, max_entries: int = None) -> None:
        """修剪缓存，保留最近使用的条目"""
        if max_entries is None:
            max_entries = max(1, self.max_size // 2)
        
        # 只保留最近使用的max_entries个条目
        while len(self.cache) > max_entries:
            self._evict_oldest()
        
        logger.info(f"修剪缓存完成, 保留条目数: {len(self.cache)}")
    
    def optimize_memory(self, target_usage_percent: float = 0.7) -> float:
        """优化内存使用，释放不常用的缓存
        
        Args:
            target_usage_percent: 目标内存使用率
            
        Returns:
            释放的内存量（MB）
        """
        if self.max_memory_mb is None:
            return 0.0
        
        target_memory = self.max_memory_mb * target_usage_percent
        released_memory = 0.0
        
        # 如果当前内存使用超过目标，释放内存
        if self.memory_usage > target_memory:
            # 按访问次数排序，优先保留高频访问的条目
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: (x[1].access_count, x[1].last_accessed),
                reverse=True
            )
            
            # 重新构建缓存，只保留重要条目
            new_cache = OrderedDict()
            new_memory_usage = 0.0
            
            for sequence_id, entry in sorted_entries:
                if new_memory_usage + entry.memory_usage <= target_memory:
                    new_cache[sequence_id] = entry
                    new_memory_usage += entry.memory_usage
                else:
                    released_memory += entry.memory_usage
            
            # 更新缓存
            self.cache = new_cache
            released = self.memory_usage - new_memory_usage
            self.memory_usage = new_memory_usage
            
            if released > 0:
                self.stats["memory_reclaims"] += released
                logger.info(f"内存优化完成: 释放内存={released:.2f}MB, 新内存使用={self.memory_usage:.2f}MB")
        
        return released

# 全局KV缓存实例
_global_kv_cache = None

def initialize_global_kv_cache(max_size: int = 10, max_memory_mb: int = 2000) -> KVCache:
    """初始化全局KV缓存实例
    
    Args:
        max_size: 缓存最大条目数
        max_memory_mb: 缓存最大内存占用（MB）
        
    Returns:
        KVCache实例
    """
    global _global_kv_cache
    if _global_kv_cache is None:
        _global_kv_cache = KVCache(max_size=max_size, max_memory_mb=max_memory_mb)
    return _global_kv_cache

def get_global_kv_cache(max_size: int = 10, max_memory_mb: int = 2000) -> KVCache:
    """获取全局KV缓存实例
    
    Args:
        max_size: 缓存最大条目数
        max_memory_mb: 缓存最大内存占用（MB）
        
    Returns:
        KVCache实例
    """
    global _global_kv_cache
    if _global_kv_cache is None:
        _global_kv_cache = KVCache(max_size=max_size, max_memory_mb=max_memory_mb)
    return _global_kv_cache