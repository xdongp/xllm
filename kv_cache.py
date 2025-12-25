"""
KV缓存实现，用于存储和复用注意力键值对以提高推理效率
"""

import torch
import time  # 添加time模块导入
from typing import Dict, List, Tuple, Optional
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
    """KV缓存管理器 - 优化版本"""
    
    def __init__(self, max_cache_size: int = 1000):
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, KVCacheEntry] = {}
        self.access_counter = 0
        self.total_hits = 0  # 总命中次数
        self.total_misses = 0  # 总未命中次数
        self.eviction_count = 0  # 淘汰次数统计
        
        # 用于LRU淘汰的访问时间排序
        self.access_order: List[str] = []
    
    def get(self, sequence_id: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """获取缓存的KV对"""
        if sequence_id in self.cache:
            entry = self.cache[sequence_id]
            # 更新访问记录
            self._update_access(sequence_id)
            self.total_hits += 1
            logger.debug(f"KV cache hit for sequence {sequence_id} (hit count: {entry.hit_count})")
            return entry.key_cache, entry.value_cache
        else:
            self.total_misses += 1
            logger.debug(f"KV cache miss for sequence {sequence_id}")
            return None
    
    def put(self, sequence_id: str, key_cache: torch.Tensor, value_cache: torch.Tensor):
        """存储KV对到缓存"""
        # 如果缓存已满，删除最久未使用的条目
        if len(self.cache) >= self.max_cache_size and sequence_id not in self.cache:
            evicted_key = self._evict_lru()
            if evicted_key:
                self.eviction_count += 1
                logger.debug(f"Evicted cache entry: {evicted_key}")
        
        # 如果条目已存在，更新它
        if sequence_id in self.cache:
            entry = self.cache[sequence_id]
            entry.key_cache = key_cache.clone()
            entry.value_cache = value_cache.clone()
            entry.last_update_time = time.time()
            logger.debug(f"Updated existing KV cache for sequence {sequence_id}")
        else:
            # 创建新条目
            entry = KVCacheEntry(key_cache.clone(), value_cache.clone(), sequence_id)
            self.cache[sequence_id] = entry
            logger.debug(f"Stored new KV cache for sequence {sequence_id}")
        
        # 更新访问记录
        self._update_access(sequence_id)
        self.access_counter += 1
    
    def _update_access(self, sequence_id: str):
        """更新访问记录，用于LRU淘汰"""
        # 更新最后访问时间
        entry = self.cache[sequence_id]
        entry.last_access_time = self.access_counter
        
        # 更新访问顺序列表 - 将当前序列移到最后（表示最近访问）
        if sequence_id in self.access_order:
            self.access_order.remove(sequence_id)
        self.access_order.append(sequence_id)
        
        # 增加命中计数
        entry.hit_count += 1
    
    def _evict_lru(self) -> Optional[str]:
        """淘汰最久未使用的缓存条目 - 优化版本"""
        if not self.cache:
            return None
        
        # 从访问顺序列表中找到最久未访问的条目
        for sequence_id in self.access_order:
            if sequence_id in self.cache:
                # 移除条目
                del self.cache[sequence_id]
                self.access_order.remove(sequence_id)
                return sequence_id
        
        # 如果上面的方法失败，使用原始方法
        if self.cache:
            lru_key = min(self.cache.keys(), 
                         key=lambda k: self.cache[k].last_access_time)
            if lru_key in self.cache:
                del self.cache[lru_key]
                if lru_key in self.access_order:
                    self.access_order.remove(lru_key)
            return lru_key
        
        return None
    
    def remove(self, sequence_id: str):
        """从缓存中移除指定序列"""
        if sequence_id in self.cache:
            del self.cache[sequence_id]
            if sequence_id in self.access_order:
                self.access_order.remove(sequence_id)
            logger.debug(f"Removed cache entry for sequence {sequence_id}")
    
    def clear(self):
        """清空缓存"""
        count = len(self.cache)
        self.cache.clear()
        self.access_order.clear()
        logger.debug(f"Cleared all cache entries ({count} entries)")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        total_requests = self.total_hits + self.total_misses
        hit_rate = self.total_hits / total_requests if total_requests > 0 else 0
        
        return {
            "current_size": len(self.cache),
            "max_size": self.max_cache_size,
            "access_counter": self.access_counter,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": hit_rate,
            "eviction_count": self.eviction_count
        }

    def get_detailed_stats(self) -> Dict[str, any]:
        """获取详细的缓存统计信息"""
        basic_stats = self.get_cache_stats()
        
        # 计算每个条目的详细信息
        entry_details = []
        current_time = time.time()
        for seq_id, entry in self.cache.items():
            entry_details.append({
                "sequence_id": seq_id,
                "hit_count": entry.hit_count,
                "last_access_time": entry.last_access_time,
                "created_time": entry.created_time,
                "last_update_time": entry.last_update_time,
                "age": current_time - entry.created_time,
                "time_since_last_access": current_time - entry.last_access_time if entry.last_access_time > 0 else 0
            })
        
        basic_stats["entries"] = entry_details
        return basic_stats

    def get_hot_entries(self, top_k: int = 10) -> List[Dict[str, any]]:
        """获取最热门的缓存条目"""
        if not self.cache:
            return []
        
        # 按命中次数排序
        sorted_entries = sorted(
            self.cache.items(), 
            key=lambda item: item[1].hit_count, 
            reverse=True
        )
        
        hot_entries = []
        for seq_id, entry in sorted_entries[:top_k]:
            hot_entries.append({
                "sequence_id": seq_id,
                "hit_count": entry.hit_count,
                "last_access_time": entry.last_access_time
            })
        
        return hot_entries

    def resize(self, new_max_size: int):
        """调整缓存大小"""
        if new_max_size < self.max_cache_size:
            # 缩小时需要淘汰多余条目
            while len(self.cache) > new_max_size:
                self._evict_lru()
        
        self.max_cache_size = new_max_size
        logger.debug(f"Resized cache to {new_max_size} entries")


# 全局KV缓存实例
global_kv_cache = KVCache()


def get_global_kv_cache() -> KVCache:
    """获取全局KV缓存实例"""
    return global_kv_cache