"""
C语言采样器的Python包装器
提供与Python版本sampler.py相同的接口
"""
import ctypes
import ctypes.util
import numpy as np
from typing import Optional
import os

# 加载C库
def load_sampler_library():
    """加载sampler动态链接库"""
    # 尝试多个可能的库路径
    possible_paths = [
        os.path.join(os.path.dirname(__file__), 'libsampler.dylib'),
        os.path.join(os.path.dirname(__file__), 'libsampler.so'),
        'libsampler.dylib',
        'libsampler.so',
    ]
    
    for path in possible_paths:
        try:
            lib = ctypes.CDLL(path)
            return lib
        except:
            continue
    
    raise ImportError(
        "无法加载sampler C库。请确保已编译sampler.c并生成动态链接库。\n"
        "编译命令: gcc -shared -fPIC -o libsampler.dylib sampler.c -lm"
    )

# 加载库
try:
    _lib = load_sampler_library()
except ImportError as e:
    raise ImportError(str(e))

# 定义C类型
SamplerHandle = ctypes.c_void_p
SamplingStrategy = ctypes.c_int

# 定义性能统计结构
class SamplerStats(ctypes.Structure):
    _fields_ = [
        ("total_samples", ctypes.c_uint64),
        ("total_time", ctypes.c_double),
        ("greedy_samples", ctypes.c_uint64),
        ("temperature_samples", ctypes.c_uint64),
        ("topk_samples", ctypes.c_uint64),
        ("topp_samples", ctypes.c_uint64),
        ("average_sample_time", ctypes.c_double),
        ("samples_per_second", ctypes.c_double),
    ]

# 设置C函数参数和返回类型
_lib.sampler_create.argtypes = [ctypes.c_int]
_lib.sampler_create.restype = SamplerHandle

_lib.sampler_destroy.argtypes = [SamplerHandle]
_lib.sampler_destroy.restype = None

_lib.sampler_sample.argtypes = [SamplerHandle, 
                                 ctypes.POINTER(ctypes.c_float), 
                                 ctypes.c_int,
                                 ctypes.c_float, 
                                 ctypes.c_float, 
                                 ctypes.c_int]
_lib.sampler_sample.restype = ctypes.c_int

_lib.sampler_sample_greedy.argtypes = [SamplerHandle, 
                                        ctypes.POINTER(ctypes.c_float), 
                                        ctypes.c_int]
_lib.sampler_sample_greedy.restype = ctypes.c_int

_lib.sampler_sample_temperature.argtypes = [SamplerHandle, 
                                             ctypes.POINTER(ctypes.c_float), 
                                             ctypes.c_int,
                                             ctypes.c_float]
_lib.sampler_sample_temperature.restype = ctypes.c_int

_lib.sampler_sample_topk.argtypes = [SamplerHandle, 
                                     ctypes.POINTER(ctypes.c_float), 
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_float]
_lib.sampler_sample_topk.restype = ctypes.c_int

_lib.sampler_sample_topp.argtypes = [SamplerHandle, 
                                      ctypes.POINTER(ctypes.c_float), 
                                      ctypes.c_int,
                                      ctypes.c_float,
                                      ctypes.c_float]
_lib.sampler_sample_topp.restype = ctypes.c_int

_lib.sampler_get_stats.argtypes = [SamplerHandle, ctypes.POINTER(SamplerStats)]
_lib.sampler_get_stats.restype = None

_lib.sampler_reset_stats.argtypes = [SamplerHandle]
_lib.sampler_reset_stats.restype = None

_lib.sampler_set_greedy_threshold.argtypes = [SamplerHandle, ctypes.c_float]
_lib.sampler_set_greedy_threshold.restype = None

_lib.sampler_set_fast_topk_threshold.argtypes = [SamplerHandle, ctypes.c_int]
_lib.sampler_set_fast_topk_threshold.restype = None

_lib.sampler_set_vocab_limit.argtypes = [SamplerHandle, ctypes.c_int]
_lib.sampler_set_vocab_limit.restype = None


class CSampler:
    """C语言采样器的Python包装器
    
    提供与Python版本sampler.py相同的接口，但使用C语言实现以获得更好的性能。
    """
    
    def __init__(self, vocab_size: int = 151936):
        """
        初始化C采样器
        
        Args:
            vocab_size: 词汇表大小
        """
        self.vocab_size = vocab_size
        self.handle = _lib.sampler_create(vocab_size)
        
        if not self.handle:
            raise RuntimeError("Failed to create C sampler")
    
    def __del__(self):
        """析构函数，释放C采样器资源"""
        if hasattr(self, 'handle') and self.handle:
            _lib.sampler_destroy(self.handle)
    
    def sample(self, logits: np.ndarray, temperature: float = 0.7, 
               top_p: float = 0.9, top_k: int = 50) -> int:
        """
        主采样函数 - 自适应策略选择
        
        Args:
            logits: Logits数组，形状为[vocab_size]或[1, vocab_size]
            temperature: 采样温度
            top_p: Nucleus采样阈值
            top_k: Top-K采样阈值
            
        Returns:
            采样的token ID
        """
        # 确保logits是一维的
        if logits.ndim > 1:
            if logits.shape[0] == 1:
                logits = logits[0]
            else:
                logits = logits[-1]
        
        # 转换为float32类型
        logits = logits.astype(np.float32)
        
        # 确保是连续的内存布局
        if not logits.flags['C_CONTIGUOUS']:
            logits = np.ascontiguousarray(logits)
        
        # 调用C函数
        token_id = _lib.sampler_sample(
            self.handle,
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(logits),
            temperature,
            top_p,
            top_k
        )
        
        return token_id
    
    def sample_greedy(self, logits: np.ndarray) -> int:
        """
        贪婪采样 - 选择概率最高的token
        
        Args:
            logits: Logits数组
            
        Returns:
            采样的token ID
        """
        # 确保logits是一维的
        if logits.ndim > 1:
            logits = logits[0] if logits.shape[0] == 1 else logits[-1]
        
        logits = logits.astype(np.float32)
        
        if not logits.flags['C_CONTIGUOUS']:
            logits = np.ascontiguousarray(logits)
        
        return _lib.sampler_sample_greedy(
            self.handle,
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(logits)
        )
    
    def sample_temperature(self, logits: np.ndarray, temperature: float) -> int:
        """
        温度采样
        
        Args:
            logits: Logits数组
            temperature: 采样温度
            
        Returns:
            采样的token ID
        """
        if logits.ndim > 1:
            logits = logits[0] if logits.shape[0] == 1 else logits[-1]
        
        logits = logits.astype(np.float32)
        
        if not logits.flags['C_CONTIGUOUS']:
            logits = np.ascontiguousarray(logits)
        
        return _lib.sampler_sample_temperature(
            self.handle,
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(logits),
            temperature
        )
    
    def sample_topk(self, logits: np.ndarray, top_k: int, temperature: float = 1.0) -> int:
        """
        Top-K采样
        
        Args:
            logits: Logits数组
            top_k: Top-K阈值
            temperature: 采样温度
            
        Returns:
            采样的token ID
        """
        if logits.ndim > 1:
            logits = logits[0] if logits.shape[0] == 1 else logits[-1]
        
        logits = logits.astype(np.float32)
        
        if not logits.flags['C_CONTIGUOUS']:
            logits = np.ascontiguousarray(logits)
        
        return _lib.sampler_sample_topk(
            self.handle,
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(logits),
            top_k,
            temperature
        )
    
    def sample_topp(self, logits: np.ndarray, top_p: float, temperature: float = 1.0) -> int:
        """
        Top-P (Nucleus)采样
        
        Args:
            logits: Logits数组
            top_p: Top-P阈值
            temperature: 采样温度
            
        Returns:
            采样的token ID
        """
        if logits.ndim > 1:
            logits = logits[0] if logits.shape[0] == 1 else logits[-1]
        
        logits = logits.astype(np.float32)
        
        if not logits.flags['C_CONTIGUOUS']:
            logits = np.ascontiguousarray(logits)
        
        return _lib.sampler_sample_topp(
            self.handle,
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(logits),
            top_p,
            temperature
        )
    
    def get_stats(self) -> dict:
        """
        获取性能统计信息
        
        Returns:
            包含性能统计的字典
        """
        stats = SamplerStats()
        _lib.sampler_get_stats(self.handle, ctypes.byref(stats))
        
        return {
            'total_samples': stats.total_samples,
            'total_time': stats.total_time,
            'greedy_samples': stats.greedy_samples,
            'temperature_samples': stats.temperature_samples,
            'topk_samples': stats.topk_samples,
            'topp_samples': stats.topp_samples,
            'average_sample_time': stats.average_sample_time,
            'samples_per_second': stats.samples_per_second,
        }
    
    def reset_stats(self):
        """重置性能统计"""
        _lib.sampler_reset_stats(self.handle)
    
    def set_greedy_threshold(self, threshold: float):
        """
        设置贪婪采样阈值
        
        Args:
            threshold: 温度阈值，低于此值使用贪婪采样
        """
        _lib.sampler_set_greedy_threshold(self.handle, threshold)
    
    def set_fast_topk_threshold(self, threshold: int):
        """
        设置快速Top-K阈值
        
        Args:
            threshold: Top-K阈值，低于此值使用快速Top-K采样
        """
        _lib.sampler_set_fast_topk_threshold(self.handle, threshold)
    
    def set_vocab_limit(self, limit: int):
        """
        设置词汇表限制
        
        Args:
            limit: 词汇表限制，用于限制词汇表采样
        """
        _lib.sampler_set_vocab_limit(self.handle, limit)


def create_sampler(vocab_size: int = 151936) -> CSampler:
    """
    创建C采样器实例的工厂函数
    
    Args:
        vocab_size: 词汇表大小
        
    Returns:
        CSampler实例
    """
    return CSampler(vocab_size)


# 兼容性：如果直接导入此模块，提供便捷函数
if __name__ == "__main__":
    # 简单测试
    print("C语言采样器Python包装器")
    print("=" * 50)
    
    try:
        sampler = CSampler(vocab_size=100)
        print("✓ C采样器创建成功")
        
        # 生成测试logits
        np.random.seed(42)
        logits = np.random.randn(100).astype(np.float32)
        
        # 测试各种采样方法
        greedy_token = sampler.sample_greedy(logits)
        print(f"✓ 贪婪采样: token_id = {greedy_token}")
        
        temp_token = sampler.sample_temperature(logits, temperature=0.8)
        print(f"✓ 温度采样: token_id = {temp_token}")
        
        topk_token = sampler.sample_topk(logits, top_k=10, temperature=1.0)
        print(f"✓ Top-K采样: token_id = {topk_token}")
        
        topp_token = sampler.sample_topp(logits, top_p=0.9, temperature=1.0)
        print(f"✓ Top-P采样: token_id = {topp_token}")
        
        auto_token = sampler.sample(logits, temperature=0.7, top_k=50, top_p=0.9)
        print(f"✓ 自适应采样: token_id = {auto_token}")
        
        # 获取统计信息
        stats = sampler.get_stats()
        print(f"\n性能统计:")
        print(f"  总采样数: {stats['total_samples']}")
        print(f"  平均采样时间: {stats['average_sample_time']*1000:.4f} ms")
        print(f"  采样速度: {stats['samples_per_second']:.2f} samples/s")
        
        print("\n✓ 所有测试通过！")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
