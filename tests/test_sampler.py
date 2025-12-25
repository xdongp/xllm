"""
采样器类的单元测试
"""
import unittest
import torch
import sys
import os

# 将父目录添加到路径中，以便我们可以从xllm包导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xllm.sampler import Sampler


class TestSampler(unittest.TestCase):
    """采样器类的测试用例"""
    
    def setUp(self):
        """设置测试夹具"""
        self.sampler = Sampler()
        # Create sample logits for testing
        self.sample_logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    
    def test_sample_greedy(self):
        """测试贪婪采样（温度=0.0）"""
        token_id = self.sampler.sample(self.sample_logits, temperature=0.0)
        # With greedy sampling, we should always get the token with the highest logit
        self.assertEqual(token_id, 4)  # 最高logit的索引(5.0)
    
    def test_sample_with_temperature(self):
        """测试带温度的采样"""
        # Test with different temperatures
        token_id_low_temp = self.sampler.sample(self.sample_logits, temperature=0.5)
        token_id_high_temp = self.sampler.sample(self.sample_logits, temperature=2.0)
        
        # Both should return valid token IDs (integers within vocabulary size)
        self.assertIsInstance(token_id_low_temp, int)
        self.assertIsInstance(token_id_high_temp, int)
        self.assertGreaterEqual(token_id_low_temp, 0)
        self.assertGreaterEqual(token_id_high_temp, 0)
        self.assertLess(token_id_low_temp, len(self.sample_logits))
        self.assertLess(token_id_high_temp, len(self.sample_logits))
    
    def test_sample_top_k(self):
        """测试带top-k过滤的采样"""
        # Test with top-k = 1 (should behave like greedy sampling)
        token_id = self.sampler.sample(self.sample_logits, temperature=1.0, top_k=1)
        self.assertEqual(token_id, 4)  # 最高logit的索引
        
        # Test with top-k = 3 (should only consider the top 3 tokens)
        token_id = self.sampler.sample(self.sample_logits, temperature=1.0, top_k=3)
        self.assertIsInstance(token_id, int)
        self.assertIn(token_id, [2, 3, 4])  # 应该是前3个令牌之一
    
    def test_sample_top_p(self):
        """测试带top-p（核）过滤的采样"""
        # Test with top-p = 0.9
        token_id = self.sampler.sample(self.sample_logits, temperature=1.0, top_p=0.9)
        self.assertIsInstance(token_id, int)
        self.assertGreaterEqual(token_id, 0)
        self.assertLess(token_id, len(self.sample_logits))
    
    def test_sample_batch(self):
        """测试批量采样"""
        # 创建logits批次
        batch_logits = torch.stack([self.sample_logits, self.sample_logits])
        temperatures = [0.7, 1.0]
        
        token_ids = self.sampler.sample_batch(batch_logits, temperatures)
        
        # 应返回令牌ID列表
        self.assertIsInstance(token_ids, list)
        self.assertEqual(len(token_ids), 2)
        for token_id in token_ids:
            self.assertIsInstance(token_id, int)
    
    def test_sample_beam_search(self):
        """测试束搜索采样"""
        beam_tokens = self.sampler.sample_beam_search(self.sample_logits, beam_width=3)
        
        # Should return a list of token IDs
        self.assertIsInstance(beam_tokens, list)
        self.assertEqual(len(beam_tokens), 3)
        for token_id in beam_tokens:
            self.assertIsInstance(token_id, int)
    
    def test_sample_contrastive_search(self):
        """测试对比搜索采样"""
        token_id = self.sampler.sample_contrastive_search(self.sample_logits)
        
        # Should return a single token ID
        self.assertIsInstance(token_id, int)
        self.assertGreaterEqual(token_id, 0)
        self.assertLess(token_id, len(self.sample_logits))


if __name__ == '__main__':
    unittest.main()