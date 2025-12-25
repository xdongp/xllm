"""
TokenizerManager类的单元测试
"""
import unittest
import asyncio
import sys
import os

# 将父目录添加到路径中，以便我们可以从xllm包导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock, patch
from xllm.tokenizer_manager import TokenizerManager, RequestState


class TestRequestState(unittest.TestCase):
    """RequestState类的测试用例"""
    
    def test_request_state_initialization(self):
        """使用各种参数测试RequestState初始化"""
        # Test with string stop argument
        request_state = RequestState(
            request_id="test-1",
            prompt="Hello world",
            tokenized_prompt=[1, 2, 3],
            stop="."
        )
        self.assertEqual(request_state.request_id, "test-1")
        self.assertEqual(request_state.prompt, "Hello world")
        self.assertEqual(request_state.tokenized_prompt, [1, 2, 3])
        self.assertEqual(request_state.stop_strings, ["."])
        
        # Test with list stop argument
        request_state = RequestState(
            request_id="test-2",
            prompt="Hello world",
            tokenized_prompt=[1, 2, 3],
            stop=[".", "!"]
        )
        self.assertEqual(request_state.stop_strings, [".", "!"])
        
        # Test with None stop argument
        request_state = RequestState(
            request_id="test-3",
            prompt="Hello world",
            tokenized_prompt=[1, 2, 3],
            stop=None
        )
        self.assertEqual(request_state.stop_strings, [])


class TestTokenizerManager(unittest.TestCase):
    """TokenizerManager类的测试用例"""
    
    def setUp(self):
        """设置测试夹具"""
        self.tokenizer_manager = TokenizerManager("test-model-path")
    
    def test_encode_decode(self):
        """Test encode and decode methods"""
        # 测试编码
        text = "Hello"
        encoded = self.tokenizer_manager.encode(text)
        self.assertIsInstance(encoded, list)
        
        # 测试解码
        decoded = self.tokenizer_manager.decode(encoded)
        self.assertIsInstance(decoded, str)
    
    @patch('xllm.tokenizer_manager.Scheduler')
    def test_generate(self, mock_scheduler):
        """Test generate method"""
        # Mock scheduler
        mock_scheduler_instance = Mock()
        mock_scheduler.return_value = mock_scheduler_instance
        
        # Create new TokenizerManager with mocked scheduler
        tokenizer_manager = TokenizerManager("test-model-path")
        
        # Test generate method
        async def test_async():
            result = await tokenizer_manager.generate(
                prompt="Hello world",
                temperature=0.7,
                max_tokens=10,
                stream=False
            )
            self.assertIsInstance(result, dict)
        
        # Run async test
        asyncio.run(test_async())
    
    def test_fallback_tokenizer(self):
        """Test that fallback tokenizer works when primary tokenizer fails"""
        # 使用假模型路径创建分词器管理器
        tokenizer_manager = TokenizerManager("fake-model-path")
        
        # Tokenizer should be None (fallback mode)
        self.assertIsNone(tokenizer_manager.tokenizer)
        
        # Test that encode/decode still works in fallback mode
        text = "Hello"
        encoded = tokenizer_manager.encode(text)
        decoded = tokenizer_manager.decode(encoded)
        
        self.assertIsInstance(encoded, list)
        self.assertIsInstance(decoded, str)


if __name__ == '__main__':
    unittest.main()