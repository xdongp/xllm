"""
ModelExecutor类的单元测试
"""
import unittest
import torch
import sys
import os

# 将父目录添加到路径中，以便我们可以从xllm包导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock, patch
from xllm.model_executor import ModelExecutor, TransformerLayer


class TestTransformerLayer(unittest.TestCase):
    """TransformerLayer类的测试用例"""
    
    def setUp(self):
        """设置测试夹具"""
        self.layer = TransformerLayer(
            hidden_size=512,
            num_heads=8,
            intermediate_size=1024
        )
    
    def test_transformer_layer_initialization(self):
        """测试TransformerLayer初始化"""
        self.assertIsInstance(self.layer, TransformerLayer)
        self.assertEqual(self.layer.hidden_size, 512)
        self.assertEqual(self.layer.num_heads, 8)
        self.assertEqual(self.layer.intermediate_size, 1024)
    
    def test_transformer_layer_forward(self):
        """测试TransformerLayer前向传递"""
        # 创建样本输入
        batch_size, seq_len, hidden_size = 2, 10, 512
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # 运行前向传递
        output = self.layer(hidden_states)
        
        # 检查输出形状
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))


class TestModelExecutor(unittest.TestCase):
    """ModelExecutor类的测试用例"""
    
    @patch('xllm.model_executor.AutoModelForCausalLM')
    @patch('xllm.model_executor.AutoConfig')
    def setUp(self, mock_auto_config, mock_auto_model):
        """设置测试夹具"""
        # Mock model loading
        mock_config = Mock()
        mock_auto_config.from_pretrained.return_value = mock_config
        
        mock_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model
        
        self.model_executor = ModelExecutor("test-model-path")
    
    def test_model_executor_initialization(self):
        """测试ModelExecutor初始化"""
        self.assertIsInstance(self.model_executor, ModelExecutor)
        self.assertEqual(self.model_executor.model_path, "test-model-path")
        self.assertEqual(self.model_executor.device, torch.device("cpu"))
    
    def test_encode_decode(self):
        """测试编码和解码方法"""
        # Test encode
        text = "Hello"
        encoded = self.model_executor.encode(text)
        self.assertIsInstance(encoded, list)
        
        # 测试解码
        decoded = self.model_executor.decode(encoded)
        self.assertIsInstance(decoded, str)
    
    @patch('xllm.model_executor.AutoModelForCausalLM')
    def test_load_model_with_quantization(self, mock_auto_model):
        """测试带量化的模型加载"""
        # Test INT8 quantization
        mock_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model
        
        model_executor = ModelExecutor("test-model-path", quantization="int8")
        self.assertEqual(model_executor.quantization, "int8")
        
        # Test FP16 quantization
        model_executor = ModelExecutor("test-model-path", quantization="fp16")
        self.assertEqual(model_executor.quantization, "fp16")
    
    def test_forward(self):
        """测试前向方法"""
        # Create sample batch inputs
        batch_inputs = {
            "input_ids": [1, 2, 3, 4, 5],
            "request_positions": [(0, 5)],
            "batch_size": 1
        }
        
        # Mock model output
        mock_logits = torch.randn(5, 32000)
        self.model_executor.model = Mock()
        self.model_executor.model.return_value = Mock(logits=mock_logits)
        
        # Test async forward method
        import asyncio
        async def test_async_forward():
            result = await self.model_executor.forward(batch_inputs)
            self.assertIn("logits", result)
            self.assertIn("request_positions", result)
            self.assertEqual(result["request_positions"], [(0, 5)])
        
        asyncio.run(test_async_forward())


if __name__ == '__main__':
    unittest.main()