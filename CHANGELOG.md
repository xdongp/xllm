# Changelog

All notable changes to xLLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CPU-optimized inference engine for large language models
- Support for Qwen3 and DeepSeek R1 models
- Multiple sampling strategies (Greedy, Temperature, Top-K, Top-P, Beam Search, Contrastive Search)
- Quantization support (INT8, FP16)
- Continuous batching for improved throughput
- RESTful API interface with streaming support
- Performance monitoring and logging
- KV cache optimization
- Radix-based prefix caching
- Demo inference engine with PD (Prompt-Decoder) separation architecture
- Interactive chat interface with template switching
- Comprehensive documentation and tutorials

### Changed
- Optimized model executor for CPU performance
- Improved scheduler with request prioritization
- Enhanced tokenizer manager for better request handling

### Fixed
- Fixed device_map compatibility issues with accelerate library
- Fixed input tensor type errors in PD separation mode
- Fixed attention mask warnings in model generation

## [0.1.0] - 2025-01-XX

### Added
- Initial release of xLLM
- Basic inference engine with CPU support
- Core components: HTTP Server, Tokenizer Manager, Scheduler, Model Executor
- Basic sampling strategies
- Model quantization support
- RESTful API endpoints
- Unit tests and benchmarks
