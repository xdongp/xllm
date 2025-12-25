from setuptools import setup, find_packages

setup(
    name="xllm",
    version="0.1.0",
    description="一个基于CPU的大语言模型推理引擎",
    author="xLLM Team",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "pydantic>=1.8.0",
        "transformers>=4.20.0",
        "bitsandbytes>=0.39.0",
    ],
    entry_points={
        "console_scripts": [
            "xllm=xllm.server:main",
        ],
    },
    python_requires=">=3.8",
)