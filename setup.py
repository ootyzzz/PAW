#!/usr/bin/env python3
"""
P2W项目安装脚本
"""

from setuptools import setup, find_packages

# 读取README文件
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取requirements文件
with open("requirements_modern.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="p2w-lora-trainer",
    version="0.1.0",
    author="P2W Team",
    author_email="your-email@example.com",
    description="现代化的LoRA训练框架，基于Hugging Face生态",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/P2W",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        "distributed": [
            "deepspeed>=0.12.0",
            "fairscale>=0.4.13",
        ],
        "quantization": [
            "auto-gptq>=0.4.0",
            "optimum>=1.14.0",
            "bitsandbytes>=0.41.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "p2w-train=scripts.train_lora:main",
            "p2w-model=scripts.model_manager:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
