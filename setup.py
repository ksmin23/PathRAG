from setuptools import setup, find_packages

setup(
    name="PathRAG",
    version="0.1.0",
    description="PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths",
    author="PathRAG Authors",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "accelerate",
        "aioboto3",
        "aiohttp",
        "graspologic",
        "hnswlib",
        "nano-vectordb",
        "networkx",
        "numpy",
        "ollama",
        "openai",
        "pydantic",
        "tenacity",
        "tiktoken",
        "torch",
        "tqdm",
        "transformers",
    ],
)
