from setuptools import setup, find_packages

setup(
    name="keafnet",
    version="1.0.0",
    description="KEAF-Net: Knowledge-Enhanced Adaptive Fusion Network for VQA",
    packages=find_packages(exclude=["tests", "scripts", "configs", "docs"]),
    python_requires=">=3.9",
    install_requires=["torch>=2.0", "numpy>=1.21", "pyyaml>=6.0"],
    extras_require={
        "full": ["transformers>=4.30", "timm>=0.9"],
        "test": ["pytest>=7.0"],
    },
)
