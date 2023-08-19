"""Setup script for codis."""
#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="codis",
    version="0.1.0",
    packages=find_packages(),
    python_requires="~=3.9",
    author="Sebastian Dziadzio",
    author_email="dziadzio@hey.com",
    install_requires=[
        "imageio>=2.28.0",
        "lightning>=2.0.2",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
        "pillow>=9.5.0",
        "opencv-python>=4.8.0",
        "scipy>=1.10.0",
        "scikit-learn",
        "torch>=1.18.0",
        "torchmetrics>=0.11.4",
        "wandb>=0.15.0",
    ],
)
