"""
Setup script for Stock Trading Prediction System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements-minimal.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="stock-trading-predictor",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced stock trading prediction system with 80-85% accuracy target",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stock-trading-predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
        ],
        "production": [
            "gunicorn>=20.1.0",
            "redis>=4.5.0",
            "celery>=5.2.0",
            "prometheus-client>=0.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-stock-predictor=main_trading_system:main",
            "download-polygon-data=polygon_downloader:download_2year_data",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.pkl", "*.pth", "*.json"],
    },
)