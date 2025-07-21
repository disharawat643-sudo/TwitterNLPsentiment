"""
Setup configuration for Twitter Sentiment Analysis package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="twitter-sentiment-analysis",
    version="1.0.0",
    authors=[
        "Sarthak Singh",
        "Himanshu Majumdar", 
        "Samit Singh Bag",
        "Sahil Raghav"
    ],
    author_email="sarthaksingh02.sudo@gmail.com",
    description="A comprehensive machine learning project for analyzing sentiment in Twitter data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sarthaksingh02-sudo/Twitter-Sentiment-Analysis",
    project_urls={
        "Bug Tracker": "https://github.com/sarthaksingh02-sudo/Twitter-Sentiment-Analysis/issues",
        "Documentation": "https://github.com/sarthaksingh02-sudo/Twitter-Sentiment-Analysis/blob/main/README.md",
        "Source Code": "https://github.com/sarthaksingh02-sudo/Twitter-Sentiment-Analysis",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
        ],
        "api": [
            "flask>=2.0.0",
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "twitter-sentiment=src.sentiment_analyzer:main",
        ],
    },
    include_package_data=True,
    keywords="sentiment analysis, twitter, nlp, machine learning, text classification",
)
