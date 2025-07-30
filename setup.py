from setuptools import setup, find_packages

setup(
    name="movielens-recommendation-system",
    version="0.1.0",
    author="Avery",
    author_email="zeglow2023@gmail.com",
    description="A comprehensive movie recommendation system built from scratch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Zeglow/movielens-recommendation-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
    ],
)