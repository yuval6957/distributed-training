from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="distributed-training",
    version="1.0.0",
    author="Distributed Training Framework Contributors",
    description="A distributed training framework for machine learning with CPU/GPU split architectures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/distributed-training",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
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
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "colab": ["pyngrok>=6.0.0"],
        "dev": ["pytest", "black", "flake8", "mypy"],
    },
    entry_points={
        "console_scripts": [
            "distributed-training-test=distributed_training.tests.test_connection:test_network",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)