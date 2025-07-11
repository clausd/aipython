from setuptools import setup, find_packages

setup(
    name="aipython",
    version="0.1.0",
    description="AI-enhanced IPython with natural language code generation",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "ipython>=7.0.0",
        "claudette>=0.0.1",
    ],
    entry_points={
        'console_scripts': [
            'aipython=aipython.cli:main',
        ],
    },
    python_requires=">=3.8",
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
)
