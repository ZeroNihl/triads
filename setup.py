#!/usr/bin/env python3

from pathlib import Path
from setuptools import setup

# Read long description from README.md
directory = Path(__file__).resolve().parent
with open(directory / 'README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='cognicum',  # Replace with your project name
    version='0.1.0',  # Your project's version
    description='A project that uses tinygrad and automated docs',
    author='Your Name',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',

    # Replace with your project’s Python packages
    packages=['cognicum'], # Add your own packages

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],

    # Python version requirement
    python_requires='>=3.8',

    # Dependencies required for the project to run
    install_requires=[
        "tinygrad>=0.9.2",  # Specify tinygrad as a dependency
    ],

    # Optional dependencies for docs and testing
    extras_require={
        'docs': [
            "mkdocs",
            "mkdocs-material",
            "mkdocstrings[python]",
            "markdown-callouts",
            "markdown-exec[ansi]",
            "black",
            "numpy",
        ],
        'testing': [
            "pytest",
            "numpy",
        ],
    },

    include_package_data=True,
)
