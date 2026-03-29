#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PatchFusionBERT: Dual-Stream Architecture for Time Series Forecasting
Setup configuration for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='patchfusionbert-forecasting',
    version='1.0.0',
    description='PatchFusionBERT: Dual-Stream Architecture for Long-Term Time Series Forecasting',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Hussein et al.',
    author_email='',
    url='https://github.com/Hussein-experiments/patchfusionbert-forecasting',
    license='MIT',
    packages=find_packages(exclude=['scripts', 'analysis', 'test_results', 'results', 'checkpoints']),
    include_package_data=True,
    install_requires=requirements,
    python_requires='>=3.8,<3.12',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    keywords='time-series forecasting transformer bert patch-based deep-learning',
    project_urls={
        'Source': 'https://github.com/Hussein-experiments/patchfusionbert-forecasting',
        'Bug Reports': 'https://github.com/Hussein-experiments/patchfusionbert-forecasting/issues',
    },
)
