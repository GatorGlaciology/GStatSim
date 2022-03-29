#!/usr/bin/env python

import os
from setuptools import setup, find_packages
from pathlib import Path
import sys

_dir = Path(__file__).resolve().parent

with open(f"{_dir}/README.md") as f:
    long_desc = ""
    try:
        long_desc += f.read()
    except UnicodeDecodeError:
        long_desc += ""


install_requires = ['numpy', 'pandas', 'scipy', 'matplotlib', 'tqdm',
                    'sklearn', 'numba', 'earthpy', 'geostatspy', 'line_profiler', 'gdal', 'statsmodels']

setup(
    name="GlacierStats",
    description="GlacierStats",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/tylern4/GlacierStats.git",
    author="",
    author_email="emackie@ufl.edu",
    packages=find_packages(where='GlacierStats'),
    version='0.1',
    #scripts=['bin/Sequential_Gaussian_Simulation.py'],
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.5",
)