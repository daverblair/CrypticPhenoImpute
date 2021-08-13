#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: davidblair
"""

###

import setuptools
import re

version = re.search(
    '^__version__\s*=\s*"(.*)"',
    open('CrypticPhenoImpute/CrypticPhenoImpute.py').read(),
    re.M).group(1)


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CrypticPhenoImpute",
    version=version,
    author="David Blair",
    author_email="david.blair@ucsf.edu",
    description="Imputes cryptic phenotypes analyzed in Blair et al. into arbitrary clinical datasets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/daverblair/CrypticPhenoImpute",
    packages=["CrypticPhenoImpute"],
    package_data={'CrypticPhenoImpute': ['Data/*','Models/*']},
    entry_points = {
        "console_scripts": ['CrypticPhenoImpute = CrypticPhenoImpute.CrypticPhenoImpute:main']
        },
    install_requires=[
        'argparse>=1.1',
        'numpy>=1.19.0',
        'pandas>=1.0.5',
        'vlpi',
        'wget'
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
