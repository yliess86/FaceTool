from setuptools import find_packages
from setuptools import setup

import os
import sys


__version__ = "0.0.1"


desciption = ( 
    f"FaceCrop: a Tool for Face Cropped Videos"
)

with open('requirements.txt') as fh:
    requirements = fh.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="facecrop",
    version=__version__,
    author="Yliess HATI",
    author_email="hatiyliess86@gmail.com",
    description=desciption,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yliess86/FaceCrop",
    python_requires=">=3.6",
    install_requires=requirements,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)