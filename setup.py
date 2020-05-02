from setuptools import find_packages
from setuptools import setup
from urllib import request

import bz2
import os
import sys


__version__ = "0.2.0"


def shape_predictor() -> str:
    """Donwload and return dlib Predictor Weights File Path"""
    # Define Bases for path and url
    base = "http://dlib.net/files"
    root = os.path.join("facecrop", "model")

    # Define file names
    bz2_file = "shape_predictor_68_face_landmarks.dat.bz2"
    dat_file = "shape_predictor_68_face_landmarks.dat"
    
    # Define file paths and url
    url = f"{base}/{bz2_file}"
    bz2_path = f"{root}/{bz2_file}"
    dat_path = f"{root}/{dat_file}"

    if not os.path.isfile(dat_path):
        os.makedirs(root, exist_ok=True)

        # Read Compressed file
        request.urlretrieve(url, bz2_path)
        with open(bz2_path, "rb") as fh:
            compressed = fh.read()
            decompressed = bz2.decompress(compressed)
        os.remove(bz2_path)
        
        # Write Uncompressed file
        with open(dat_path, "wb") as fh:
            fh.write(decompressed)

    return dat_path


dat_file = shape_predictor()

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
    include_package_data=True,
    data_files=[(
        os.path.join(sys.prefix, "facecrop", "model"), [dat_file]
    )],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)