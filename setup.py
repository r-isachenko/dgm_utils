from distutils.core import setup
from setuptools import find_packages
import sys


if sys.version_info < (3, 11):
    sys.exit('Sorry, Python < 3.11 is not supported')

setup(
    name="dgm_utils",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy==2.1.1",
        "matplotlib==3.9.2",
        "pillow==10.4.0",
        "tqdm==4.66.5",
        "torch==2.4.1",
        "torchvision==0.19.1",
        "torchaudio==2.4.1",
        "gdown==5.2.0",
        "scikit-learn==1.5.1"
    ],
)
