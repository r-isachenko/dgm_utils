from distutils.core import setup
from setuptools import find_packages
import sys


if sys.version_info < (3, 10):
    exit_string = (
        'Sorry, Python < 3.10 is not supported. '
        f'Current verion is {".".join(map(str, sys.version_info[:2]))}'
    )
    sys.exit(exit_string)

setup(
    name="dgm_utils",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pillow",
        "tqdm",
        "torch",
        "torchvision",
        "torchaudio",
        "gdown",
        "scikit-learn",
        "torchdiffeq",
        "torch-ema",
        "pot"
    ],
)
