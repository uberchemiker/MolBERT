import io
import os
import re

from setuptools import find_packages, setup

# Get the version from molbert/__init__.py
# Adapted from https://stackoverflow.com/a/39671214
this_directory = os.path.dirname(os.path.realpath(__file__))
version_matches = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open(f"{this_directory}/molbert/__init__.py", encoding="utf_8_sig").read(),
)
if version_matches is None:
    raise Exception("Could not determine MOLBERT version from __init__.py")
__version__ = version_matches.group(1)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="molbert",
    version=__version__,
    author="BenevolentAI",
    author_email="chemval@benevolent.ai",
    description="Language modelling on chem/bio sequences",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BenevolentAI/MolBERT",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.2",
        "pytorch-lightning>=2.5.5",
        "pyyaml>=6.0",
        "rdkit>=2025.9.1",
        "scikit-learn>=1.5",
        "scipy>=1.12",
        "torch>=2.9",
        "transformers>=4.49",
        "tqdm>=4.66",
    ],
    extras_require={
        "dev": [
            "flake8>=7.1",
            "mypy>=1.10",
            "pytest>=8.3",
            "ruff>=0.6",
        ],
    },
    include_package_data=True,
    zip_safe=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ),
)
