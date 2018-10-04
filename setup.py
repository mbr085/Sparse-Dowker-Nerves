#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dowker homology

An python library for calculating persistent homology from Sparse Dowker Nerves as described in:

N. Blaser, M. Brun (2018). Sparse Dowker Nerves (https://arxiv.org/abs/1802.03655)
N. Blaser, M. Brun (2018). Sparse Filtered Nerves 

Installation

Before installing this this python library, you should install miniball:
    git clone https://github.com/weddige/miniball.git
    cd miniball
    pip install .

To install the latest version of this python library:

    git clone https://github.com/mbr085/Sparse-Dowker-Nerves.git
    cd Sparse-Dowker-Nerves
    pip install .

For better performance phat can be installed via pip:

    pip install phat

"""

# import modules
from setuptools import setup, find_packages

setup(
    name = "dowker_homology", 
    version = "0.0.3", 
    description = "An python library for calculating persistent homology from Sparse Dowker Nerves",
    url = "https://github.com/mbr085/Sparse-Dowker-Nerves", 
    author = "Nello Blaser, Morten Brun", 
    author_email = "nello.blaser@uib.no, morten.brun@uib.no", 
    license = "GPL-3", 
    packages = ["dowker_homology", "dowker_comparison"],
    classifiers = [
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Data scientists',
        'Topic :: Topological data analysis :: Persistent homology',

        # Pick your license as you wish (should match "license" above)
        'License :: GPL-3',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # 'Programming Language :: Python :: 2',
        # 'Programming Language :: Python :: 2.6',
        # 'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        # 'Programming Language :: Python :: 3.2',
        # 'Programming Language :: Python :: 3.3',
        # 'Programming Language :: Python :: 3.4'
    ], 
    keywords = "tda", 
    install_requires = ["numpy", "scipy", "pandas", "pynverse", "matplotlib"]
)
