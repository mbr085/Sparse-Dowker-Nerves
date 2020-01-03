[![Build Status](https://travis-ci.com/mbr085/Sparse-Dowker-Nerves.svg?branch=master)](https://travis-ci.com/mbr085/Sparse-Dowker-Nerves)
[![Documentation Status](https://readthedocs.org/projects/sparse-dowker-nerves/badge/?version=latest)](https://sparse-dowker-nerves.readthedocs.io/en/latest/?badge=latest)

# Sparse-Dowker-Nerves

## Description

An python library for calculating persistent homology from Sparse Dowker Nerves as described in:

> N. Blaser, M. Brun (2019). [Sparse Dowker nerves](https://link.springer.com/article/10.1007/s41468-019-00028-9),

> N. Blaser, M. Brun (2018). [Sparse Filtered Nerves](https://arxiv.org/abs/1810.02149), and

> N. Blaser, M. Brun (2019). [Sparse Nerves in Practice](https://link.springer.com/chapter/10.1007/978-3-030-29726-8_17)

## Prerequisites

This package is tested with python 3.6. 

In order to use it you must install [the miniball package](https://github.com/weddige/miniball). 
In addition you need to install [gudhi](http://gudhi.gforge.inria.fr/python/latest/index.html). 
We recomend you do this in a new anaconda environment as shown below. 

## Installation

To install the latest version of this python library:

    pip install -r requirements.txt
    pip install git+https://github.com/weddige/miniball
    conda install -c conda-forge gudhi 
    pip install git+https://github.com/mbr085/Sparse-Dowker-Nerves

## Getting started

Please have a look at the online [tutorial](https://sparse-dowker-nerves.readthedocs.io/en/latest/tutorial.html). 
