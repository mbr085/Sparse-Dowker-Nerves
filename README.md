[![Build Status](https://travis-ci.com/mbr085/Sparse-Dowker-Nerves.svg?branch=master)](https://travis-ci.com/mbr085/Sparse-Dowker-Nerves)

# Sparse-Dowker-Nerves

## Description

An python library for calculating persistent homology from Sparse Dowker Nerves as described in:
 

> N. Blaser, M. Brun (2018). [Sparse Filtered Nerves](https://arxiv.org/abs/1810.02149)

and

> N. Blaser, M. Brun (2018). [Sparse Dowker Nerves](https://arxiv.org/abs/1802.03655)

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
    pip install git+https://github.com/Sparse-Dowker-Nerves/dowker_homology

## Getting started

Please have a look at the online [tutorial](https://mbr085.github.io/Sparse-Dowker-Nerves). 
