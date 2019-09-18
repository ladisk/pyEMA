#!/usr/bin/env python
# -*- coding: utf-8 -*-
desc = """\
Experimental and operational modal analysis
===========================================
This module supports experimental and operational modal analysis. 

For the showcase see: https://github.com/ladisk/pyEMA/blob/master/pyEMA%20Showcase.ipynb

Documentation is located here: https://pyema.readthedocs.io/en/latest/
"""

from setuptools import setup
setup(name='pyEMA',
      version='0.21',
      author='Klemen Zaletelj, Tomaž Bregar, Domen Gorjup, Janko Slavič',
      author_email='janko.slavic@fs.uni-lj.si, ladisk@gmail.com',
      description='Experimental and operational modal analysis.',
      url='https://github.com/ladisk/pyEMA',
      packages=['pyEMA'],
      long_description=desc,
      install_requires=['numpy>=1.15.4', 'tqdm', 'scipy>=1.1.0', 'matplotlib>=3.0.0']
      )