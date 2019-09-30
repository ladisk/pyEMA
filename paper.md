---
title: 'pyEMA: A Python Package for Experimental and Operational Modal Analysis'
tags:
- Python
- structural dynamics
- experimental modal analysis
- modal identification
- least-squares complex frequency
- least-squares frequency domain
authors:
- name: Klemen Zaletelj
- name: Tomaž Bregar
- name: Domen Gorjup
- name: Janko Slavič
date: 18 September 2019
bibliography: paper.bib
---

# Summary

An important part of structural dynamics is the identification of modal parameters, such as natural frequencies, damping coefficients and modal constants. These parameters are obtained based on Frequency Response Functions (FRFs), that describe the response of one location, normalized by excitation at another. Many methods exist for such identification, namely circle fitting, Ewins-Gleeson method, etc. PyEMA uses a combination of Least-Squares Complex Frequency method and Least-Squares Frequency Domain method to perform the identification.
