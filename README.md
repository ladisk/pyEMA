# pyEMA

Experimental and operational modal analysis

Check out the [documentation](https://pyema.readthedocs.io/en/latest/basic_usage.html).

## Basic usage

### Make an instance of `Model` class:
```
a = pyema.Model(
    frf_matrix,
    frequency_array,
    lower=50,
    upper=10000,
    pol_order_high=60
    )
```
### Compute poles:
```
a.get_poles()
```
### Determine correct poles:
The stable poles can be determined in two ways:
1. Display **stability chart**
    ```
    a.stab_chart()
    ```
    The stability chart displayes calculated poles and the user can hand-pick the stable ones. Reconstruction is done on-the-fly. In this case the reconstruction is not necessary since the user can access FRF matrix and modal constant matrix:
    ```
    a.H # FRF matrix
    a.A # modal constants matrix
    ```
2. If the approximate values of natural frequencies are already known, it is not necessary to display the stability chart as it can be computationally expensive:
    ```
    approx_nat_freq = [314, 864]
    a.select_closest_poles(approx_nat_freq)
    ```
    In this case, the reconstruction is not computed. `get_constants` must be called (see below).

Natural frequencies and damping coefficients can now be accessed:
```
a.nat_freq # natrual frequencies
a.nat_xi # damping coefficients
```
### Reconstruction:
There are two types of reconstruction possible:
1. Reconstruction using **own** poles
    ```
    H, A = a.get_constants(whose_poles='own', FRF_ind='all') 
    ```
    **H** is reconstructed FRF matrix and **A** is a matrix of modal constants.

2. Reconstruction on **c** using poles from **a**
    ```
    c = pyema.Model(frf_matrix, 
        frequency_array, 
        lower=50, 
        upper=10000, 
        pol_order_high=60)
        
    H, A = c.get_constants(whose_poles=a, FRF_ind='all')
    ```

[![Build Status](https://travis-ci.com/ladisk/pyEMA.svg?branch=master)](https://travis-ci.com/ladisk/pyEMA)