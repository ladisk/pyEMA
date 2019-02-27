# pyEMA

Experimental and operational modal analysis

## Basic usage

### Make an instance of **lscf** class:
```
a = pyema.lscf(
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
1. Display **stability chart**
    ```
    a.stab_chart(poles='all', legend=False)
    ```
    The stability chart displayes calculated poles and the user can hand-pick the stable ones. Reconstruction is done on-the-fly. In this case the reconstruction is not necessary since the user can access FRF matrix and modal constant matrix:
    ```
    a.H # FRF matrix
    a.A # modal constants matrix
    ```
2. If the approximate values of natural frequencies are already known, it is not necessary to display the stability chart as it is computationally expensive:
    ```
    approx_nat_freq = [314, 864]
    a.select_closest_poles(approx_nat_freq)
    ```
    ### Access the identified natural frequencies and damping coefficients:
    ```
    a.nat_freq # natrual frequencies
    a.nat_xi # damping coefficients
    ```
### Reconstruction:
There are two types of reconstruction possible:
1. Reconstruction using **own** poles
    ```
    H, A = a.lsfd(whose_poles='own', FRF_ind='all') 
    ```
    **H** is reconstructed FRF matrix and **A** is a matrix of modal constants.

2. Reconstruction on **c** using poles from **a**
    ```
    c = pyema.lscf(frf_matrix, 
        frequency_array, 
        lower=50, 
        upper=10000, 
        pol_order_high=60)
        
    H, A = c.lsfd(whose_poles=a, FRF_ind='all')
    ```

[![Build Status](https://travis-ci.com/ladisk/pyEMA.svg?branch=master)](https://travis-ci.com/ladisk/pyEMA)