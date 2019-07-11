Basic usage
===========
A case of typical usage is presented here.

Instance of the `lscf` class
----------------------------
First make a ``lscf`` object:
::
    a = pyEMA.lscf(
        frf_matrix,
        frequency_array,
        lower=10,
        upper=10000,
        pol_order_high=60
    )

Compute the poles
-----------------
Compute the poles of the given frfs:
::
    a.get_poles(show_progress=True)

Select stable poles
-------------------

After the poles are computed, the stable ones must be selected. To select stable poles, two ways are possible.

1. Display the **stability chart**
::
    a.stab_chart()

The stability chart displayes calculated poles and the user can hand-pick the stable ones. 
Reconstruction is done on-the-fly. In this case the reconstruction is not necessary since the user can access FRF matrix and modal constant matrix: 
::
    a.H # reconstructed FRF matrix
    a.A # modal constants

2. Use automatic selection

If the approximate values of natural frequencies are already known, it is not necessary to display the stability chart as it can be computationally expensive:
::
    approx_nat_freq = [314, 864]
    a.select_closest_poles(approx_nat_freq)

In this case the reconstruction is not computed. ``lsfd`` method must be called (see below).

Reconstruction
--------------

There are two types of reconstruction possible:

1. Reconstruction on own poles:
::
    H, A = a.lsfd()

**H** is reconstructed FRF matrix and **A** is a matrix of modal constants.

2. Reconstruction on ``c`` usign poles from ``a``:
::
    c = pyEMA.lscf(
        frf_matrix,
        frequency_array,
        lower=10,
        upper=10000,
        pol_order_high=60
    )

    H, A = c.lsfd(whose_poles=a)