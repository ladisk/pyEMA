Basic usage
===========
A case of typical usage is presented here.

Instance of the `Model` class
-----------------------------
First make a ``Model`` object:

::

    a = pyEMA.Model(
        frf_matrix,
        frequency_array,
        lower=10,
        upper=10000,
        pol_order_high=60
    )

Compute the poles
-----------------
Compute the poles of the given FRFs:
::

    a.get_poles()

Select stable poles
-------------------

After the poles are computed, the stable ones must be selected. To select stable poles, two ways are possible.

Option 1: Display the **stability chart**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    a.stab_chart()

The stability chart displayes calculated poles and the user can hand-pick the stable ones. 
Reconstruction is done on-the-fly. In this case the reconstruction is not necessary since the user can access FRF matrix and modal constant matrix: 
::

    a.H # reconstructed FRF matrix
    a.A # modal constants

Option 2: Use automatic selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the approximate values of natural frequencies are already known, it is not necessary to display the stability chart:
::

    approx_nat_freq = [314, 864]
    a.select_closest_poles(approx_nat_freq)

In this case the reconstruction and modal constants are not computed. ``get_constants`` method must be called (see below).

Reconstruction
--------------

There are two types of reconstruction possible:

Option 1: Reconstruction on own poles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    H, A = a.get_constants()

**H** is reconstructed FRF matrix and **A** is a matrix of modal constants.

Option 2: Reconstruction on ``c`` usign poles from ``a``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Create a new object using different set of FRFs:
::

    c = pyEMA.Model(
        frf_matrix,
        frequency_array,
        lower=10,
        upper=10000,
        pol_order_high=60
    )

Compute reconstruction based on poles determined on object ``a``:
::

    H, A = c.get_constants(whose_poles=a)