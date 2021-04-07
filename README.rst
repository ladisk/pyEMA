pyEMA
=====

Experimental and operational modal analysis

Check out the `documentation`_.

Basic usage
-----------

Make an instance of ``Model`` class:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   a = pyema.Model(
       frf_matrix,
       frequency_array,
       lower=50,
       upper=10000,
       pol_order_high=60
       )

Compute poles:
~~~~~~~~~~~~~~

.. code:: python

   a.get_poles()

Determine correct poles:
~~~~~~~~~~~~~~~~~~~~~~~~

The stable poles can be determined in two ways: 

1. Display **stability chart** (deprecated) 

.. code:: python

    a.stab_chart()

or use the new function that also contains the stability chart and more:

.. code:: python
    
    a.select_poles()

The stability chart displayes calculated poles and the user can hand-pick the stable ones. Reconstruction is done on-the-fly. 
In this case the reconstruction is not necessary since the user can access the FRF matrix and modal constant matrix:

.. code:: python

    a.H # FRF matrix     
    a.A # modal constants matrix

2. If the approximate values of natural frequencies are already known, it is not necessary to display the stability chart:

.. code:: python

    approx_nat_freq = [314, 864]     
    a.select_closest_poles(approx_nat_freq)

In this case, the reconstruction is not computed. ``get_constants`` must be called (see below).

Natural frequencies and damping coefficients can now be accessed:

.. code:: python

   a.nat_freq # natrual frequencies
   a.nat_xi # damping coefficients

Reconstruction:
~~~~~~~~~~~~~~~

There are two types of reconstruction possible: 

1. Reconstruction using **own** poles:

.. code:: python

    H, A = a.get_constants(whose_poles='own', FRF_ind='all')

where **H** is reconstructed FRF matrix and **A** is a matrix of modal constants.

2. Reconstruction on **c** using poles from **a**:

.. code:: python

    c = pyema.Model(frf_matrix, frequency_array, lower=50, upper=10000, pol_order_high=60)

    H, A = c.get_constants(whose_poles=a, FRF_ind=‘all’) 

|DOI|
|Build Status|

.. _documentation: https://pyema.readthedocs.io/en/latest/basic_usage.html

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4016671.svg?
   :target: https://doi.org/10.5281/zenodo.4016671

.. |Build Status| image:: https://travis-ci.com/ladisk/pyEMA.svg?branch=master
   :target: https://travis-ci.com/ladisk/pyEMA



