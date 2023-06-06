pyEMA
=====

Experimental and operational modal analysis.

------------

``pyEMA`` is no **longer develped**. The **successor** of ``pyEMA`` is part of `SDyPy <https://github.com/sdypy/sdypy>`_ project.

All ``pyEMA`` functionality can be used by installing ``sdypy``:

.. code:: cmd
   
   pip install sdypy

and importing the ``EMA`` module:

.. code:: python

   from sdypy import EMA
   
Your existing code can stay exactly the same by only changing the existing import from:

.. code:: python

   import pyEMA
   
to:

.. code:: python

   from sdypy import EMA as pyEMA

------------

Basic ``pyEMA`` usage
---------------------

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

1. Display **stability chart**

.. code:: python
    
    a.select_poles()

The stability chart displayes calculated poles and the user can hand-pick the stable ones.

2. If the approximate values of natural frequencies are already known, it is not necessary to display the stability chart:

.. code:: python

    approx_nat_freq = [314, 864]     
    a.select_closest_poles(approx_nat_freq)

After the stable poles are selected, the natural frequencies and damping coefficients can now be accessed:

.. code:: python

   a.nat_freq # natrual frequencies
   a.nat_xi # damping coefficients

Reconstruction:
~~~~~~~~~~~~~~~

There are two types of reconstruction possible: 

1. Reconstruction using **own** poles (the default option):

.. code:: python

    H, A = a.get_constants(whose_poles='own')

where **H** is reconstructed FRF matrix and **A** is a matrix of modal constants.

2. Reconstruction on **c** using poles from **a**:

.. code:: python

    c = pyema.Model(frf_matrix, frequency_array, lower=50, upper=10000, pol_order_high=60)

    H, A = c.get_constants(whose_poles=a)

|DOI|
|Build Status|

.. _documentation: https://pyema.readthedocs.io/en/latest/index.html

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4016671.svg?
   :target: https://doi.org/10.5281/zenodo.4016671

.. |Build Status| image:: https://travis-ci.com/ladisk/pyEMA.svg?branch=master
   :target: https://travis-ci.com/ladisk/pyEMA

.. _sdypy: https://github.com/sdypy/sdypy

.. _sdypy-EMA: https://github.com/ladisk/sdypy-EMA


