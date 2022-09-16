Basic usage
===========
A case of typical usage is presented here.

To use real code, take a look at the `showcase notebook <https://github.com/ladisk/pyEMA/blob/master/pyEMA%20Showcase.ipynb>`_.

Instance of the ``Model`` class
-------------------------------
First, make a ``Model`` object:

.. code:: python

    a = pyEMA.Model(
        frf_matrix,
        frequency_array,
        lower=10,
        upper=10000,
        pol_order_high=60
    )

For the description of arguments, see :ref:`code documentation <code-doc-Model>`.

Compute the poles
-----------------
Compute the poles of the given FRFs:

.. code:: python

    a.get_poles()

Select stable poles
-------------------

After the poles are computed, the stable ones must be selected. To select stable poles, two ways are possible.

Option 1: Display the **stability chart**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    a.select_poles()

The stability chart displays the calculated poles and the user can hand-pick the stable ones. 
Reconstruction is done on-the-fly. In this case the reconstruction is not necessary (although can still be done)
since the user can access the FRF matrix and modal constant matrix: 

.. code:: python

    a.H # reconstructed FRF matrix
    a.A # modal constants

Option 2: Use automatic selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the approximate values of natural frequencies are already known, it is not necessary to display the stability chart:

.. code:: python

    approx_nat_freq = [314, 864]
    a.select_closest_poles(approx_nat_freq)

In this case the reconstruction and modal constants are not computed. ``get_constants`` method must be called (see :ref:`below <basic-usage-reconstruction>`).

.. _basic-usage-reconstruction:

Reconstruction
--------------

There are two types of reconstruction possible:

Option 1: Reconstruction on own poles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    H, A = a.get_constants()

**H** is reconstructed FRF matrix and **A** is a matrix of modal constants.

Option 2: Reconstruction on ``c`` usign poles from ``a``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Create a new object using different set of FRFs:

.. code:: python

    c = pyEMA.Model(
        frf_matrix,
        frequency_array,
        lower=10,
        upper=10000,
        pol_order_high=60
    )

Compute reconstruction based on poles determined on object ``a``:

.. code:: python

    H, A = c.get_constants(whose_poles=a)


