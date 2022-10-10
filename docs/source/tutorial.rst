The tutorial
========
A case of typical usage is presented here.

To use real code, take a look at the `showcase notebook <https://github.com/ladisk/pyEMA/blob/master/pyEMA%20Showcase.ipynb>`_.

Instance of the ``Model`` class
-------------------------------

.. code:: python

    a = pyEMA.Model(
        frf_matrix,
        frequency_array,
        lower=10,
        upper=10000,
        pol_order_high=60
    )

``frf_matrix`` and ``frequency vector`` arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The main inputs in the ``Model`` class are the Frequency Response Function (``frf_matrix``) with shape ``(n_locations, n_frequencies)`` and
the frequency vector (``frequency_vector``) with shape ``(n_frequencies,)``.

``lower`` and ``upper`` arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To set the frequency limits on the ``frf_matrix``, the ``lower`` and ``upper`` arguments are used. The poles will be combuted based on the 
``frf_matric`` within these limits, however, the reconstruction will be compted between the limits ``(0, upper)``.

``pol_order_high`` argument
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This argument determines the highest polynomial order used to approximate the ``frf_matrix``.

``driving_point`` argument
~~~~~~~~~~~~~~~~~~~~~~~~~~
If given, the ``driving_point`` is the index of the ``frf_matrix`` of the driving location. This location is used to scale the modal constants to
the modal shapes.

.. math::

   \phi_k = \frac{A_k}{\sqrt{A_{k, j}}}

where ``j`` is the driving point index.

For futhera description of the arguments, see :ref:`code documentation <code-doc-Model>`.

Compute the poles
-----------------
This step must always be carried out. The increasing polynomial order (up to ``pol_order_high``) is used to approximate the FRFs.

.. code:: python

    a.get_poles()

Select stable poles
-------------------

After the poles are computed, the stable ones must be selected. To select stable poles, two ways are possible.

Option 1: Display the **stability chart**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    a.select_poles()

Option 2: Use automatic selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the approximate values of natural frequencies are already known, it is not necessary to display the stability chart:

.. code:: python

    approx_nat_freq = [314, 864]
    a.select_closest_poles(approx_nat_freq)

Reconstruction
--------------

To identify the modal constants, the ``get_constants()`` method must be called. The method currently supports two methods, 
``lsfd`` and ``lsfd_proportional``. Both methods are based on the Least-Squares Frequency Domain method, however, the ``lsfd_proportional``
assumes the proportional damping and thus return real-values modal constants.

.. code:: python

    H, A = a.get_constants(method='lsfd')

The method returns the reconstructed FRFs, ``H``, and the modal constants, ``A``. The lower and upper residuals can also bi accessed through ``LR`` and ``UR``
attributes, respectively.

.. code:: python

    lower_residual = a.LR
    upper_residual = a.UR

If the ``driving_point`` argument was passed to the ``Model`` class, the modal shapes are available through ``phi`` attribute:

.. code:: python

    modal_shapes = a.phi



Reconstruction on ``c`` usign poles from ``a``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``pyEMA`` enables the use of the poles identified using one set of measurments, to identify the modal constants using a different set of measurments.

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


