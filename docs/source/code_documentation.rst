Code documentation
==================

.. _code-doc-Model:

The ``Model`` class
-------------------

.. autoclass:: pyEMA.Model
    :members:


Stable pole selection
---------------------

The ``SelectPoles`` class is called from the ``Model`` class by calling ``select_poles()`` method:

.. code:: python

    # Initiating the ``Model`` class
    m = Model(...)
    m.get_poles()

    # Selecting the poles
    m.select_poles()

In this case, the ``Model`` argument is passed automatically.

.. autoclass:: pyEMA.pole_picking.SelectPoles
    :members:


Tools
-----

.. automodule:: pyEMA.tools
    :members:


Normal modes
------------

.. autofunction:: pyEMA.normal_modes.complex_to_normal_mode

.. autofunction:: pyEMA.normal_modes._large_normal_mode_approx
