import pytest
import numpy as np

import pyEMA

def test_MCF():
    """
    Test Modal Complexity Factor.
    """

    assert np.alltrue(pyEMA.MCF(np.random.rand(5, 5) - 0.5) == 0)
    assert np.alltrue(pyEMA.MCF((np.random.rand(5, 5) - 0.5)*1j) == 0)
    assert np.alltrue(pyEMA.MCF(np.random.rand(5, 5) -0.5 + (np.random.rand(5, 5) - 0.5)*1j) != 0)

    # Same angle for all components
    a = np.random.rand(10, 10) - 0.5
    mode = a + a*1j
    assert np.alltrue(pyEMA.MCF(mode) == 0)


def test_MCF_single_mode():
    """
    Test for the single mode.
    """
    assert pyEMA.MCF(np.random.rand(10) - 0.5).shape == (1,)
    assert pyEMA.MCF(np.random.rand(10) - 0.5)[0] == 0
    assert pyEMA.MCF((np.random.rand(10) - 0.5)*1j)[0] == 0
    assert pyEMA.MCF(np.random.rand(10) - 0.5 + (np.random.rand(10) - 0.5)*1j)[0] != 0