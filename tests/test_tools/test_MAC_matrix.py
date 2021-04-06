import pytest
import numpy as np

from sdypy import EMA as pyEMA

def test_MAC_matrix():
    """
    Test MAC matrix computation on simple complex vector matrices.
    """

    PX = np.zeros((5, 5), dtype=complex)
    PX[::2] = 1.
    
    PY = np.zeros((5, 5), dtype=complex)
    PY[1::2] = 1.
    
    PZ = np.array([
        [0., 1., 0.],
        [0.+1.j, 0. ,0.+1.j]
        ], dtype=complex).T
    
    assert np.allclose(pyEMA.MAC(PX, PY), np.zeros((5, 5)))
    assert np.allclose(pyEMA.MAC(PX, PX), np.ones((5, 5)))
    assert np.allclose(pyEMA.MAC(PZ, PZ), np.identity(2))