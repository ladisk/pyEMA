import pytest
import numpy as np

from sdypy import EMA as pyEMA

def test_complex_freq_to_freq_and_damp():
    f = 13
    x = 0.00324

    fc = -x*2*np.pi*f + 1j*2*np.pi*f * np.sqrt(1-x**2)

    f_, x_ = pyEMA.complex_freq_to_freq_and_damp(fc)

    np.testing.assert_almost_equal(f, f_, 5)
    np.testing.assert_almost_equal(x, x_, 5)