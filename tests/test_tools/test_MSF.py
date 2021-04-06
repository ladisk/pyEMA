import pytest
import numpy as np

from sdypy import EMA as pyEMA

def test_MSF():
    eigvec_exp = np.random.rand(10, 4) + 1j*np.random.rand(10, 4)

    vec_alt = eigvec_exp * -13.4
    msf = pyEMA.MSF(eigvec_exp, vec_alt)
    assert np.allclose(msf, -13.4)
    

def test_MSF_single():
    eigvec_exp = np.random.rand(10) + 1j*np.random.rand(10)

    vec_alt = eigvec_exp * -13.4
    msf = pyEMA.MSF(eigvec_exp, vec_alt)
    assert np.allclose(msf, -13.4)