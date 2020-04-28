import pytest
import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyEMA

from test_data import *


freq, H1_main = np.load("./data/acc_data.npy", allow_pickle=True)
FRF = H1_main[:,1,:]
freq = freq
acc = pyEMA.Model(frf=FRF, freq=freq, lower=10, 
                upper=5000, pol_order_high=60)

acc.get_poles(show_progress=False)

n_freq = [176, 476, 932, 1534, 2258, 3161, 4180]
acc.select_closest_poles(n_freq)
H, A = acc.get_constants(whose_poles='own', FRF_ind='all')
    

def test_data_shape():
    assert A.shape[0] == 6
    assert A.shape[1] == 7
    assert H.shape[0] == 6
    assert H.shape[1] == 4998
    
    assert acc.A.shape[0] == A.shape[0]
    assert acc.A.shape[1] == A.shape[1]
    assert acc.H.shape[0] == H.shape[0]
    assert acc.H.shape[1] == H.shape[1]


def test_natural_frequencies():
    assert np.allclose(np.array(acc.nat_freq), nat_freq_true)


def test_modal_constants_complex():
    assert np.allclose(complex_modes_true, acc.A)
    

def test_normal_modes():
    assert np.allclose(normal_modes_maxdof3_long, pyEMA.normal_modes.complex_to_normal_mode(acc.A, max_dof=3, long=True))
    assert np.allclose(normal_modes_true, acc.normal_mode())
    

def test_autoMAC():
    assert np.allclose(mac_true, acc.autoMAC())

