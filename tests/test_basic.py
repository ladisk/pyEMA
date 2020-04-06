import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyEMA

from test_data import *

def test_1():
    freq, H1_main = np.load("./data/acc_data.npy", allow_pickle=True)
    FRF = H1_main[:,1,:]
    freq = freq
    acc = pyEMA.Model(frf=FRF, freq=freq, lower=10, 
                    upper=5000, pol_order_high=60)
    
    acc.get_poles()
    
    n_freq = [176, 476, 932, 1534, 2258, 3161, 4180]
    acc.select_closest_poles(n_freq)

    assert np.allclose(np.array(acc.nat_freq), nat_freq_true)
    
    H, A = acc.get_constants(whose_poles='own', FRF_ind='all')

    assert A.shape[0] == 6
    assert A.shape[1] == 7
    assert H.shape[0] == 6
    assert H.shape[1] == 4998
    
    assert acc.A.shape[0] == A.shape[0]
    assert acc.A.shape[1] == A.shape[1]
    assert acc.H.shape[0] == H.shape[0]
    assert acc.H.shape[1] == H.shape[1]

    assert np.allclose(complex_modes_true, acc.A)
    assert np.allclose(normal_modes_true, pyEMA.tools.complex_to_normal_mode(acc.A))
    assert np.allclose(normal_modes_true, acc.normal_mode())
    assert np.allclose(mac_true, acc.autoMAC())
    
    

if __name__ == '__main__':
    test_1()

if __name__ == '__mains__':
    np.testing.run_module_suite()