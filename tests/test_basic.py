import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyEMA


  
def test_1():
    freq, H1_main = np.load("./data/acc_data.npy")
    FRF = H1_main[:,1,:]
    freq = freq
    acc = pyEMA.lscf(frf=FRF, freq=freq, lower=50, 
                    upper=1000, pol_order_high=60)
    
    acc.get_poles()
    
    n_freq = [175., 476.]
    acc.select_closest_poles(n_freq)
    np.testing.assert_almost_equal(acc.nat_freq[0], 176.0508429408248)
    np.testing.assert_almost_equal(acc.nat_freq[1], 476.4662361973787)
    
    H, A = acc.lsfd(whose_poles='own', FRF_ind='all')
    assert A.shape[0]==6
    assert A.shape[1]==4
    assert H.shape[0]==6
    assert H.shape[1]==9999



if __name__ == '__main__':
    test_1()

if __name__ == '__mains__':
    np.testing.run_module_suite()