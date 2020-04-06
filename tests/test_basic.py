import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyEMA

  
def test_1():
    freq, H1_main = np.load("./data/acc_data.npy", allow_pickle=True)
    FRF = H1_main[:,1,:]
    freq = freq
    acc = pyEMA.Model(frf=FRF, freq=freq, lower=10, 
                    upper=5000, pol_order_high=60)
    
    acc.get_poles()
    
    n_freq = [176, 476, 932, 1534, 2258, 3161, 4180]
    acc.select_closest_poles(n_freq)

    nat_freq_true = np.array([176.07332578,  476.4634351, 932.28540465, 1534.78951957, 
            2286.31989538, 3162.15866336, 4181.71710178])
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

    mac_true = np.array([[1.00000000e+00, 9.02515351e-03, 3.71827973e-02, 1.16220047e-03,
        2.44205973e-01, 4.52598141e-03, 1.81037366e-01],
       [9.02515351e-03, 1.00000000e+00, 8.49283153e-04, 8.35112639e-02,
        3.00653562e-03, 9.58444691e-02, 2.59570265e-03],
       [3.71827973e-02, 8.49283153e-04, 1.00000000e+00, 2.73984973e-03,
        4.95162216e-01, 1.72581774e-03, 6.16883253e-01],
       [1.16220047e-03, 8.35112639e-02, 2.73984973e-03, 1.00000000e+00,
        1.58600848e-03, 5.02742321e-01, 2.58771991e-03],
       [2.44205973e-01, 3.00653562e-03, 4.95162216e-01, 1.58600848e-03,
        1.00000000e+00, 6.98982327e-03, 9.59439695e-01],
       [4.52598141e-03, 9.58444691e-02, 1.72581774e-03, 5.02742321e-01,
        6.98982327e-03, 1.00000000e+00, 4.68264434e-05],
       [1.81037366e-01, 2.59570265e-03, 6.16883253e-01, 2.58771991e-03,
        9.59439695e-01, 4.68264434e-05, 1.00000000e+00]])

    assert np.allclose(mac_true, acc.autoMAC())

if __name__ == '__main__':
    test_1()

if __name__ == '__mains__':
    np.testing.run_module_suite()