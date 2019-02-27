import unittest

import numpy as np
import pyEMA


class TestPyEMA(unittest.TestCase):
    
    def test_1(self):
        freq, H1_main = np.load("./data/acc_data.npy")
        FRF = H1_main[:,1,:]
        freq = freq
        acc = pyEMA.lscf(frf=FRF, freq=freq, lower=50, 
                        upper=1000, pol_order_high=60)
        
        acc.get_poles()
        
        n_freq = [175., 476.]
        acc.select_closest_poles(n_freq)
        self.assertAlmostEqual(acc.nat_freq[0], 176.0508, 3)
        self.assertAlmostEqual(acc.nat_freq[1], 476.4662, 3)
        
        H, A = acc.lsfd(whose_poles='own', FRF_ind='all')
        self.assertEqual(A.shape[0], 6)
        self.assertEqual(A.shape[1], 4)
        self.assertEqual(H.shape[0], 6)
        self.assertEqual(H.shape[1], 9999)     


if __name__ == '__main__':
    unittest.main()