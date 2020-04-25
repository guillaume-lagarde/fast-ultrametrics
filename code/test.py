import unittest
import numpy as np
from union_find import *

class TestSum(unittest.TestCase):

    def test_test(self):
        self.assertEqual(2 + 2, 4)

    def test_single_linkage_label(self):
        P = np.array([[0., 0.], [0., 2.], [0., 3.], [3., 3.]])
        mst = np.array([[1, 2], [0, 1], [3, 2]])
        
        N = len(P)
        assert(len(mst) == N-1)
        
        cut_weights = cut_weight(P, mst)
        res = single_linkage_label(N, mst, cut_weights)

if __name__ == '__main__':
    unittest.main()
    


print("OK")
