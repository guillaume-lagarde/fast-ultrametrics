import unittest
import numpy as np
from union_find import *
K = 6
class TestSum(unittest.TestCase):

    def test_test(self):
        self.assertEqual(2 + 2, 4)

    def test_single_linkage_label(self):
        P = np.array([[i**2] for i in range(K)])
        print("points", P)
        mst = np.array([[i,i+1] for i in range(K-1)])

        print("mst", mst)
        # P = np.array([[0., 0.], [0., 2.], [0., 3.], [3., 3.]])
        # mst = np.array([[1, 2], [0, 1], [3, 2]])
        
        N = len(P)
        assert(len(mst) == N-1)
        print(N)
        
        cut_weights = cut_weight(P, mst)
        print("cut weights: {}".format(cut_weights))
        res = single_linkage_label(N, mst, cut_weights)
        print("res: {}".format(res))

if __name__ == '__main__':
    unittest.main()
    


print("OK")
