import unittest
import numpy as np
from union_find import *

class TestSum(unittest.TestCase):

    def test_test(self):
        self.assertEqual(2 + 2, 4)

    def test_single_linkage_label(self):
        K = 6
        P = np.array([[i**2] for i in range(K)])
        mst = np.array([[i,i+1] for i in range(K-1)])
        N = len(P)
        assert(len(mst) == N-1)

        cut_weights = cut_weight(P, mst)
        self.assertEqual(list(cut_weights), [5. * (i+1)**2 for i in range(K-1)])
        res = single_linkage_label(N, mst, cut_weights)
        expected = [[ 0., 1.,   5., 2.],
                    [ 6., 2.,  20., 3.],
                    [ 7., 3.,  45., 4.],
                    [ 8., 4.,  80., 5.],
                    [ 9., 5., 125., 6.]]
        self.assertEqual( [list(node) for node in res], expected)

#    def other(self):
        # P = np.array([[0., 0.], [0., 2.], [0., 3.], [3., 3.]])
        # mst = np.array([[1, 2], [0, 1], [3, 2]])
        
        
if __name__ == '__main__':
    unittest.main()
    


print("OK")
