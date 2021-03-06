import unittest
import numpy as np
from fast_ultrametrics import *
from fast_ultrametrics.distortion import *
from fast_ultrametrics.utils import *

class TestSum(unittest.TestCase):

    def test_test(self):
        self.assertEqual(2 + 2, 4)

    def test_single_linkage_label_squares(self):
        K = 6
        P = np.array([[i**2] for i in range(K)], dtype = np.float64)
        mst = np.array([[i,i+1] for i in range(K-1)])
        N = len(P)
        assert(len(mst) == N-1)

        cut_weights = cut_weight(P, mst)
        self.assertEqual(list(cut_weights), [5. * (i+1)**2 for i in range(K-1)])
        res = single_linkage_label(mst, cut_weights)
        expected = [[ 0., 1.,   5., 2.],
                    [ 6., 2.,  20., 3.],
                    [ 7., 3.,  45., 4.],
                    [ 8., 4.,  80., 5.],
                    [ 9., 5., 125., 6.]]
        self.assertEqual( [list(node) for node in res], expected)

    def test_mst(self):
        K = 10
        P = np.array([[i**2] for i in range(K)], dtype = np.float64)
        edges = spanner(P)
        MST = mst(P, edges)
        # check that the mst is sorted
        self.assertEqual(MST.shape, (K-1, 2))
        for i in range(K-2):
            self.assertTrue(
                dist(P[MST[i][0]], P[MST[i][1]]) <=
                dist(P[MST[i+1][0]], P[MST[i+1][1]])
            )
        # explicit test
        P = np.array(
            [[3.4, 4.5], [2.3, 7.2], [2.2, 2.3], [3.8, 9.9], [4.8, 6.7]],
        dtype = np.float64)
        edges = np.array([[0,1], [0,3], [0,4], [1,2], [1,4], [2,3], [3,4]])
        MST = mst(P, edges)
        expected = [[1, 4], [0, 4], [3, 4], [1, 2]]
        self.assertEqual([ list(edge) for edge in MST ], expected)
        
    def test_single_linkage_label2(self):
        P = np.array([[-150.], [-110.], [-50.], [0.], [1.], [70.]])
        N = len(P)
        mst = np.array([[i,i+1] for i in range(N-1)])

        cut_weights = cut_weight(P, mst)
        self.assertEqual(
            list(cut_weights),
            [5.*40., 5.*150., 5.*50., 5.*1., 5.*80.],
        )
        res = single_linkage_label(mst, cut_weights)
        expected = [[ 3., 4.,   5., 2.],
                    [ 0., 1., 200., 2.],
                    [ 2., 6., 250., 3.],
                    [ 8., 5., 400., 4.],
                    [ 7., 9., 750., 6.]]
        self.assertEqual( [list(node) for node in res], expected)
        
    def test_lsh(self):
        P = np.array([[ 0., 1.,   5., 2.],
                      [ 6., 2.,  20., 3.],
                      [ 7., 3.,  45., 4.],
                      [ 8., 4.,  80., 5.],
                      [ 9., 5., 125., 6.]])
        lsh_balls(2., 4, P, {})

    def test_spanner(self):
        P = np.array([[ 0., 1.,   5., 2.],
                      [ 6., 2.,  20., 3.],
                      [ 7., 3.,  45., 4.],
                      [ 8., 4.,  80., 5.],
                      [ 9., 5., 125., 6.]])
        G = spanner(P, d_min=1, d_max=400)

    def test_spanner_lipschitz(self):
        P = np.array([[ 0., 1.,   5., 2.],
                      [ 6., 2.,  20., 3.],
                      [ 7., 3.,  45., 4.],
                      [ 8., 4.,  80., 5.],
                      [ 9., 5., 125., 6.]])
        G = spanner(P, d_min=1, d_max=400, lsh='lipschitz')
       
    def test_infix_order(self):
        tree = [[ 3., 4.,   5., 2.],
                [ 0., 1., 200., 2.],
                [ 2., 6., 250., 3.],
                [ 8., 5., 400., 4.],
                [ 7., 9., 750., 6.]]
        infix_order(tree)

    def test_rmq(self):
        tree = [[ 3., 4.,   5., 2.],
                [ 0., 1., 200., 2.],
                [ 2., 6., 250., 3.],
                [ 8., 5., 400., 4.],
                [ 7., 9., 750., 6.]]
        rmq = RMQ(tree)
        self.assertEqual(rmq.search(7,3), 10)
        self.assertEqual(rmq.search(2,5), 9)
        self.assertEqual(rmq.search(1,1), 1)
        self.assertEqual(rmq.dist(2,5), 400)
        self.assertEqual(rmq.dist(2,2), 0)

    def test_distortion(self):
        P = np.array([[ 0., 1.,   5., 2.],
                      [ 6., 2.,  20., 3.],
                      [ 7., 3.,  45., 4.],
                      [ 8., 4.,  80., 5.],
                      [ 9., 5., 125., 6.]])
        P = np.genfromtxt("datasets/DIABETES.csv", delimiter=",")
        tree = ultrametric(P)
        well_formed_tree(tree)

    def test_clusters(self):
        P = np.array([[ 0., 1.,   5., 2.],
                      [ 6., 2.,  20., 3.],
                      [ 7., 3.,  45., 4.],
                      [ 8., 4.,  80., 5.],
                      [ 0., 1.,   5., 2.],
                      [ 6., 2.,  20., 3.],
                      [ 7., 3.,  45., 4.],
                      [ 8., 4.,  80., 5.],
                      [ 9., 5., 125., 6.]])
        tree = ultrametric(P)
        well_formed_tree(tree)

def well_formed_tree(tree):
    n = len(tree) + 1
    root = 2 * n - 2
    seen = [False] * (2 * n - 1)
    stack = [ root ]
    while stack:
        node = stack.pop()
        if seen[node]: return False
        seen[node] = True
        if node >= n:
            stack.append(int(tree[node - n][0]))
            stack.append(int(tree[node - n][1]))
    return all(seen)
        
if __name__ == '__main__':
    unittest.main()
