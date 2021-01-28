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
        self.assertEqual(list(cut_weights), [((i+1)**2 + i**2) for i in range(K-1)])
        res = single_linkage_label(mst, cut_weights)
        expected = [[ 0., 1.,   1., 2.],
                    [ 6., 2.,  5., 3.],
                    [ 7., 3.,  13., 4.],
                    [ 8., 4.,  25., 5.],
                    [ 9., 5., 41., 6.]]
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
        edges = set([(0,1), (0,3), (0,4), (1,2), (1,4), (2,3), (3,4)])
        MST = mst(P, edges)
        expected = [[1, 4], [0, 4], [3, 4], [1, 2]]
        self.assertEqual([ list(edge) for edge in MST ], expected)
        # Prim mst
        all_edges = set((i, j) for j in range(len(P)) for i in range(j))
        MST = mst(P, all_edges)
        MST2 = exact_mst(P)
        self.assertEqual(sorted([sorted(list(e)) for e in MST2 ]),
                         sorted([sorted(list(e)) for e in MST ]))

    def test_sort_edges(self):
        t = np.array([-1., 0., 24., 42.])
        assert(is_sorted(t))
        t = np.array([-1., 0., 24., 15.])
        assert(not is_sorted(t))
        assert(is_sorted(np.array([])))
        #
        P = np.array([[3., 4.], [2., 7.], [3., 9.]])
        edges = np.array([[0, 1], [0, 2], [2, 1]])
        expected = np.array([[2, 1], [0, 1], [0, 2]])
        sort_edges(P, edges)
        self.assertEqual([ list(x) for x in edges ], [ list(x) for x in expected])
        sort_edges(P, edges)
        self.assertEqual([ list(x) for x in edges ], [ list(x) for x in expected])        
        
    def test_single_linkage_label2(self):
        P = np.array([[-150.], [-110.], [-50.], [0.], [1.], [70.]])
        N = len(P)
        mst = np.array([[i,i+1] for i in range(N-1)])

        cut_weights = cut_weight(P, mst)
        self.assertEqual(
            list(cut_weights),
            [1.0, 40.0, 51.0, 200.0, 220.0],
        )
      #  res = single_linkage_label(mst, cut_weights)
      #  expected = [[ 3., 4.,   1., 2.],
      #              [ 0., 1., 40., 2.],
      #              [ 2., 6., 51., 3.],
      #              [ 8., 5., 200., 4.],
      #              [ 7., 9., 220., 6.]]
      #  self.assertEqual( [list(node) for node in res], expected)

    def test_bounding_balls(self):
        K = 10
        P = np.array([[i**2, (i - K)**3] for i in range(K)], dtype = np.float64)
        tree = ultrametric(P, cut_weights='bounding balls')

    def test_tree_structure(self):
        P = np.array([[ 0., 0.],
                      [ 1., 0.],
                      [ 3., 0.],
                      [ 8., 0.],
                      [ 8.9, 0.]])
        mst = np.array([[3, 4], [0, 1], [1, 2], [2, 3]])
        sort_edges(P, mst)
        index, nodes, max_stack = tree_structure(mst)
        
        
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
        G = spanner(P, d_min=1)

    def test_spanner_lipschitz(self):
        P = np.array([[ 0., 1.,   5., 2.],
                      [ 6., 2.,  20., 3.],
                      [ 7., 3.,  45., 4.],
                      [ 8., 4.,  80., 5.],
                      [ 9., 5., 125., 6.]])
        G = spanner(P, d_min=1, lsh='lipschitz')
       
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

    def test_distortion_with_mst(self):
        # 1D
        P = np.array(
            [[10.], [5.], [0.], [8.], [1.], [7.]])
        tree = [[ 3., 4., 10., 2.],
                [ 0., 1., 20., 2.],
                [ 2., 6., 20., 3.],
                [ 8., 5., 40., 4.],
                [ 7., 9., 30., 6.]]
        mst = exact_mst(P)
        d = distortion(P, tree)
        d2 = distortion_with_mst(P, mst, tree)
        #assert(d >= d2)

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

    def test_gamma(self):
        P = np.array([[ 0., 1.,   5., 2.],
                      [ 6., 2.,  20., 3.],
                      [ 7., 3.,  45., 4.],
                      [ 8., 4.,  80., 5.],
                      [ 2., 1.,   5., 2.],
                      [ 4., 2.,  20., 3.],
                      [ 6., 3.,  45., 4.],
                      [ 7., 4.,  80., 5.],
                      [ 1., 5., 125., 6.]])
        mst = exact_mst(P)
        Mst = exact_mst(P, maximal=True)
        other_st = np.array( [ [i, i+1] for i in range(len(P)-1) ])
        self.assertEqual(compute_gamma(P, mst, mst), 1.)
        compute_gamma(P, mst, other_st)

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
