from union_find import *
from distortion import *

import time
import scipy
from scipy.sparse.csgraph import minimum_spanning_tree
import sklearn
import numpy as np
from sklearn.neighbors import kneighbors_graph

def lib_mst(P, degree=50):
    Graph = kneighbors_graph(P,
                             metric='euclidean',
                             n_neighbors=min(degree, len(P)-1),
                             mode='connectivity')
    print(Graph.nnz)
    result = scipy.sparse.coo_matrix(minimum_spanning_tree(Graph))
    return np.array([[u, v] for u, v in zip(result.row, result.col)], dtype=np.intp)

def lib_all_together(points):
    N = points.shape[0]
    MST = lib_mst(points)
    CW = cut_weight(points,MST)
    return single_linkage_label(MST,CW)

class Algo:
    def __init__(self):
        self.data = []
    
    def test(self, X):
        for p in self.params:
            tic = time.perf_counter()
            result = self.run(p, X)
            tac = time.perf_counter()
            dist = distortion(X, result)
            self.data.append((p, dist, tac-tic))

class AlgoBall(Algo):
    def __init__(self):
        Algo.__init__(self)
        self.params = [1.5, 2., 3.]
        
    def run(self, p, X):
        return all_together(X, p, algorithm='balls')

class AlgoLip(Algo):
    def __init__(self):
        Algo.__init__(self)
        self.params = [1.5, 2., 3.]
        
    def run(self, p, X):
        return all_together(X, p, algorithm='lipschitz')

def compare(name):
    file_name = "datasets/"+name+".csv"
    X = np.genfromtxt(file_name, delimiter=",")

    for algo in [AlgoBall(), AlgoLip()]:
        algo.test(X)


if __name__ == '__main__':
    import timeit

    #compare("PENDIGITS")
    compare("MICE")
    
    #print(timeit.timeit("test(\"MICE\")",
     #                   setup="from __main__ import test",
      #                  number=4))
