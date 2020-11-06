from ultrametric import *
from distortion import *

import matplotlib.pyplot as plt

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

def lib_all_together(points, degree):
    N = points.shape[0]
    MST = lib_mst(points, degree=degree)
    CW = cut_weight(points,MST)
    return single_linkage_label(MST,CW)

class Algo:
    def __init__(self, N = 20):
        self.data = []
        self.N = N # Number of iterations for each parameter
    def test(self, X):
        for p in self.params:
            dist = 0
            chrono = 0
            for i in range(self.N):
                print("{}/{}".format(i, self.N))
                chrono -= time.perf_counter()
                result = self.run(p, X)
                chrono += time.perf_counter()
                dist += distortion(X, result)
            self.data.append((p, dist/self.N, chrono/self.N))

class AlgoBall(Algo):
    def __init__(self):
        Algo.__init__(self)
        self.name = 'ball'
        self.params = [1.05, 1.1, 1.2, 1.3, 1.5, 2.]
        
    def run(self, p, X):
        return ultrametric(X, scale_factor=p, lsh='balls')

class AlgoLip(Algo):
    def __init__(self):
        Algo.__init__(self)
        self.name = 'lipschitz'
        self.params = [1.05, 1.1, 1.2, 1.3, 1.5, 2.]
        
    def run(self, p, X):
        return ultrametric(X, scale_factor=p, lsh='lipschitz')

class AlgoLibrary(Algo):
    def __init__(self):
        Algo.__init__(self)
        self.name = 'scikit-learn nearest neighbors'
        self.params = [20, 50, 100, 200] # degree
        
    def run(self, p, X):
        return lib_all_together(X, p)
    
def compare(name):
    file_name = "datasets/"+name+".csv"
    X = np.genfromtxt(file_name, delimiter=",")

    for algo in [AlgoBall(), AlgoLip(), AlgoLibrary()]:
        algo.test(X)

        plt.plot([t for (_, _, t) in algo.data], [d for (_, d, _) in algo.data], label=algo.name)
    plt.legend()
    plt.show()
            
if __name__ == '__main__':
    import timeit

    #compare("PENDIGITS")
    compare("DIABETES")