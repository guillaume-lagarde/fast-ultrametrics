from fast_ultrametrics import *
from fast_ultrametrics.distortion import *

import matplotlib.pyplot as plt
import random
import time
import scipy
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
                
class Algo:
    def __init__(self, N = 1):
        self.data = []
        self.N = N # Number of iterations for each parameter
    def test(self, X, mMst):
        for p in self.params:
            dist = 0
            chrono = 0
            print(self.name)
            for i in range(self.N):
                print("{}/{}".format(i, self.N))
                chrono -= time.perf_counter()
                result = self.run(p, X)
                chrono += time.perf_counter()
                dist += distortion_with_mst(X, mMst, result)
            self.data.append((p, dist/self.N, chrono/self.N))

common_params = [1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8]
            
class AlgoBall(Algo):
    def __init__(self):
        Algo.__init__(self)
        self.name = 'balls'
        self.params = common_params
        
    def run(self, p, X):
        return ultrametric(X, scale_factor=p, lsh='balls')

class AlgoLip(Algo):
    def __init__(self):
        Algo.__init__(self)
        self.name = 'lipschitz'
        self.params = common_params
        
    def run(self, p, X):
        return ultrametric(X, scale_factor=p, lsh='lipschitz')

class AlgoExp(Algo):
    def __init__(self):
        Algo.__init__(self)
        self.name = 'experimental'
        self.params = common_params
        
    def run(self, p, X):
        return ultrametric(X, scale_factor=p, lsh='experimental')

class AlgoMST(Algo):
    def __init__(self, N=1):
        Algo.__init__(self)
        self.name = 'prim + normal cutweights'
        self.params = [()]
        
    def run(self, p, X):
        return ultrametric(X, lsh='exact')
    
class AlgoMSTBoundingballs(Algo):
    def __init__(self, N=1):
        Algo.__init__(self, N=1)
        self.name = 'prim + bounding balls'
        self.params = [()]
        
    def run(self, p, X):
        return ultrametric(X, lsh='exact', cut_weights='bounding balls')
    
class AlgoExact(Algo):
    def __init__(self):
        Algo.__init__(self, N=1)
        self.name = 'Farach et al.'
        self.params = [()]
        
    def run(self, p, X):
        return ultrametric(X, lsh='exact', cut_weights='exact')

class AlgoBoundingBall(Algo):
    def __init__(self):
        Algo.__init__(self)
        self.name = 'bounding balls'
        self.params = common_params
        
    def run(self, p, X):
        return ultrametric(X, scale_factor=p,lsh='lipschitz',
                           cut_weights='bounding balls')


def compare(name):
    file_name = "datasets/"+name+".csv"
    X = np.genfromtxt(file_name, delimiter=",")
    mMst = np.load(mst_file(file_name))
    print(X.shape)

    for algo in [
#            AlgoBall(),
#            AlgoExp(),
            AlgoLip(),
            AlgoExact(),
            AlgoMST(),
            AlgoBoundingBall(),
            AlgoMSTBoundingballs(),
    ]:
        algo.test(X, mMst)

        plt.plot([t for (_, _, t) in algo.data], [d for (_, d, _) in algo.data],
                 label=algo.name,
                 marker='o',
        )
    plt.legend()
    plt.show()
            
if __name__ == '__main__':
    import timeit

    
#    compare("SHUTTLE")
#    compare("MICE")
    compare("IRIS")
#    compare("DIABETES")
#    compare("PENDIGITS")
#
