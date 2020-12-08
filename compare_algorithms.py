from fast_ultrametrics import *
from fast_ultrametrics.distortion import *

import matplotlib.pyplot as plt
import random
import time
import scipy
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np

def dist(a, b):
    res = 0.
    assert(len(a) == len(b))
    for i in range(len(a)):
        res += (a[i] - b[i])**2
    return res
                
class Algo:
    def __init__(self, N = 5):
        self.data = []
        self.N = N # Number of iterations for each parameter
    def test(self, X):
        for p in self.params:
            dist = 0
            chrono = 0
            print(self.name)
            for i in range(self.N):
                print("{}/{}".format(i, self.N))
                chrono -= time.perf_counter()
                result = self.run(p, X)
                chrono += time.perf_counter()
                dist += distortion(X, result)
            self.data.append((p, dist/self.N, chrono/self.N))

common_params = [1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2, 2.1]
            
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
    def __init__(self):
        Algo.__init__(self)
        self.name = 'prim'
        self.params = [(),()]
        
    def run(self, p, X):
        return ultrametric(X, lsh='exact')
    
class AlgoExact(Algo):
    def __init__(self):
        Algo.__init__(self)
        self.name = 'prim + exact cutweight'
        self.params = [(),()]
        
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
    print(X.shape)

    for algo in [
#            AlgoBall(),
#            AlgoExp(),
            AlgoLip(),
#            AlgoExact(),
#            AlgoMST(),
            AlgoBoundingBall(),
    ]:
        algo.test(X)

        plt.plot([t for (_, _, t) in algo.data], [d for (_, d, _) in algo.data], label=algo.name)
    plt.legend()
    plt.show()
            
if __name__ == '__main__':
    import timeit

    
#    compare("SHUTTLE")
#compare("MICE")
    compare("IRIS")
   # compare("DIABETES")
    #compare("PENDIGITS")
#
