from fast_ultrametrics import *
from fast_ultrametrics.distortion import *
from scipy.cluster.hierarchy import linkage

import matplotlib.pyplot as plt
import random
import time
import scipy
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
                
class Algo:
    def __init__(self, deterministic=False, rescale=False):
        self.data = []
        self.deterministic = deterministic
        self.rescale = rescale
    def test(self, X, mst, N=3):
        if self.deterministic:
            N=1
        for i in range(1):
            dist = 0
            chrono = 0
            print(self.name)
            for i in range(N):
                print("{}/{}".format(i+1, N))
                chrono -= time.perf_counter()
                result = self.run(X)
                chrono += time.perf_counter()
                if self.rescale:
                    dist += fast_distortion_with_mst(X, mst, result, nsample=100000)
                else:
                    dist += distortion_with_mst(X, mst, result)
            self.data.append(((), dist/N, chrono/N))

common_params = [1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8]
p_default = 1.8

class AlgoMST(Algo):
    def __init__(self):
        Algo.__init__(self, deterministic=True)
        self.name = 'prim + normal cutweights'
        
    def run(self, X):
        return ultrametric(X, lsh='exact')
    
class AlgoMSTBB(Algo):
    def __init__(self):
        Algo.__init__(self, deterministic=True)
        self.name = 'prim + bounding balls'
        
    def run(self, X):
        return ultrametric(X, lsh='exact', cut_weights='bounding balls')
    
class AlgoCKL(Algo):
    def __init__(self, p=p_default):
        Algo.__init__(self, rescale=True)
        self.name = 'Cohen-Addad, Karthik, Lagarde (p={})'.format(p)
        self.p = p
        
    def run(self, X):
        return ultrametric(X, scale_factor=self.p, cut_weights='5-approx')
    
class Algo3(Algo):
    def __init__(self, p=p_default):
        Algo.__init__(self)
        self.name = '3-approx (p={})'.format(p)
        self.p = p
        
    def run(self, X):
        return ultrametric(X, scale_factor=self.p, cut_weights='approximate')
    
class AlgoNew(Algo):
    def __init__(self, p=p_default):
        Algo.__init__(self)
        self.name = 'bounding balls (p={})'.format(p)
        self.p = p
        
    def run(self, X):
        return ultrametric(X, scale_factor=self.p, cut_weights='bounding balls')
    
class AlgoExact(Algo):
    def __init__(self):
        Algo.__init__(self, deterministic=True)
        self.name = 'Farach'
        
    def run(self, X):
        return ultrametric(X, lsh='exact', cut_weights='exact')

class AlgoSKL(Algo):
    def __init__(self, algo):
        Algo.__init__(self, rescale=True, deterministic=True)
        self.name = "scikitlrearn: {}".format(algo)
        self.algo = algo
        
    def run(self, X):
        return linkage(X, self.algo)

def compare(name):
    file_name = "datasets/"+name+".csv"
    X = np.genfromtxt(file_name, delimiter=",")
    mst = np.load(mst_file(file_name))
    print(X.shape)

    for algo in [
            AlgoCKL(),
            AlgoExact(),
            AlgoMSTBB(),
#            Algo3(),
#            AlgoMST(),
            AlgoNew(),
#            AlgoSKL('ward'),
#            AlgoSKL('single'),
#            AlgoSKL('average'),
#            AlgoSKL('complete'),
    ]:
        algo.test(X, mst)
        plt.plot([t for (_, _, t) in algo.data], [d for (_, d, _) in algo.data],
                 label=algo.name,
                 marker='o',
        )
    plt.legend()
    plt.show()
            
if __name__ == '__main__':
    import timeit

    
#    compare("SHUTTLE")
    compare("MICE")
#    compare("IRIS")
#    compare("DIABETES")
#    compare("PENDIGITS")
#
