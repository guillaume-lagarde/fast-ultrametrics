from fast_ultrametrics import *
from fast_ultrametrics.distortion import *

import matplotlib.pyplot as plt
import random
import time
import scipy
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
                
class Algo:
    def __init__(self, N = 5):
        self.data = []
        self.N = N # Number of iterations for each parameter
    def test(self, X, mMst: (np.array, np.array), best_dist):
        for p in self.params:
            dist = 0
            gamma = 0
            chrono = [0., 0.]
            print(self.name)
            for i in range(self.N):
                print("{}/{}".format(i, self.N))
                # Spanning tree
                chrono[0] -= time.perf_counter()
                approx_mst = self.mst(p, X)
                chrono[0] += time.perf_counter()
                gamma += compute_gamma(X, mMst[0], approx_mst)
                # ultrametric
                chrono[1] -= time.perf_counter()
                ultrametric = self.ultrametric(X, approx_mst)
                chrono[1] += time.perf_counter()
                dist += distortion_with_mst(X, mMst, ultrametric)
            gamma /= self.N
            dist /= self.N
            for i in range(2):
                chrono[i] /= self.N
            self.data.append((gamma, dist/best_dist))

#common_params = [1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8]
common_params = [ 1. + i * 0.05 for i in range(6, 20)]
            
class AlgoBB(Algo):
    def __init__(self, algo='bounding balls'):
        Algo.__init__(self)
        self.name = algo
        self.params = common_params
        
    def mst(self, p, X):
        edges = spanner(X, scale_factor=p, d_min=0.001, lsh='lipschitz')    
        return mst(X, edges)

    def ultrametric(self, X, MST):
        CW = cut_weight_bounding_ball(X, MST)
        #CW = cut_weight(X, MST)
        return single_linkage_label(MST, CW)

def compare(name):
    file_name = "datasets/"+name+".csv"
    X = np.genfromtxt(file_name, delimiter=",")
    mMst = np.load(mst_file(file_name))
    print(X.shape)

    best_ultrametric = single_linkage_label(mMst[0],
                                            exact_cut_weight(X, mMst[0]))
    exact_dist = distortion_with_mst(X, mMst, best_ultrametric)
    for algo in [
            AlgoBB(),
    ]:
        algo.test(X, mMst, exact_dist)

        plt.plot([x for (x, _) in algo.data], [y for (_, y) in algo.data],
                 label=algo.name,
                 marker='o',
        )
    plt.legend()
    plt.show()
            
if __name__ == '__main__':
    import timeit

    
#    compare("SHUTTLE")
#    compare("MICE")
#    compare("IRIS")
    compare("DIABETES")
#    compare("PENDIGITS")
#
