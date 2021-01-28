from fast_ultrametrics import *
from fast_ultrametrics.distortion import *

import matplotlib.pyplot as plt
import random
import time
import scipy
import fastcluster
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
                
class Algo:
    def __init__(self, deterministic=False, rescale=False):
        self.data = []
        self.deterministic = deterministic
        self.rescale = rescale
    def test(self, X, mst, N=10):
        if self.deterministic:
            N=1
        for i in range(1):
            dist = 0.
            chrono = 0.
            for i in range(N):
                chrono -= time.perf_counter()
                result = self.run(X, mst)
                chrono += time.perf_counter()
                #
                if self.rescale:
                    dist += distortion_(X, result)
                else:
                    dist += distortion_with_mst(X, mst, result)
            self.data.append((chrono/N, dist/N))
        #print(self.data)
        print('"{}" {}'.format(self.name, self.data[0][1]))
        

common_params = [1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8]
p_default = 1.2

class AlgoMST(Algo):
    def __init__(self, eps=0.2):
        Algo.__init__(self, deterministic=True)
        self.eps = eps
        self.name = 'bounding balls (eps={})'.format(eps)
        
    def run(self, X, mst):
        CW = cut_weight_bounding_ball(X, mst, eps=self.eps)
        return single_linkage_label(mst, CW)
    
class AlgoCKL(Algo):
    def __init__(self, rescale=True):
        Algo.__init__(self, deterministic=True, rescale=rescale)
        self.name = 'Cohen-Addad, Karthik, Lagarde (rescale={})'.format(rescale)
        
    def run(self, X, mst):
        CW = old_cut_weight(X, mst)
        return single_linkage_label(mst, CW)
    
class AlgoExact(Algo):
    def __init__(self):
        Algo.__init__(self, deterministic=True, rescale=False)
        self.name = 'Farach'
        
    def run(self, X, mst):
        CW = exact_cut_weight(X, mst)
        return single_linkage_label(mst, CW)
    
def compare(name):
    file_name = "datasets/"+name+".csv"
    X = np.genfromtxt(file_name, delimiter=",")
    mst = np.load(mst_file(file_name))
    print(X.shape)

    for algo in [
            AlgoCKL(rescale=False),
            AlgoCKL(),
            AlgoMST(eps=0.0001),
            AlgoMST(eps=0.2),
            AlgoMST(eps=1.),
            AlgoExact(),
    ]:
        algo.test(X, mst)
        plt.plot([x for (x, _) in algo.data], [y for (_, y) in algo.data],
                 label=algo.name,
                 marker='o',
        )
    plt.legend()
    plt.show()
            
if __name__ == '__main__':
    import timeit

#    compare("blobsN10000d100")
#    compare("SHUTTLE")
#    compare("MICE")
    compare("IRIS")
#    compare("DIABETES")
#    compare("PENDIGITS")
##
