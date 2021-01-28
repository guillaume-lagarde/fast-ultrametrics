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
    def test(self, X, mst, N=20):
        if self.deterministic:
            N=1
        for i in range(1):
            dist = 0
            chrono = 0
            for i in range(N):
                print("{}/{}".format(i+1, N), flush=True, end=" ")
                if i == N-1: print()
                chrono -= time.perf_counter()
                result = self.run(X)
                chrono += time.perf_counter()
                #
                if self.rescale:
                    dist += distortion_(X, result)
                else:
                    dist += distortion_with_mst(X, mst, result)
            self.data.append((chrono/N, dist/N))
        #print(self.data)
        print('"{}" {} {}'.format(self.name, self.data[0][1], self.data[0][0]))
        

common_params = [1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8]
p_default = 1.2

class AlgoMST(Algo):
    def __init__(self):
        Algo.__init__(self, deterministic=True)
        self.name = 'prim + normal cutweights'
        
    def run(self, X):
        return ultrametric(X, lsh='exact')
    
class AlgoMSTCKL(Algo):
    def __init__(self):
        Algo.__init__(self, deterministic=True, rescale=True)
        self.name = 'MST + Cohen-Addad, Karthik, Lagarde'
        
    def run(self, X):
        return ultrametric(X, lsh='exact', cut_weights='5-approx')
    
class AlgoMSTBB(Algo):
    def __init__(self):
        Algo.__init__(self, deterministic=True)
        self.name = 'prim + bounding balls'
        
    def run(self, X):
        return ultrametric(X, lsh='exact', cut_weights='bounding balls')
    
class AlgoCKL(Algo):
    def __init__(self, p=p_default, rescale=True):
        Algo.__init__(self)
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
        Algo.__init__(self, deterministic=True, rescale=True)
        self.name = "scikitlrearn: {}".format(algo)
        self.algo = algo
        
    def run(self, X):
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage=self.algo)
        clusters = model.fit(X)

        counts = np.zeros(clusters.children_.shape[0])
        n_samples = len(clusters.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([clusters.children_, clusters.distances_,
                                          counts]).astype(float)
        return linkage_matrix

class AlgoFastCluster(Algo):
    def __init__(self, algo):
        Algo.__init__(self, deterministic=True, rescale=True)
        self.name = "fastcluster: {}".format(algo)
        self.algo = algo
    
    def run(self, X):
        return fastcluster.linkage_vector(X, method=self.algo)
    
def compare(name):
    file_name = "datasets/"+name+".csv"
    X = np.genfromtxt(file_name, delimiter=",")
    mst = np.load(mst_file(file_name))
    print(X.shape)

    for algo in [
            AlgoNew(p=1.2),
            AlgoCKL(p=1.2),
#            AlgoExact(),
#            AlgoMSTBB(),
            AlgoMSTCKL(),
#            Algo3(p=1.2),
#            AlgoMST(),
#            AlgoNew(),
#            AlgoSKL('ward'),
#            AlgoSKL('single'),
#            AlgoSKL('average'),
#            AlgoSKL('complete'),
            AlgoFastCluster('ward'),
            AlgoFastCluster('single'),
            AlgoFastCluster('centroid'),
            AlgoFastCluster('median'),
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

    compare("blobsN10000d100")
#    compare("SHUTTLE")
#    compare("MICE")
#    compare("IRIS")
#    compare("DIABETES")
#    compare("PENDIGITS")
##
