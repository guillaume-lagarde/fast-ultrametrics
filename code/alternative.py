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


def compare(name):
    file_name = "datasets/"+name+".csv"
    X = np.genfromtxt(file_name, delimiter=",")
    print("a")
    tic = time.perf_counter()
    res1 = all_together(X, 2., d_min=1)
    toc = time.perf_counter()
    print("b ({})".format(toc-tic))
    tic = time.perf_counter()
    res2 = lib_all_together(X)
    toc = time.perf_counter()
    print("c ({})".format(toc-tic))
    d1=distortion(X, res1)
    d2=distortion(X, res2)
    print("dist old:{}, dist lib:{}".format(d1, d2))


if __name__ == '__main__':
    import timeit

    compare("PENDIGITS")
    
    #print(timeit.timeit("test(\"MICE\")",
     #                   setup="from __main__ import test",
      #                  number=4))
