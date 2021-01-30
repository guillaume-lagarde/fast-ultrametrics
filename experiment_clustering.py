from fast_ultrametrics import *
from fast_ultrametrics.distortion import *
from fast_ultrametrics.utils import *

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import time
import scipy
import fastcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.random_projection import GaussianRandomProjection
from scipy.optimize import linear_sum_assignment
import numpy as np

import sklearn.datasets as datasets

def cost(common, N, N_real):
    return -common

def preprocess(X):
    X = StandardScaler().fit_transform(X)
    return X

def dim_reduction(X):
    X = GaussianRandomProjection(eps=0.3).fit_transform(X)
 #   print(X.shape)
    return X
    
class Stat:
    def __init__(self, Y, real_Y, n_class):
        self.count = (np.zeros(n_class+1, dtype=int), np.zeros(n_class, dtype=int)) # Y, realY
        self.cor = np.zeros((n_class+1, n_class), dtype=int)
        self.N = len(real_Y)
        self.n_class = n_class
        #
        for i in range(self.N):
            self.count[0][Y[i]] += 1
            self.count[1][real_Y[i]] += 1
            self.cor[Y[i]][real_Y[i]] += 1
        # compute best assignement
        cost_matrix = np.zeros((n_class, n_class))
        for i in range(n_class):
            for j in range(n_class):
                cost_matrix[i][j] = cost(self.cor[i][j], self.count[0][i], self.count[1][j])
        print(self.cor)
        self.matching = linear_sum_assignment(cost_matrix)
        # evaluate score
        matched = 0
        score = 0.
        for i in range(n_class):
            u = self.matching[0][i]
            v = self.matching[1][i]
            matched += self.cor[u][v]
            score += cost_matrix[u][v]
        missed = self.N -matched
        print(self.count[0][n_class], " outliers")
        print(matched, missed, (matched * 100.)/self.N,"%")
#        print("score:", score)

p_default = 1.1

class ClusterTest:
    def __init__(self, n_clusters, homogeneous=False):
        self.n_clusters = n_clusters
        self.homogeneous = homogeneous
        
    def test(self):
        X, Y = self.load()
        if not self.homogeneous:
            X = preprocess(X)
        else:
            X = np.array(X)
        #
        result = ultrametric(X, scale_factor=p_default, lsh='lipschitz', cut_weights='bounding balls')
        clustering = clusters(result, n_clusters=self.n_clusters, min_size=10)
        s = Stat(clustering, Y, self.n_clusters)
        
class Iris(ClusterTest):
    def __init__(self):
        super().__init__(n_clusters=3)

    def load(ClusterTest):
        iris = datasets.load_iris()
        return (iris.data, iris.target)

class Wine(ClusterTest):
    def __init__(self):
        super().__init__(n_clusters=3)

    def load(ClusterTest):
        data = datasets.load_wine()
        return (data.data, data.target)

class BCW(ClusterTest):
    def __init__(self):
        super().__init__(n_clusters=2)

    def load(ClusterTest):
        data = datasets.load_breast_cancer()
        return (data.data, data.target)

class Newsgroup(ClusterTest):
    def __init__(self):
        super().__init__(n_clusters=20)

    def load(ClusterTest):
        data = datasets.fetch_20newsgroups_vectorized()
        return (dim_reduction(data.data), data.target)

class Digits(ClusterTest):
    def __init__(self):
        super().__init__(n_clusters=10, homogeneous=True)

    def load(ClusterTest):
        data = datasets.load_digits()
        return (data.data, data.target)

Iris().test()
Wine().test()
BCW().test()
#Newsgroup().test()
Digits().test()
