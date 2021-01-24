
import math, random, sys
from itertools import chain
import numpy as np
from pathlib import Path
from fast_ultrametrics import *

# input: array left cluster, right cluster, delta, size subtree
def infix_order(tree):
    stack = [(-1,-1, -1)]
    res = []
    n = len(tree)+1
    current = 2*n-2
    depth = 0
    while current != -1:
        while current >= n:
            stack.append((current, depth, int(tree[current-n][1])))
            current = int(tree[current-n][0])
            depth+=1

        res.append((current, depth)) # append leaf

        parent, depth, current  = stack.pop()
        if parent != -1:
            res.append((parent,depth))
        depth+=1
            
    return res

class RMQ:
    def __init__(self, tree):
        order = infix_order(tree)
        self.tree = tree
        self.n = len(tree)+1
        self.depth = [x for (_,x) in order]
        self.indices = [x for (x,_) in order]
        self.positions = [0]*len(order)
        for t, i in enumerate(self.indices):
            self.positions[i] = t

        # sparse table for efficient search
        self.N = len(order)
        self.K = math.floor(math.log2(self.N)) # max exponent
        self.st = [[i]*(self.K+1) for i in range(self.N)] # st[i][j] = index of min between [i, i + 2**j-1]

        # filling the sparse table
        for j in range(1,self.K+1):
            for i in range(0, self.N):
                if i + (1 <<(j-1) ) >= self.N: break
                i1, i2 = self.st[i][j-1], self.st[i+(1<<(j-1))][j-1]
                if self.depth[i1] < self.depth[i2]:
                    self.st[i][j] = i1
                else:
                    self.st[i][j] = i2
                    
    def search(self, u, v):
        iu, iv = self.positions[u], self.positions[v]
        if iu > iv:
            iu, iv = iv, iu

        j = math.floor(math.log2(iv-iu+1))
        i1, i2 = self.st[iu][j], self.st[iv-(1<<j)+1][j]
        if self.depth[i1] < self.depth[i2]:
            return self.indices[i1]
        else:
            return self.indices[i2]

    def dist(self, u, v):
        if u!= v:
            return self.tree[self.search(u, v)- self.n][2]
        return 0

def dist(a, b):
    assert(len(a) == len(b))
    return math.sqrt(sum( (x-y)**2 for x, y in zip(a, b) ))
    
def fast_distortion(data, tree, nsample=10000):
    rmq = RMQ(tree)
    n = len(data)

    MAX, MIN = 0.1, 10000.
    for _ in range(nsample):
        i = random.randrange(n)
        j = random.randrange(n)
        if i != j:
            l2 = dist(data[i], data[j])
            ratio = rmq.dist(i, j) / l2
            MAX = max(ratio, MAX)
            MIN = min(ratio, MIN)
    return MAX/MIN
    
def average_distortion(data, tree, nsample=10000):
    rmq = RMQ(tree)
    n = len(data)
    S = 0
    N = 0
    MAX, MIN = 1., 1000.
    for _ in range(nsample):
        i = random.randrange(n)
        j = random.randrange(n)
        if i != j:
            l2 = dist(data[i], data[j])
            if l2 !=0:
                ratio = rmq.dist(i, j) / l2
                S+=ratio
                N+=1
                MAX = max(ratio, MAX)
                MIN = min(ratio, MIN)
    return (S/N)*(1/MIN)

def distortion(data, tree):
    rmq = RMQ(tree)
    n = len(data)
    MAX, MIN = 0., sys.float_info.max
    for i in range(n):
        for j in range(i):
            l2 = dist(data[i], data[j])
            if l2 > 0.:
                ratio = rmq.dist(i, j) / l2
                MAX = max(ratio, MAX)
                MIN = min(ratio, MIN)
    assert(rmq.dist(i, j) >= l2)
    return MAX/MIN

def mst_file(dataset_name: str):
    MST_DIR = "mst/"
    return MST_DIR + Path(dataset_name).stem + ".mst.npy"

def distortion_with_mst(data, mMst: (np.array, np.array), tree):
    rmq = RMQ(tree)
    mst, Mst = mMst
    n = len(data)
    assert(mst.shape == (n-1, 2))
    assert(Mst.shape == (n-1, 2))
    MAX, MIN = 0., sys.float_info.max
    for i, j in chain(mst, Mst):
        l2 = dist(data[i], data[j])
        if l2 > 0.:
            ratio = rmq.dist(i, j) / l2
            MAX = max(ratio, MAX)
            MIN = min(ratio, MIN)
#    assert(MAX/MIN == distortion(data, tree))
    return MAX/MIN

def fast_distortion(data, tree, nsample=10000):
    rmq = RMQ(tree)
    n = len(data)

    MAX, MIN = 0.1, 10000.
    for _ in range(nsample):
        i = random.randrange(n)
        j = random.randrange(n)
        if i != j:
            l2 = dist(data[i], data[j])
            if l2 > 0.:
                ratio = rmq.dist(i, j) / l2
                MAX = max(ratio, MAX)
                MIN = min(ratio, MIN)
    return MAX/MIN

def weight(data, tree: np.ndarray) -> float:
    n = data.shape[0]
    assert(tree.shape == (n-1, 2))
    res = 0.
    for i, j in tree:
        res += dist(data[i], data[j])
    return res

def compute_gamma(data, mst: np.ndarray, spanning_tree: np.ndarray):
    assert(weight(data, mst) <= weight(data, spanning_tree) + 0.000001)
    
    # Compute the single linkage
    weights = np.array([dist(data[i], data[j]) for i, j in spanning_tree])
    tree = single_linkage_label(spanning_tree, weights)
    
    rmq = RMQ(tree)
    n = len(data)
    MAX = 0.,
    for i, j in mst:
        l2 = dist(data[i], data[j])
        assert(rmq.dist(i, j) >= l2)
        ratio = rmq.dist(i, j) / l2
        MAX = max(ratio, MAX)
    return MAX
