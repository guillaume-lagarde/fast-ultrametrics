# cython: profile=True
# cython: linetrace=True

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, round
import random


# Numpy must be initialized.
np.import_array()

# from ..neighbors._dist_metrics cimport DistanceMetric

# C++
#from cython.operator cimport dereference as deref, preincrement as inc
#from libcpp.map cimport map as cpp_map

# C types as in scikit-learn
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ITYPE = np.intp
ctypedef np.intp_t ITYPE_t

cdef class UnionFind():

    cdef ITYPE_t[:] parent
    cdef ITYPE_t[:] size
    cdef ITYPE_t[:] succ
    
    def __init__(self, N):
        self.parent = np.full(N, -1., dtype=ITYPE, order='C')
        self.size = np.ones(N,dtype=ITYPE)
        self.succ = np.array(range(N),dtype=ITYPE)
        
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef void union(self, ITYPE_t m, ITYPE_t n):
        cdef ITYPE_t r_m = self.find(m)
        cdef ITYPE_t r_n = self.find(n)
       
        if r_m == r_n: return

        if self.size[r_m] >= self.size[r_n]:
            r_n, r_m = r_m, r_n
        self.parent[r_m] = r_n
        self.size[r_n] += self.size[r_m]
        self.succ[r_m], self.succ[r_n] = self.succ[r_n], self.succ[r_m]
            
        return

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef ITYPE_t find(self, ITYPE_t n):
        cdef ITYPE_t p = n
        
        # find the root
        while self.parent[n] != -1:
            n = self.parent[n]
            
        # path compression
        if n!=p:
            while self.parent[p] != n:
                p, self.parent[p] = self.parent[p], n
        return n


# Distance
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef DTYPE_t dist(DTYPE_t[::1] x, DTYPE_t[::1] y, ITYPE_t dim):
    cdef DTYPE_t res = 0
    cdef ITYPE_t index
    for index in range(dim):
         res += (x[index] - y[index])**2
    return sqrt(res)


#------------------------------------
# minimum spanning tree
cpdef mst(DTYPE_t[:, ::1] points, graph):
    cdef ITYPE_t N = points.shape[0]
    cdef ITYPE_t dim = points.shape[1]
    cdef ITYPE_t m = len(graph)
    cdef ITYPE_t count, x, y, i

    cdef ITYPE_t[:, ::1] edges = np.zeros(shape=(m,2), dtype=ITYPE)
    weight = np.zeros(shape=m, dtype=DTYPE)
    cdef DTYPE_t[::1] weight_view = weight 
    i = 0
    for x, y in graph:
        edges[i][0] = x
        edges[i][1] = y
        weight_view[i] = dist(points[x], points[y], dim)
        i += 1
    cdef ITYPE_t[::1] order = np.argsort(weight)
    
    mst = np.zeros(shape=(N-1, 2), dtype=ITYPE)
#   edges = sorted(edges, key = lambda e: dist(points[e[0]], points[e[1]], dim))
    U = UnionFind(N)
    count = 0
    for i in order:
        x = edges[i][0]
        y = edges[i][1]
        if U.find(x) != U.find(y):
            mst[count][0] = x
            mst[count][1] = y
            U.union(x,y)
            count+=1
            if count == N-1:
                break
    assert(count == N-1)
    return mst
   
#--------------------------------------
# Cut weight

#cdef np.ndarray[DTYPE_t] cut_weight(np.ndarray[DTYPE_t, ndim=2] points, mst):
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def cut_weight(DTYPE_t[:, ::1] points, ITYPE_t[:, ::1] mst):
    cdef ITYPE_t N = points.shape[0]
    cdef ITYPE_t dim = points.shape[1]

    cdef DTYPE_t[:] radius = np.zeros(N, dtype=DTYPE)
    cdef ITYPE_t c0, c1, v
    cdef DTYPE_t d

    sets = UnionFind(N)
    result = np.zeros(N - 1, dtype=DTYPE)
    cdef DTYPE_t[:] result_c = result
    
    # Can be remove if the mst is sorted
    mst_dist = np.zeros(N-1)
    cdef DTYPE_t[:] mst_dist_c = mst_dist
    for i in range(N-1):
        mst_dist[i] = dist(points[mst[i][0]], points[mst[i][1]], dim)
    order = mst_dist.argsort()

    # 
    for i in range(N-1):
        c0 = sets.find(mst[order[i]][0])
        c1 = sets.find(mst[order[i]][1])
        if sets.size[c0] < sets.size[c1]:
            c0, c1 = c1, c0

        # c0 is the new root of c1
        d = dist(points[c0], points[c1], dim)
        # Expression of the 5-approximation of the cut-weight
        result_c[order[i]] = 5. * max(d, radius[c0] - d, radius[c1] - d)          
          
        # Update radius of the new component with center c0
        v = c1
        while True:
            d = dist(points[c0], points[v], dim)
            if d > radius[c0]: radius[c0] = d
            v = sets.succ[v]
            if v == c1: break
            
        sets.union(c0, c1)
    return result

   
#---------------------------------------
# Computing ultrametric from cut weights

# we need a tweaked UnionFind
cdef class UnionFindUltrametric(object):
    cdef ITYPE_t next_label
    cdef ITYPE_t[:] parent
    cdef ITYPE_t[:] size

    def __init__(self, N):
        self.parent = np.full(2 * N - 1, -1., dtype=ITYPE, order='C')
        self.next_label = N
        self.size = np.hstack((np.ones(N, dtype=ITYPE),
                               np.zeros(N - 1, dtype=ITYPE)))

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef void union(self, ITYPE_t m, ITYPE_t n):
        self.parent[m] = self.next_label
        self.parent[n] = self.next_label
        self.size[self.next_label] = self.size[m] + self.size[n]
        self.next_label += 1

        return

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    cdef ITYPE_t find(self, ITYPE_t n):
        cdef ITYPE_t p
        p = n
        # find the highest node in the linkage graph so far
        while self.parent[n] != -1:
            n = self.parent[n]
        # provide a shortcut up to the highest node
        while self.parent[p] != n:
            p, self.parent[p] = self.parent[p], n
        return n

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] _single_linkage_label(DTYPE_t[:, ::1] L):
    cdef np.ndarray[DTYPE_t, ndim=2] result

    cdef ITYPE_t left, left_cluster, right, right_cluster, index
    cdef DTYPE_t delta

    result = np.zeros((L.shape[0], 4), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] result_c = result

    cdef UnionFindUltrametric U = UnionFindUltrametric(L.shape[0] + 1)

    for index in range(L.shape[0]):

        left = <ITYPE_t> L[index, 0]
        right = <ITYPE_t> L[index, 1]
        delta = L[index, 2]

        left_cluster = U.find(left)
        right_cluster =U.find(right)
        
        result_c[index][0] = left_cluster
        result_c[index][1] = right_cluster
        result_c[index][2] = delta
        result_c[index][3] = U.size[left_cluster] + U.size[right_cluster]

        U.union(left_cluster, right_cluster)

    return result

# Hash functions
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef lsh(w, t, DTYPE_t[:, ::1] points):
#'''Compute a locality sensitive hashing w: ball radius u: number of offsets t: dimension of the projection space'''
    cdef ITYPE_t U = t*2**t
    cdef DTYPE_t[:, ::1] shifts = np.random.uniform(0, 1, size=(U, t))
    cdef ITYPE_t N = points.shape[0]
    cdef ITYPE_t dim = points.shape[1]
    cdef DTYPE_t girth = 4 * w
    A = np.random.normal(size=(dim, t)) / (np.sqrt(t) * girth) # everything is normalized by 4w 
    cdef DTYPE_t[:, ::1] proj = points @ A 
    cdef ITYPE_t i, j, u, n, hashed
    
    cdef dict buckets = {}
    cdef DTYPE_t[::1] center = np.zeros(shape=t)
    cdef DTYPE_t[::1] shifts_u
    cdef DTYPE_t shift
#    missed = N
    for i in range(N):
        for u in range(U):
            shifts_u = shifts[u]
            for j in range(t):
                shift = shifts_u[j]
                center[j] = round(proj[i][j] - shift) + shift
            if dist(center, proj[i], t) <= 0.25:
                hashed = pre_hash(center)
#                hashed = hash(tuple(center)) #maybe slow
                if hashed not in buckets:
                    buckets[hashed]=[i]
                else:
                    buckets[hashed].append(i)
#                missed -= 1
                break
#    print("Missed {}/{} ({}%) dim {}".format(missed, N, missed/N*100., t))
    return buckets

# Experimental hash for tests # scikitlearn has a murmurhash module for fast hashing
cdef ITYPE_t pre_hash(DTYPE_t[::1] point):
    cdef ITYPE_t dim = point.shape[0]
    cdef ITYPE_t result = 0
    cdef ITYPE_t i = 0
    for i in range(dim):
        result = hash((result, point[i]))
    return result

# Spanner
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef spanner(DTYPE_t[:, ::1] points, U=4, d_min=0.0001, d_max=1000):
    cdef ITYPE_t N = points.shape[0]
    cdef ITYPE_t dim = points.shape[1]
    cdef ITYPE_t t = min(max(1, np.log2(N)**(2./3.)), dim)
    graph = set()
    cdef DTYPE_t scale = d_min
    cdef ITYPE_t u, center, e
    while scale < d_max:
        for u in range(U):
            buckets = lsh(scale, t, points)
            for bucket in buckets.values():
                center = np.random.choice(bucket)
                for e in bucket:
                    if e < center:
                        graph.add((e, center))
                    elif e > center: # We do nothin if e=center
                        graph.add((center, e))
        scale*=2
    # Add a star to ensure connectivity
    for e in range(1,N):
        graph.add((0, e))
        
    return graph

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] single_linkage_label(ITYPE_t[:, :] mst, DTYPE_t[:] cut_weights):
    cdef ITYPE_t N = len(mst)
    cdef ITYPE_t i, j
    
    cdef ITYPE_t[::1] index_cut_weights = np.argsort(cut_weights)
    cdef DTYPE_t[:, ::1] L = np.zeros((N,3))
    for i in range(N):
        j = index_cut_weights[i]
        L[i][0] = mst[j][0]
        L[i][1] = mst[j][1]
        L[i][2] = cut_weights[j]
    return _single_linkage_label(L)


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def all_together(points, approx, d_min=0.01, d_max=1000):
    cdef ITYPE_t N = points.shape[0]
    cdef ITYPE_t U = max(int(N**(approx**(-2))), 1)
    edges = spanner(points, U, d_min, d_max)
    
    MST = mst(points, edges)
    CW = cut_weight(points,MST)
    return single_linkage_label(MST,CW)
