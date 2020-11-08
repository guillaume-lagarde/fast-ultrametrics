# cython: profile=True
# cython: linetrace=True

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, round
import random

# Numpy must be initialized.
np.import_array()

# C++

# C types as in scikit-learn
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ITYPE = np.intp
ctypedef np.intp_t ITYPE_t

#---------------------------------------
# Computing ultrametric from cut weights

# A Union-Find structure that maintains a circular list of each cluster
cdef class UnionFind():

    cdef ITYPE_t[::1] parent
    cdef ITYPE_t[::1] size
    cdef ITYPE_t[::1] succ
    
    def __init__(self, N):
        self.parent = np.full(N, -1., dtype=ITYPE, order='C')
        self.size = np.ones(N,dtype=ITYPE)
        self.succ = np.array(range(N),dtype=ITYPE)
        
    # Make unify the clusters of n and m
    # Returns the representative of the new cluster, or -1 if none is created
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef ITYPE_t union(self, ITYPE_t m, ITYPE_t n):
        cdef ITYPE_t r_m = self.find(m)
        cdef ITYPE_t r_n = self.find(n)
       
        if r_m == r_n: return -1

        if self.size[r_m] >= self.size[r_n]:
            r_n, r_m = r_m, r_n
        self.parent[r_m] = r_n
        self.size[r_n] += self.size[r_m]
        self.succ[r_m], self.succ[r_n] = self.succ[r_n], self.succ[r_m]
            
        return r_n

    # Return the representative of the cluster containing n
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

# Euclidan distance
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

    cdef ITYPE_t[:, ::1] edges = np.array(list(graph), dtype=ITYPE)

    # Compute the order of the edges with respect to their weight
    weight = np.zeros(shape=m, dtype=DTYPE)
    cdef DTYPE_t[::1] weight_view = weight
    for i in range(m):
        weight_view[i] = dist(points[edges[i][0]], points[edges[i][1]], dim)        
    cdef ITYPE_t[::1] order = np.argsort(weight)

    # Kruskal algorithm
    mst = np.zeros(shape=(N-1, 2), dtype=ITYPE)
    U = UnionFind(N)
    count = 0
    for i in range(len(order)):
        x = edges[order[i]][0]
        y = edges[order[i]][1]
        if U.union(x, y) != -1:
            mst[count][0] = x
            mst[count][1] = y
            
            count+=1
            if count == N-1:
                break
    assert(count == N-1)
    return mst
   
#--------------------------------------
# Cut weight

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def cut_weight(DTYPE_t[:, ::1] points, ITYPE_t[:, ::1] mst):
    cdef ITYPE_t N = points.shape[0]
    cdef ITYPE_t dim = points.shape[1]

    cdef DTYPE_t[:] radius = np.zeros(N, dtype=DTYPE)
    cdef ITYPE_t c0, c1, v, i
    cdef DTYPE_t d

    sets = UnionFind(N)
    result = np.zeros(N - 1, dtype=DTYPE)
    cdef DTYPE_t[:] result_c = result
    
    # Can be remove if the mst is sorted
    mst_dist = np.zeros(N-1)
    cdef DTYPE_t[:] mst_dist_c = mst_dist
    for i in range(N-1):
        mst_dist[i] = dist(points[mst[i][0]], points[mst[i][1]], dim)
    cdef ITYPE_t[:] order = mst_dist.argsort()

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

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] _single_linkage_label(DTYPE_t[:, ::1] L):
    cdef np.ndarray[DTYPE_t, ndim=2] result

    cdef ITYPE_t left, left_cluster, right, right_cluster, index
    cdef DTYPE_t delta

    result = np.zeros((L.shape[0], 4), dtype=DTYPE)
    cdef DTYPE_t[:, ::1] result_c = result

    #cdef UnionFindUltrametric U = UnionFindUltrametric(L.shape[0] + 1)
    cdef UnionFind U = UnionFind(L.shape[0] + 1)

    # Whenever i is a representative of a cluster in U then
    # highest[i] is the highest point of this cluster (nothing is assumed otherwise)
    # This invarriant is preserved at each step
    cdef DTYPE_t[::1] highest = np.zeros(L.shape[0] * 2 + 1, dtype=DTYPE)
    for i in range(L.shape[0] + 1):
        highest[i] = i
    
    for index in range(L.shape[0]):

        left = <ITYPE_t> L[index, 0]
        right = <ITYPE_t> L[index, 1]
        delta = L[index, 2]

        left_cluster = U.find(left)
        right_cluster = U.find(right)
        
        result_c[index][0] = highest[left_cluster]
        result_c[index][1] = highest[right_cluster]
        result_c[index][2] = delta
        new_cluster = U.union(left_cluster, right_cluster)
        result_c[index][3] = U.size[new_cluster]

        highest[new_cluster] = index + L.shape[0] + 1

    return result

# Locality sensitive hash functions
# The lsh_ functions compute clusters corresponding to a particular lsh
# and add a random star on each of these clusters to the graph in input
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef lsh_lipschitz(DTYPE_t w, ITYPE_t t, DTYPE_t[:, ::1] points, graph):
    """
    Compute a locality sensitive hashing using Lipschitz paritions
    w: cells size
    t: dimension of the projection space
    """
    cdef DTYPE_t[::1] shifts = np.random.uniform(0, 1, size=t)
    cdef ITYPE_t N = points.shape[0]
    cdef ITYPE_t dim = points.shape[1]
    A = np.random.normal(size=(dim, t)) / (np.sqrt(t) * 2 * w) # everything is normalized by 2*w 
    cdef DTYPE_t[:, ::1] proj = points @ A 
    cdef ITYPE_t i, j, u, n, bucket_center, l
    
    cdef dict buckets = {}
    cdef DTYPE_t[::1] center = np.zeros(shape=t)
    cdef DTYPE_t shift
    cdef ITYPE_t[::1] order = np.random.permutation(N)
    
    for l in range(N):
        i = order[l]
        for j in range(t):
            center[j] = round(proj[i][j] - shifts[j])
        bucket_center = buckets.setdefault(pre_hash(center), i)
        if bucket_center != i:
            if i < bucket_center:
                graph.add((i, bucket_center))
            else:
                graph.add((bucket_center, i))

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef lsh_balls(DTYPE_t w, ITYPE_t t, DTYPE_t[:, ::1] points, graph):
    """
    Compute a locality sensitive hashing using Andoni and Indyk construction
    w: balls size
    t: dimension of the projection space
    """
    cdef ITYPE_t U = 4*t*2**t
    cdef DTYPE_t[:, ::1] shifts = np.random.uniform(0, 1, size=(U, t))
    cdef ITYPE_t N = points.shape[0]
    cdef ITYPE_t dim = points.shape[1]
    cdef DTYPE_t girth = 4 * w
    A = np.random.normal(size=(dim, t)) / (np.sqrt(t) * girth) # everything is normalized by 4w 
    cdef DTYPE_t[:, ::1] proj = points @ A 
    cdef ITYPE_t i, j, l, u, n, bucket_center
    
    cdef dict buckets = {}
    cdef DTYPE_t[::1] center = np.zeros(shape=t)
    cdef DTYPE_t shift
    
    cdef ITYPE_t[::1] order = np.random.permutation(N)
    for l in range(N):
        i = order[l]
        for u in range(U):
            for j in range(t):
                shift = shifts[u][j]
                center[j] = round(proj[i][j] - shift) + shift
            if dist(center, proj[i], t) <= 0.25:
                bucket_center = buckets.setdefault(pre_hash(center), i)
                if bucket_center != i:
                    if i < bucket_center:
                        graph.add((i, bucket_center))
                    else:
                        graph.add((bucket_center, i))
                break

# temporary lsh function for testing modification
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef lsh_experimental(DTYPE_t w, ITYPE_t t, DTYPE_t[:, ::1] points, graph):
    cdef ITYPE_t U = 4*t*2**t
    cdef DTYPE_t[:, ::1] shifts = np.random.uniform(0, 1, size=(U, t))
    cdef ITYPE_t N = points.shape[0]
    cdef ITYPE_t dim = points.shape[1]
    cdef DTYPE_t girth = 4 * w
    A = np.random.normal(size=(dim, t)) / (np.sqrt(t) * girth) # everything is normalized by 4w 
    cdef DTYPE_t[:, ::1] proj = points @ A 
    cdef ITYPE_t i, j, l, u, n, bucket_center
    
    cdef dict buckets = {}
    cdef DTYPE_t[::1] center = np.zeros(shape=t)
    cdef DTYPE_t shift
    
    for i in range(N):
        for u in range(U):
            for j in range(t):
                shift = shifts[u][j]
                center[j] = round(proj[i][j] - shift) + shift
            if dist(center, proj[i], t) <= 0.25:
                bucket_center = buckets.setdefault(pre_hash(center), i)
                if bucket_center != i:
                    if i < bucket_center:
                        graph.add((i, bucket_center))
                    else:
                        graph.add((bucket_center, i))
                break

            
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
cpdef spanner(DTYPE_t[:, ::1] points, scale_factor=2, d_min=0.1, d_max=1000, lsh='balls'):
    cdef ITYPE_t N = points.shape[0]
    cdef ITYPE_t dim = points.shape[1]
    cdef set graph = set()
    cdef DTYPE_t scale = d_min
    cdef ITYPE_t u, center, e, t
    while scale < d_max:
        if lsh == 'balls':
            t = min(max(1, np.log2(N)**(2./3.)), dim)
            buckets = lsh_balls(scale, t, points, graph)
        elif lsh == 'lipschitz':
            t = min(max(1, np.log2(N)), dim)
            lsh_lipschitz(scale, t, points, graph)
        elif lsh == 'experimental':
            t = min(max(1, np.log2(N)**(2./3.)), dim)
            lsh_experimental(scale, t, points, graph)
        else:
            raise ValueError('lsh must be either "lipschitz" or "balls"')
        scale*=scale_factor
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
def ultrametric(points, d_min=0.01, d_max=1000, scale_factor = 1.1, lsh='balls'):
    '''
    Compute the ultrametric
    points: the set of points as a ndarray of shape (n,d) where n is the number of points, d the dimension of the space.
    d_min: estimation of the minimal distance between two points. If d_min is greater than the real value then the smallest clusters might not be relevant.
    d_max: estimation of the diameter of the set of points. If d_max is too small, the biggest clusters might not be relevant.
    scale_factor: the cells size of the lsh families used to construct the spanner are given by d_min*scale_factor**i as soon as this is smaller than d_max. Therefore the number of lsh families is given by log(d_max/d_min)/log(scale_factor).
    lsh: "lipschitz" or "balls". 
    "lipschitz" is for using the construction of the so-called Lipschitz paritions that can be found in [CCG+98] M. Charikar, C. Chekuri, A. Goel, S. Guha, and S. A. Plotkin. "Approximating a finite metric by a small number of tree metrics". It achieves a approximation of sqrt(log n).
    "balls" is for using the construction of the space with Euclidean balls, as explained in [AI06] A. Andoni and P. Indyk. "Near-optimal hashing algorithms for approximate nearest neighbor in high dimensions". It achieves asymptotically a 5*gamma approximation when the scale_factor is set to 2**(1 / n ** (1/approx**2)).
    '''
    assert scale_factor > 1
    cdef ITYPE_t N = points.shape[0]
    edges = spanner(points, scale_factor=scale_factor, d_min=d_min, d_max=d_max, lsh=lsh)
    
    MST = mst(points, edges)
    CW = cut_weight(points,MST)
    return single_linkage_label(MST,CW)
