import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DOUBLE
ctypedef np.npy_intp INTP
ctypedef np.int8_t INT8

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults

np.import_array()

# from ..neighbors._dist_metrics cimport DistanceMetric
# from ..utils._fast_dict cimport IntFloatDict

# C++
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.map cimport map as cpp_map
from libc.math cimport fmax

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ITYPE = np.intp
ctypedef np.intp_t ITYPE_t

from numpy.math cimport INFINITY

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
    @cython.nonecheck(False)
    cdef ITYPE_t find(self, ITYPE_t n):
        cdef ITYPE_t p
        p = n
        # find the root
        while self.parent[n] != -1:
            n = self.parent[n]
        # path compression
        while self.parent[p] != n:
            p, self.parent[p] = self.parent[p], n
        return n


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
    cdef void union(self, ITYPE_t m, ITYPE_t n):
        self.parent[m] = self.next_label
        self.parent[n] = self.next_label
        self.size[self.next_label] = self.size[m] + self.size[n]
        self.next_label += 1

        return

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    cdef ITYPE_t fast_find(self, ITYPE_t n):
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
cpdef np.ndarray[DTYPE_t, ndim=2] _single_linkage_label(np.ndarray[DTYPE_t, ndim=2] L):
    cdef np.ndarray[DTYPE_t, ndim=2] result_arr
    cdef DTYPE_t[:, ::1] result

    cdef ITYPE_t left, left_cluster, right, right_cluster, index
    cdef DTYPE_t delta

    result_arr = np.zeros((L.shape[0], 4), dtype=DTYPE)
    result = result_arr
    U = UnionFindUltrametric(L.shape[0] + 1)

    for index in range(L.shape[0]):

        left = <ITYPE_t> L[index, 0]
        right = <ITYPE_t> L[index, 1]
        delta = L[index, 2]

        left_cluster = U.fast_find(left)
        right_cluster = U.fast_find(right)

        result[index][0] = left_cluster
        result[index][1] = right_cluster
        result[index][2] = delta
        result[index][3] = U.size[left_cluster] + U.size[right_cluster]

        U.union(left_cluster, right_cluster)

    return result_arr
    

def single_linkage_label(N, mst, cut_weights):
     index_cut_weights = numpy.argsort(cut_weights, dtype=ITYPE)
     L = np.zeros((N-1,3))
     for i in range(N-1):
         j = index_cut_weights[i]
         L[i][0] = mst[j][0]
         L[i][1] = mst[j][1]
         L[i][2] = cut_weights[j][2]
    return _single_linkage_label(L)
