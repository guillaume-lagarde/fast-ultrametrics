# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>

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

cdef class UnionFind(object):

    # cdef ITYPE_t next_label
    
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
        cdef ITYPE_t r_m
        cdef ITYPE_t r_n
        
        r_m = self.find(m)
        r_n = self.find(n)

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
        # find the highest node in the linkage graph so far
        while self.parent[n] != -1:
            n = self.parent[n]
        # provide a shortcut up to the highest node
        while self.parent[p] != n:
            p, self.parent[p] = self.parent[p], n
        return n
