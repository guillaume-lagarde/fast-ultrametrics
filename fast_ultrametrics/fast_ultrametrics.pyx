# cython: boundscheck=False, initializedcheck=False, wraparound=False, nonecheck=False, overflowcheck=False, cdivision=True, infer_types=True
# distutils: language = c++

# cython: language_level=3
# //cython profile=True
# //cython linetrace=True
from __future__ import print_function

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, round, log2, floor
import random
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from cython.operator cimport dereference, preincrement

import cyminiball

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
        cdef ITYPE_t tmp
       
        if r_m == r_n: return -1

        if self.size[r_m] >= self.size[r_n]:
            tmp = r_n
            r_n = r_m
            r_m = tmp
        self.parent[r_m] = r_n
        self.size[r_n] += self.size[r_m]
        tmp = self.succ[r_m]
        self.succ[r_m] = self.succ[r_n]
        self.succ[r_n] = tmp
            
        return r_n

    # Return the representative of the cluster containing n
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef inline ITYPE_t find(self, ITYPE_t n):
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

# 2-approximation of the maximal distance between two points
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef DTYPE_t dist_max(DTYPE_t[:, ::1] points):
    cdef ITYPE_t N = points.shape[0]
    cdef ITYPE_t dim = points.shape[1]
    cdef DTYPE_t res = 0.
    cdef ITYPE_t i
    cdef DTYPE_t d
    for i in range(1, N):
        d = dist(points[i], points[0], dim)
        if d > res:
            res = d
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[ITYPE_t, ndim=2] set_to_numpy_array(set graph):
    cdef ITYPE_t m = len(graph)
    cdef np.ndarray[ITYPE_t, ndim=2] res = np.zeros(shape=(m, 2), dtype=ITYPE)
    cdef ITYPE_t[:, ::1] res_view = res
    cdef ITYPE_t count = 0
    for x in graph:
        res_view[count][0] = x[0]
        res_view[count][1] = x[1]
        count += 1
    return res

#------------------------------------
# minimum spanning tree
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef np.ndarray[DTYPE_t, ndim=2] mst(DTYPE_t[:, ::1] points, set graph):
    cdef ITYPE_t N = points.shape[0]
    cdef ITYPE_t dim = points.shape[1]
    cdef ITYPE_t count, x, y, i
    cdef ITYPE_t[:, ::1] edges = set_to_numpy_array(graph)
    cdef ITYPE_t m = edges.shape[0]

    # Compute the order of the edges with respect to their weight
    cdef np.ndarray[DTYPE_t, ndim=1] weight = np.zeros(shape=m, dtype=DTYPE)
    cdef DTYPE_t[::1] weight_view = weight
    for i in range(m):
        weight_view[i] = dist(points[edges[i][0]], points[edges[i][1]], dim)        
    cdef ITYPE_t[::1] order = np.argsort(weight)

    # Kruskal algorithm
    cdef np.ndarray[ITYPE_t, ndim=2] mst = np.zeros(shape=(N-1, 2), dtype=ITYPE)
    cdef ITYPE_t[:, ::1] mst_view = mst
    cdef UnionFind U = UnionFind(N)
    count = 0
    for i in range(len(order)):
        x = edges[order[i]][0]
        y = edges[order[i]][1]
        if U.union(x, y) != -1:
            mst_view[count][0] = x
            mst_view[count][1] = y
            
            count+=1
            if count == N-1:
                break
    assert(count == N-1)
    return mst

@cython.nonecheck(False)
cdef bool xor(a:bool, b: bool):
    if a:
        return not b
    else:
        return b

# Exact mst using Prim algorithm on the complete graph
# If maximal = True, compute the Maximum weight spanning tree instead#
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef np.ndarray[ITYPE_t, ndim=2] exact_mst(DTYPE_t[:, ::1] points, maximal=False):
    cdef ITYPE_t pivot, e, u, smallest_index
    cdef DTYPE_t d, smallest_dist
    cdef ITYPE_t N = points.shape[0]
    cdef ITYPE_t dim = points.shape[1]
    cdef DTYPE_t[::1] dist_to_set = np.zeros(N, dtype=DTYPE) # set to -1 if already in set
    cdef ITYPE_t[::1] closest_in_set = np.zeros(N, dtype=ITYPE)
    mst = np.zeros(shape=(N-1, 2), dtype=ITYPE)

    # Initialization
    pivot = 0
    
    for e in range(N-1):
        dist_to_set[pivot] = -1. # Mark the pivot as already in the tree
        smallest_dist = -1.
        smallest_index = -1
        # Update and find minimum
        for u in range(N):
            if dist_to_set[u] != -1.: # If u is not already in the tree
                d = dist(points[u], points[pivot], dim)
                if e == 0 or xor(d < dist_to_set[u], maximal) : # switch the order if maximal is True
                    dist_to_set[u] = d
                    closest_in_set[u] = pivot
                if smallest_index == -1 or xor(dist_to_set[u] < smallest_dist, maximal):
                    smallest_dist = dist_to_set[u]
                    smallest_index = u
        # Add the edge
        pivot = smallest_index
        mst[e][0] = pivot
        mst[e][1] = closest_in_set[pivot]
    return mst

# Check if the array is sorted
cpdef bool is_sorted(DTYPE_t[:] array): 
    cdef ITYPE_t N = len(array)
    cdef ITYPE_t i
    for i in range(N-1):
        if array[i] > array[i+1]:
            return False
    return True

# Sort an array of edges in increasing order of their length
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef void sort_edges(DTYPE_t[:, ::1] points, ITYPE_t[:, ::1] edges):
    cdef ITYPE_t m = edges.shape[0]
    cdef ITYPE_t dim = points.shape[1]
    cdef ITYPE_t count, x, y, i
    cdef ITYPE_t[:] order
    cdef ITYPE_t[:, ::1] edges_copy
    
    edge_dist = np.zeros(m, dtype=DTYPE)
    cdef DTYPE_t[::1] edge_dist_c = edge_dist
    for i in range(m):
        edge_dist[i] = dist(points[edges[i][0]], points[edges[i][1]], dim)

    if not is_sorted(edge_dist):
        order = edge_dist.argsort()
        edges_copy = np.array(edges)
        for i in range(m):
            edges[i] = edges_copy[order[i]]

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
    cdef DTYPE_t d, dist_c0_to_cluster1

    sets = UnionFind(N)
    result = np.zeros(N - 1, dtype=DTYPE)
    cdef DTYPE_t[:] result_c = result

    # Sort the mst if needed
    sort_edges(points, mst)

    for i in range(N-1):
        c0 = sets.find(mst[i][0])
        c1 = sets.find(mst[i][1])
        if sets.size[c0] < sets.size[c1]:
            c0, c1 = c1, c0

        # Compute the maximum distance between c0 and points of the other cluster
        dist_c0_to_cluster1 = 0.
        # For each v in the cluster of c1..
        v = c1
        while True:
            d = dist(points[c0], points[v], dim)
            if d > dist_c0_to_cluster1:
                dist_c0_to_cluster1 = d
            v = sets.succ[v]
            if v == c1: break

        # Expression of the 3-approximation of the cut-weight
        result_c[i] = dist_c0_to_cluster1 + radius[c0]      

        # Test
#        m = 0.
#        u = c0
#        while True:
#            v = c1
#            while True:
#                d = dist(points[u], points[v], dim)
#                m = max(m, d)
#                v = sets.succ[v]
#                if v == c1: break
#            u = sets.succ[u]
#            if u == c0: break
#        m2 = result_c[i]
#        assert(m2 >= m)
#        assert(m2 <= m * 3.)
        
        # Perform the union and update the radius
        assert( sets.union(c0, c1) == c0 )
        radius[c0] = max(radius[c0], dist_c0_to_cluster1)
    return result

# 5-approximation of the cut-weight
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def old_cut_weight(DTYPE_t[:, ::1] points, ITYPE_t[:, ::1] mst):
    cdef ITYPE_t N = points.shape[0]
    cdef ITYPE_t dim = points.shape[1]

    cdef DTYPE_t[:] radius = np.zeros(N, dtype=DTYPE)
    cdef ITYPE_t c0, c1, v, i
    cdef DTYPE_t d, dist_c0_to_cluster1

    sets = UnionFind(N)
    result = np.zeros(N - 1, dtype=DTYPE)
    cdef DTYPE_t[:] result_c = result

    # Sort the mst if needed
    sort_edges(points, mst)

    for i in range(N-1):
        c0 = sets.find(mst[i][0])
        c1 = sets.find(mst[i][1])
        if sets.size[c0] < sets.size[c1]:
            c0, c1 = c1, c0

        # Compute the maximum distance between c0 and points of the other cluster
        dist_c0_to_cluster1 = 0.
        # For each v in the cluster of c1..
        v = c1
        while True:
            d = dist(points[c0], points[v], dim)
            if d > dist_c0_to_cluster1:
                dist_c0_to_cluster1 = d
            v = sets.succ[v]
            if v == c1: break

        # Expression of the 5-approximation of the cut-weight
        d = dist(points[c0], points[c1], dim)
        result_c[i] = 5 * max(d, radius[c0] - d, radius[c1] - d)

        # Perform the union and update the radius
        assert( sets.union(c0, c1) == c0 )
        radius[c0] = max(radius[c0], dist_c0_to_cluster1)
    return result

# Exact cut weight algorithm in quadratic time for the sake of comparison
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def exact_cut_weight(DTYPE_t[:, ::1] points, ITYPE_t[:, ::1] mst):
    cdef ITYPE_t N = points.shape[0]
    cdef ITYPE_t dim = points.shape[1]

    cdef DTYPE_t[:] radius = np.zeros(N, dtype=DTYPE)
    cdef ITYPE_t c0, c1, u, v, e
    cdef DTYPE_t d, cw

    sets = UnionFind(N)
    result = np.zeros(N - 1, dtype=DTYPE)
    cdef DTYPE_t[:] result_c = result
    
    # Sort the mst in place if needed
    sort_edges(points, mst)

    for e in range(N-1):
        c0 = sets.find(mst[e][0])
        c1 = sets.find(mst[e][1])

        # Compute the exact cut weight
        cw = 0.

        # For each u in the component of c0..
        u = c0
        while True:
            # For each v in the component of c1..
            v = c1
            while True:
                d = dist(points[u], points[v], dim)
                if d > cw:
                    cw = d
                    max_u = u
                    max_v = v
                v = sets.succ[v]
                if v == c1: break
            u = sets.succ[u]
            if u == c0: break
        result_c[e] = cw
        
        sets.union(c0, c1)
    return result

# Takes a list of edges 
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef tree_structure(ITYPE_t[:, ::1] mst):
    cdef ITYPE_t N = mst.shape[0] + 1
    
    cdef ITYPE_t i, mid, left, right, left_size, right_size, size, node
    cdef DTYPE_t d

    cdef DTYPE_t[:] mst_dist = np.array([float(x) for x in range(N-1)])
    
    # Single linkage label related to legnth of edges
    # Format: tree[n] = left cluster, right_cluster, delta, size
    cdef np.ndarray[DTYPE_t, ndim=2] tree = single_linkage_label(mst, mst_dist)

    assert(tree.shape[0] == N-1)
    index = np.zeros(N, dtype=ITYPE)
    cdef ITYPE_t[::1] index_view = index 
    node_info = np.zeros(shape=(N-1, 4), dtype=ITYPE) # start, mid, end, edge num
    cdef ITYPE_t[:, ::1] node_info_view = node_info
    stack = [2*N-2]
    i = N-1
    cdef ITYPE_t count_leaves = N
    
    while stack != []:
        node = stack.pop()
        if node >= N:
            i -= 1
            left, right, _, size = tree[node - N]
            if left >= N:
                left_size = tree[left - N][3]
            else:
                left_size = 1
            right_size = size - left_size
            # First the largest node
            if right_size > left_size:
                left, right = right, left
                mid = count_leaves - left_size
            else:
                mid = count_leaves - right_size
            node_info_view[i][0] = count_leaves - size # Start of the node
            node_info_view[i][1] = mid # Separation between left and right children
            node_info_view[i][2] = count_leaves # end of the node
            node_info_view[i][3] = node - N # index of the corresponding edge
            stack.append(left)
            stack.append(right)
        else:
            count_leaves -= 1
            index_view[count_leaves] = node
    assert(count_leaves == 0)
    assert(i == 0)
    return (index, node_info, int(np.log2(N)) + 1)

# A structures to stack 2-dimensional arrays or extend the highest one
cdef class ArrayStack():
    cdef DTYPE_t[:, ::1] data
    cdef ITYPE_t[::1] index
    cdef ITYPE_t size
    cdef ITYPE_t data_size
    cdef ITYPE_t dim
    
    def __init__(self, dim, max_stack, capacity):
        self.data = np.empty(shape=(capacity, dim), dtype=DTYPE)
        self.index = np.zeros(max_stack, dtype=ITYPE)
        self.size = 0
        self.data_size = 0  
        self.dim = dim
        
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef inline void push_array(self, DTYPE_t[::1] array):
        self.data[self.data_size] = array
        self.data_size += 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef inline void push(self):
        self.size += 1
        self.index[self.size] = self.data_size

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef inline void pop(self):
        self.data_size = self.index[self.size]
        self.size -= 1

# Estimate the cut-weight by fitting the clusters into bounding-balls
# This gives a (sqrt(2) + epsilon)-approximation of cut-weights
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef cut_weight_bounding_ball(DTYPE_t[:, ::1] points, ITYPE_t[:, ::1] mst, DTYPE_t eps=0.2):

    cdef ITYPE_t N = points.shape[0]
    cdef ITYPE_t dim = points.shape[1]
    
    cdef ITYPE_t i0, i, max_stack, start, mid, end, start_index, edge, max_point
    cdef DTYPE_t d, r, r2, max_d
    cdef DTYPE_t[::1] C = np.zeros(dim, dtype=DTYPE)
    cdef ITYPE_t[::1] index
    
    cdef UnionFind sets = UnionFind(N)
    result = np.zeros(N - 1, dtype=DTYPE)
    cdef DTYPE_t[:] result_c = result

    # Sort the mst in place
    sort_edges(points, mst)

    index, node_info, max_stack = tree_structure(mst)

    # Estimate the needed capacity of the stack
    cdef DTYPE_t d_max = dist_max(points)
    cdef DTYPE_t d_min = d_max
    for i in range(N-1):
        d = dist(points[mst[i][0]], points[mst[i][1]], dim)
        if d > 0.:
            d_min = d
            break
    cdef ITYPE_t core_set_max_size = int(2. * np.log(d_max/d_min) / np.log(1 + eps)) + 1
    cdef ITYPE_t capacity = min(N, max_stack * core_set_max_size)
    
    cdef DTYPE_t[:, ::1] centers = np.zeros(shape=(max_stack, dim), dtype=DTYPE) # Centers of bounding balls
    cdef DTYPE_t[::1] boundind_ball_radius = np.zeros(max_stack, dtype=DTYPE) # Radius of bounding ball
    cdef DTYPE_t[::1] radius = np.zeros(max_stack, dtype=DTYPE) # Real radius of the cluster wrt centers
    cdef ArrayStack stack = ArrayStack(dim, max_stack, capacity)
    
    for j in range(N-1):
        start = node_info[j][0]
        mid = node_info[j][1]
        end = node_info[j][2]
        edge = node_info[j][3]
        # If some of the children are leaves, add them to the stack
        if mid == start + 1:
            stack.push()
            stack.push_array(points[index[start]])
            centers[stack.size-1] = points[index[start]]
            boundind_ball_radius[stack.size-1] = 0.
            radius[stack.size-1] = 0.
        if end == mid + 1:
            stack.push()
            stack.push_array(points[index[mid]])
            centers[stack.size-1] = points[index[mid]]
            boundind_ball_radius[stack.size-1] = 0.
            radius[stack.size-1] = 0.

        # Destroy the top of the stack
        stack.pop()

        # Initialization
        C[:] = centers[stack.size-1]
        r = radius[stack.size-1]
        
        # Computes the distance
        max_d = 0.
        max_point = -1
        for i0 in range(mid, end):
            i = index[i0]
            d = dist(C, points[i], dim)
            if d > max_d:
                max_d = d
                max_point = i

        # Expression of the (sqrt(2)+eps)-approximation of the cutweight
        result_c[edge] = max_d + radius[stack.size-1]
        r = max(r, max_d)

        # Test
#        m = 0.
#        for i in range(start, mid):
#            for j in range(mid, end):
#                d = dist(points[index[i]], points[index[j]], dim)
#                m = max(m, d)
#        m2 = result_c[edge]
#        assert(m2 >= m)
#        assert(m2 <= m * (1.42 + eps))
        
        while r > boundind_ball_radius[stack.size-1] * (1. + eps):
            stack.push_array(points[max_point])
            C, r2 = cyminiball.compute(stack.data[stack.index[stack.size]:stack.data_size])
            r = sqrt(r2)
            boundind_ball_radius[stack.size-1] = r
            
            # If the previous ball of the left child is in the ball B(C, r),
            # then only check the right child on the next iteration
            #if dist(C, centers[stack.size-1], dim) + radius[stack.size-1] * (1. + eps) < r * (1. + eps):
            #    start_index = mid
            #else:
            #    start_index = start
            start_index = start
            
            for i0 in range(start_index, end):
                i = index[i0]
                d = dist(C, points[i], dim)
                if d > r:
                    r = d
                    max_point = i
        # Values for the new cluster
        radius[stack.size-1] = r
        centers[stack.size-1] = C
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

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_t, ndim=2] project(DTYPE_t girth, ITYPE_t t, DTYPE_t[:, ::1] points):
    cdef ITYPE_t dim = points.shape[1]
    cdef scale = 1. / (sqrt(t) * girth) # everything is normalized by 2*w
    cdef np.ndarray[DTYPE_t, ndim=2] A = np.random.normal(size=(dim, t), scale = scale)
    return points @ A

# Locality sensitive hash functions
# The lsh_ functions compute clusters corresponding to a particular lsh
# and add a random star on each of these clusters to the graph in input.
# All these functions return the number of different hashes found.
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef lsh_lipschitz(DTYPE_t w, ITYPE_t t, DTYPE_t[:, ::1] points, set graph):
    """
    Compute a locality sensitive hashing using Lipschitz partitions
    and connect the collision clusters with a star
    w: cells size
    t: dimension of the projection space
    graph: graph to update
    """
    cdef DTYPE_t[::1] shifts = np.random.uniform(0, 1, size=t)
    cdef ITYPE_t N = points.shape[0]
    cdef DTYPE_t[:, ::1] proj = project(2 * w, t, points) 
    cdef ITYPE_t i, j, u, n, bucket_center, l
    cdef DTYPE_t prehash

    cdef dict buckets_init = {}
    cdef map[DTYPE_t, ITYPE_t] buckets = buckets_init
    cdef DTYPE_t[::1] center = np.zeros(shape=t, dtype=DTYPE)
    cdef DTYPE_t shift
    cdef ITYPE_t[::1] order = np.random.permutation(N)
    cdef DTYPE_t[::1] linear_hash = np.random.random(size=t)
    for l in range(N):
        i = order[l]
        for j in range(t):
            center[j] = round(proj[i][j] - shifts[j])
        prehash = scal_prod(center, linear_hash, t)
        bucket_center = dereference(buckets.insert(pair[DTYPE_t,ITYPE_t](prehash, i)).first).second
        if bucket_center != i:
            if i < bucket_center:
                graph.add((i, bucket_center))
            else:
                graph.add((bucket_center, i))
    return len(buckets)

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef lsh_balls(DTYPE_t w, ITYPE_t t, DTYPE_t[:, ::1] points, graph):
    """
    Compute a locality sensitive hashing using Andoni and Indyk construction
    and connect the collision clusters with a star
    w: cells size
    t: dimension of the projection space
    graph: graph to update
    """
    cdef ITYPE_t U = 4*t*2**t
    cdef DTYPE_t[:, ::1] shifts = np.random.uniform(0, 1, size=(U, t))
    cdef ITYPE_t N = points.shape[0]
    cdef DTYPE_t[:, ::1] proj = project(4 * w, t, points)
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
    return len(buckets)

# temporary lsh function for testing modification
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef ITYPE_t lsh_experimental(DTYPE_t w, ITYPE_t t, DTYPE_t[:, ::1] points, graph):
    cdef DTYPE_t[::1] shifts = np.random.uniform(0, 1, size=t)
    cdef ITYPE_t N = points.shape[0]
    cdef ITYPE_t dim = points.shape[1]
    cdef ITYPE_t[::1] succ = np.full(N, -1, dtype=ITYPE)
    cdef DTYPE_t[::1] d = np.zeros(shape=N, dtype=DTYPE)
    A = np.random.normal(size=(dim, t)) / (sqrt(t) * 2 * w)
    cdef DTYPE_t[:, ::1] proj = points @ A 
    cdef ITYPE_t i, j, u, n, bucket_center, l
    
    cdef dict buckets = {}
    cdef DTYPE_t[::1] center = np.zeros(shape=t)
    cdef DTYPE_t shift
    
    for i in range(N):
        for j in range(t):
            center[j] = round(proj[i][j] - shifts[j])
        bucket_center = buckets.setdefault(pre_hash(center), i)
        if i == bucket_center:
            succ[i] = i
        else:
            succ[i] = succ[bucket_center]
            succ[bucket_center] = i
        d[i] = dist(center, proj[i], t)

    cdef DTYPE_t min_d
    cdef ITYPE_t succ_j
    for i in range(N):
        if succ[i] != -2:
            min_d = d[i]
            u = i
            j = succ[i]
            while j != i:
                if d[j] < min_d:
                    min_d = d[j]
                    u = j
                j = succ[j]
            j = succ[u]
            succ[u] = -2
            while j != u:
                if j < u:
                    graph.add((j, u))
                else:
                    graph.add((u, j))
                succ_j = succ[j]
                succ[j] = -2
                j = succ_j
    return len(buckets)

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef DTYPE_t scal_prod(DTYPE_t[::1] a, DTYPE_t[::1] b, ITYPE_t dim):
    cdef DTYPE_t res = 0.
    for i in range(dim):
        res += a[i] * b[i]
    return res

# Experimental hash for tests # scikitlearn has a murmurhash module for fast hashing
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef int pre_hash(DTYPE_t[::1] point):
    cdef ITYPE_t dim = point.shape[0]
    cdef int result = 0
    cdef ITYPE_t i = 0
    cdef int coord
    for i in range(dim):
        coord = int(point[i])
        result = hash((result, coord))
    return result

# Spanner
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef set spanner(DTYPE_t[:, ::1] points, scale_factor=2, d_min=0.01, d_max=None, lsh='lipschitz'):
    cdef ITYPE_t N = points.shape[0]
    cdef ITYPE_t dim = points.shape[1]
    cdef set graph = set()
    cdef DTYPE_t scale = dist_max(points)
    if d_max != None:
        scale = min(scale, d_max)
    cdef ITYPE_t u, center, e, t
    cdef ITYPE_t bucket_count = 0
    while scale > d_min and bucket_count < N:
        if lsh == 'balls':
            t = min(max(1, np.log2(N)**(2./3.)), dim)
            bucket_count = lsh_balls(scale, t, points, graph)
        elif lsh == 'lipschitz':
            t = min(max(1, np.log2(N) ), dim)
            bucket_count = lsh_lipschitz(scale, t, points, graph)
        elif lsh == 'experimental':
            t = min(max(1, np.log2(N)), dim)
            bucket_count = lsh_experimental(scale, t, points, graph)
        else:
            raise ValueError('lsh must be "lipschitz", "balls" or "exact"')
        scale/=scale_factor
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
def ultrametric(points, d_min=0.0001, scale_factor = 1.1, lsh='lipschitz', cut_weights='approximate'):
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
    if lsh == 'exact':
        MST = exact_mst(points)
    else:
        edges = spanner(points, scale_factor=scale_factor, d_min=d_min, lsh=lsh)    
        MST = mst(points, edges)
    if cut_weights == 'approximate':
        CW = cut_weight(points, MST)
    elif cut_weights == '5-approx':
        CW = old_cut_weight(points, MST)
    elif cut_weights == 'exact':
        CW = exact_cut_weight(points, MST)
    elif cut_weights == 'bounding balls':
        CW = cut_weight_bounding_ball(points, MST)
    else:
        raise ValueError('cut_weights must be "approximate", "exact" or "bounding balls"')
    
    return single_linkage_label(MST,CW)


################### Distortion
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
        self.depth = np.array([x for (_,x) in order])
        self.indices = np.array([x for (x,_) in order])
        self.positions = np.array([0]*len(order))
        for t, i in enumerate(self.indices):
            self.positions[i] = t

        # sparse table for efficient search
        self.N = len(order)
        self.K = int(floor(log2(self.N))) # max exponent
        self.st = np.array([[i]*(self.K+1) for i in range(self.N)])

        # filling the sparse table
        for j in range(1,self.K+1):
            for i in range(0, self.N):
                if i + (1 <<(j-1) ) >= self.N: break
                i1, i2 = self.st[i][j-1], self.st[i+(1<<(j-1))][j-1]
                if self.depth[i1] < self.depth[i2]:
                    self.st[i][j] = i1
                else:
                    self.st[i][j] = i2

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    def search(self, ITYPE_t u, ITYPE_t v):
        cdef ITYPE_t iu, iv, i1, i2
        iu, iv = self.positions[u], self.positions[v]
        if iu > iv:
            iu, iv = iv, iu

        cdef ITYPE_t j = int(floor(log2(iv-iu+1)))
        cdef ITYPE_t a = ( 1 << j )
        cdef ITYPE_t iv2 = iv - a + 1
        i1, i2 = self.st[iu][j], self.st[iv2][j]
        if self.depth[i1] < self.depth[i2]:
            return self.indices[i1]
        else:
            return self.indices[i2]

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    def dist(self, ITYPE_t u, ITYPE_t v):
        if u!= v:
            return self.tree[self.search(u, v)- self.n][2]
        return 0.

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def distortion_(data, tree):
    cdef ITYPE_t i, j
    cdef DTYPE_t ratio, MIN, MAX, l2
    rmq = RMQ(tree)
    n = len(data)
    d = data.shape[1]
    MAX, MIN = 0., 10000000000.
    for i in range(n):
        for j in range(i):
            l2 = dist(data[i], data[j], d)
            if l2 > 0.:
                ratio = rmq.dist(i, j) / l2
                MAX = max(ratio, MAX)
                MIN = min(ratio, MIN)
    return MAX/MIN
