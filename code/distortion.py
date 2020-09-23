import math, random

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

# OLD one
# class RMQ:
#     def __init__(self, tree):
#         order = infix_order(tree)
#         self.tree = tree
#         self.n = len(tree)+1
#         self.depth = [x for (_,x) in order]
#         self.indices = [x for (x,_) in order]
#         self.positions = [0]*len(order)
#         for t, i in enumerate(self.indices):
#             self.positions[i] = t

#     def search(self, u, v):
#         iu, iv = self.positions[u], self.positions[v]
#         if iu > iv:
#             iu, iv = iv, iu
#         im = iu
#         m = self.depth[im]
#         for i in range(iu+1, iv+1):
#             mc = self.depth[i]
#             if mc < m:
#                 im, m = i, mc
#         return self.indices[im]

#     def dist(self, u, v):
#         if u!= v:
#             return self.tree[self.search(u, v)- self.n][2]
#         return 0


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
    
def distortion(data, tree):
    rmq = RMQ(tree)
    n = len(data)
    MAX, MIN = 1., 1000.
    for i in range(n):
        for j in range(i):
            l2 = dist(data[i], data[j])
            assert(rmq.dist(i, j) >= l2)
            ratio = rmq.dist(i, j) / l2
            MAX = max(ratio, MAX)
            MIN = min(ratio, MIN)
    return MAX/MIN

def fast_distortion(data, tree, nsample=10000):
    rmq = RMQ(tree)
    n = len(data)

    MAX, MIN = 1., 1000.
    for _ in range(nsample):
        i = random.randrange(n)
        j = random.randrange(n)
        if i != j:
            l2 = dist(data[i], data[j])
            ratio = rmq.dist(i, j) / l2
            MAX = max(ratio, MAX)
            MIN = min(ratio, MIN)
    return MAX/MIN
    
