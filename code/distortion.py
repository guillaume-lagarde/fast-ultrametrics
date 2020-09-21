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

    def search(self, u, v):
        iu, iv = self.positions[u], self.positions[v]
        if iu > iv:
            iu, iv = iv, iu
        im = iu
        m = self.depth[im]
        for i in range(iu+1, iv+1):
            mc = self.depth[i]
            if mc < m:
                im, m = i, mc
        return self.indices[im]

    def dist(self, u, v):
        if u!= v:
            return self.tree[self.search(u, v)- self.n][2]
        return 0
