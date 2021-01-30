def dfs(root, tree, callback):
    stack = [root]
    n = len(tree) + 1
    while stack != []:
        top = stack.pop()
        if top >= n:
            stack.append(int(tree[top - n][0]))
            stack.append(int(tree[top - n][1]))
        else:
            callback(top)

def assign(tab, i: int, j: int):
    tab[i] = j

def subtree_size(tree, node):
    n = len(tree) + 1
    if node < n:
        return 1
    else:
        return int(tree[node - n][3])
    
def clusters(tree, n_clusters=3, min_size=0):
    n = len(tree) + 1
    root = n*2 - 2
    cluster_roots = [root]
    while len(cluster_roots) < n_clusters:
        inner_roots = [ r for r in cluster_roots if r >= n ]
        if len(inner_roots) == 0:
            break
        to_split = max(inner_roots, key=lambda r: tree[r - n][2])
        child0 = int(tree[to_split - n][0])
        child1 = int(tree[to_split - n][1])
        cluster_roots.remove(to_split)
        for child in [child0, child1]:
            if subtree_size(tree, child) >= min_size:
                cluster_roots.append(child)
                
    result = [n_clusters] * n
    for (num, root) in enumerate(cluster_roots):
        dfs( root, tree, lambda leaf: assign(result, leaf, num) )
    return result
