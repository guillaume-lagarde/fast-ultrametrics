def dfs(root, tree, callback):
    n = len(tree) + 1
    if root >= n:
        dfs(int(tree[root - n][0]), tree, callback)
        dfs(int(tree[root - n][1]), tree, callback)
    else:
        callback(root)

def assign(tab, i: int, j: int):
    tab[i] = j

def clusters(tree, n_clusters=3):
    n = len(tree) + 1
    root = n*2 - 2
    cluster_roots = [root]
    while len(cluster_roots) < n_clusters:
        inner_roots = [ r for r in cluster_roots if r >= n ]
        if len(inner_roots) == 0:
            break
        to_split = max(inner_roots, key=lambda r: tree[r - n][2])
        cluster_roots.remove(to_split)
        cluster_roots.append( int(tree[to_split - n][0]) )
        cluster_roots.append( int(tree[to_split - n][1]) )

    result = [-1] * n
    for (num, root) in enumerate(cluster_roots):
        dfs( root, tree, lambda leaf: assign(result, leaf, num) )
    return result
