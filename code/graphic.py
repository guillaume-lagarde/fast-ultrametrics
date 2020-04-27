from union_find import *
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

K = 10
P = np.array([[i**2] for i in range(K)])
mst = np.array([[i,i+1] for i in range(K-1)])
N = len(P)
assert(len(mst) == N-1)

cut_weights = cut_weight(P, mst)
L = linkage_matrix(N, mst, cut_weights)
print(L)
res = single_linkage_label(N, mst, cut_weights)
plt.title('Hierarchical Clustering Dendrogram')

dendrogram(res)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
