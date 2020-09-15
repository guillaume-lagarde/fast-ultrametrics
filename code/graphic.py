from union_find import *
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import random

file_name = "datasets/PENDIGITS.csv"

X = np.genfromtxt(file_name, delimiter=",")
K = 1000
P = np.array([[random.random()*1000] for i in range(K)])
# P = np.array([[i**2] for i in range(K)])
mst = np.array([[i,i+1] for i in range(K-1)])
N = len(P)
assert(len(mst) == N-1)

# cut_weights = cut_weight(P, mst)
# res = single_linkage_label(N, mst, cut_weights)
# res = all_together(P)
res =all_together(X, 2.5, d_min=1)
print("############",res)
plt.title('Hierarchical Clustering Dendrogram')

#dendrogram(res)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
