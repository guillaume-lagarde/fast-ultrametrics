import pstats, cProfile
from fast_ultrametrics import *
import numpy as np
import sklearn.neighbors as nb

n_samples = 50000
dim = 20
X = np.random.randn(n_samples, dim)

X = np.genfromtxt("datasets/SHUTTLE.csv", delimiter=",")
#X = np.genfromtxt("datasets/blobsN10000d100.csv", delimiter=",")
#print(X.shape)


# Profiling
#cProfile.runctx("ultrametric(X, 1.2, lsh='lipschitz')", globals(), locals(), "Profile.prof")


cProfile.runctx("ultrametric(X, scale_factor=1.2, lsh='lipschitz', cut_weights='bounding balls')", globals(), locals(), "Profile2.prof")

print("done")

# Load and display Profile.prof
#s = pstats.Stats("Profile.prof")
#s.strip_dirs().sort_stats("time").print_stats()
