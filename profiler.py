import pstats, cProfile
from fast_ultrametrics import *
import numpy as np
import sklearn.neighbors as nb

n_samples = 10000
dim = 100
X = 1000*np.random.randn(n_samples, dim)

X = np.genfromtxt("datasets/DIABETES.csv", delimiter=",")
#print(len(X))


# Profiling
cProfile.runctx("ultrametric(X, 1.2, lsh='lipschitz')", globals(), locals(), "Profile.prof")
cProfile.runctx("ultrametric(X, 1.2, lsh='lipschitz', cut_weights='experimental')", globals(), locals(), "Profile2.prof")

# Load and display Profile.prof
#s = pstats.Stats("Profile.prof")
#s.strip_dirs().sort_stats("time").print_stats()
