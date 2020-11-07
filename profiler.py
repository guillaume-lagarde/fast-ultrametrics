import pstats, cProfile
from fast_ultrametrics import *

file_name = "datasets/PENDIGITS.csv"
X = np.genfromtxt(file_name, delimiter=",")
print(len(X))

# Profiling
cProfile.runctx("ultrametric(X, 1.2, lsh='balls')", globals(), locals(), "Profile.prof")

# Load and display Profile.prof
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
