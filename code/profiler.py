import pstats, cProfile
from union_find import *

file_name = "datasets/PENDIGITS.csv"
X = np.genfromtxt(file_name, delimiter=",")
print(len(X))

# Profiling
cProfile.runctx("all_together(X, 1.5)", globals(), locals(), "Profile.prof")

# Load and display Profile.prof
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
