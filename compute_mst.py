# Compute the mst of the dataset in argv[0] and save it in a numpy file
import sys
from fast_ultrametrics import *
from fast_ultrametrics.distortion import *

file = sys.argv[1]

if not file.endswith('.csv'):
    raise ValueError('The dataset should be a csv file')

target = mst_file(file)

X = np.genfromtxt(file, delimiter=",")

print("data shape: ", X.shape)

mst = exact_mst(X)

np.save(target, mst)
print("mst saved in " + str(target))
