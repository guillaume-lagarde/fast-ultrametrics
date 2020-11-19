# Fast hierarchical clustering via ultrametrics

A sub-quadratic algorithm for hierarchical clustering using the method developped in [this paper](https://arxiv.org/abs/2008.06700).

The running time and the space used by the algorithm is $O(N^{1+ε})$. The algorithm gives a proven $5ε^{-1/2}$-approximation for maximal distortion. The major interrest of this algorithm is to beat the quadratic running time and space used by the classic linkage algorithms, which makes them impractical to handle huge datasets.

## Setup

The package uses `cython`, so make sure it is installed.
```
pip install cython
```
Dowload the package from github
```
git clone https://github.com/guillaume-lagarde/fast_ultrametrics.git
```
Build the library with make in the package folder
```
cd fast_ultrametrics
make
```
You can now test the installation
```
make test
```

## Usage

The main function is `ultrametrics`.

```python
ultrametric(points, d_min=0.01, d_max=1000, scale_factor = 1.1, lsh='balls')
```
### Output
This function returns a linkage matrix as in [scipy.cluster.hierarchy](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html) representing the ultrametric.

### Arguments:
- points: the set of points as a ndarray of shape (n,d) where n is the number of points, d the dimension of the space.
- d_min: estimation of the minimal distance between two points. If d_min is greater than the real value then the smallest clusters might not be relevant.
- d_max: estimation of the diameter of the set of points. If d_max is too small, the biggest clusters might not be relevant.
- scale_factor: the cells size of the lsh families used to construct the spanner are given by d_min*scale_factor**i as soon as this is smaller than d_max. Therefore the number of lsh families is given by log(d_max/d_min)/log(scale_factor).
- lsh: "lipschitz" or "balls". 
    "lipschitz" is for using the construction of the so-called Lipschitz paritions that can be found in [CCG+98] M. Charikar, C. Chekuri, A. Goel, S. Guha, and S. A. Plotkin. "Approximating a finite metric by a small number of tree metrics". It achieves a approximation of sqrt(log n).
    "balls" is for using the construction of the space with Euclidean balls, as explained in [AI06] A. Andoni and P. Indyk. "Near-optimal hashing algorithms for approximate nearest neighbor in high dimensions". It achieves asymptotically a 5*gamma approximation when the scale_factor is set to 2**(1 / n ** (1/approx**2)).


## Basic example

```python
from fast_ultrametrics import ultrametric
import numpy as np

X = np.array([[0.0936, 0.627], [0.826, 0.228],[0.884, 0.0104],[0.165 , 0.616],[0.506, 0.597]]))

X_ultrametrics = ultrametric(X, d_min=0.001, d_max=10)
```
Analysis of the linkage matrix `X_ultrametrics` can be done with [scipy.cluster.hierarchy](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html).

For more examples see the [jupyter notebook](https://github.com/guillaume-lagarde/fast-ultrametrics/blob/master/exemple.ipynb)
