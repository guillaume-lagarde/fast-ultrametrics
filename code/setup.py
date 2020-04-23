import numpy
from distutils.core import Extension, setup
from Cython.Build import cythonize

# define an extension that will be cythonized and compiled
ext = Extension(name="union_find", sources=["union_find.pyx"], language = "c++")
setup(ext_modules=cythonize(ext))
