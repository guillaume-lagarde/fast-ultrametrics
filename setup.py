import numpy
from distutils.core import Extension, setup
from Cython.Build import cythonize

# define an extension that will be cythonized and compiled
ext = Extension(name="fast_ultrametrics.fast_ultrametrics",
                sources=["fast_ultrametrics/fast_ultrametrics.pyx"],
                include_dirs=[numpy.get_include()],
                language = "c++")
setup(
    name='fast_ultrametrics',
    author=['Guillaume Lagarde', 'Rémi de Verclos'],
    ext_modules=cythonize(ext)
)
