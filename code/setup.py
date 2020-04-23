# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Build import cythonize
# from Cython.Distutils import build_ext
 
# extensions = [
#     Extension("union_find", ["union_find.pyx"], language = ["c++"])  # à renommer selon les besoins
# ]
 
# setup(
#     cmdclass = {'build_ext':build_ext},
#     ext_modules = cythonize(extensions),
# )

from distutils.core import Extension, setup
from Cython.Build import cythonize

# define an extension that will be cythonized and compiled
ext = Extension(name="union_find", sources=["union_find.pyx"], language = "c++")
setup(ext_modules=cythonize(ext))
