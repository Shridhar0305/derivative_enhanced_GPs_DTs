# setup.py

from setuptools import setup
from Cython.Build import cythonize
import numpy

# For GCC/Clang on Linux/macOS
compile_args = ['-fopenmp']
link_args = ['-fopenmp']

# For MSVC on Windows, the flags would be:
# compile_args = ['/openmp']
# link_args = []

setup(
    # ext_modules=cythonize("KL_minimization_cython_V2.pyx"),
    # ext_modules=cythonize("Cov_g3.pyx"),
    ext_modules=cythonize("Cov_g.pyx"),
    include_dirs=[numpy.get_include()]
)