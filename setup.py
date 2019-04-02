from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy

ext_modules = [
    Extension("CMB_c", ["CMB_c.pyx"], include_dirs=[numpy.get_include()],
              extra_compile_args=["-ffast-math"]),
    Extension("boltzmann_c", ["boltzmann_c.pyx"], include_dirs=[numpy.get_include()],
              extra_compile_args=["-ffast-math"])]

setup(
  name = 'CMB Calculator',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

