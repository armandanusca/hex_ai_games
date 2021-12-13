# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext

# compiler_args = []
# files = ["gamestate", "lgrm_mcts", "meta", "naive_mcts", "rave_mcts", "RootThread", "RootThreadingAgent", "unionfind", "utils"]

# ext_modules = [
#                Extension(f"{f}", sources=[f"{f}.pyx"],
#                extra_compile_args=compiler_args) for f in files
#               ]

# setup(
#     cmdclass={'build_ext': build_ext},
#       ext_modules=ext_modules
# )

from setuptools import setup
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy

Options.annotate = True

setup(
    ext_modules = cythonize("*.pyx", language_level=3, annotate = True),
    include_dirs=[numpy.get_include()],
)

# from setuptools import setup, find_packages, Extension
# from Cython.Build import cythonize
# from glob import glob

# extensions = [
#     Extension(
#         'my_proj',
#         glob('*.pyx')
#         + glob('*.cxx'))
# ]

# setup(
#     name='my-proj',
#     packages=find_packages(exclude=['doc', 'tests']),
#     ext_modules=cythonize(extensions, language="c++", language_level=3))