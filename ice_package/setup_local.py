# encoding: utf-8
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import Cython.Compiler.Options
#Cython.Compiler.Options.annotate = True
import numpy as np
import mpi4py
import sys
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import os

# asdasdasdasdasd
# get the list of extensions
#extNames = scandir("ice_package")
#print extNames
# and build up the set of Extension objects
#ext_modules = [makeExtension(name, ".", ".") for name in extNames]
#ext_modules = cythonize(["*.pyx", "cython_modules/graph_invariants_cython.pyx", "cython_modules/water_algorithm_cython.pyx"])#["cython_modules/graph_invariants_cython.pyx", "cython_modules/water_algorithm_cython.pyx", "cython_modules/result_group_cython.pyx", "cython_modules/symmetries/symmetry_operation_cython.pyx", "cython_modules/symmetries/interface_cython.pyx", "cython_modules/symmetries/self_symmetry_group_cython.pyx"]) 
inc = [mpi4py.get_include(), np.get_include(), ".", "cython_modules/", "cython_modules/symmetries/"]
ext_modules = [Extension("symmetries.symmetry_operation_cython", ["cython_modules/symmetries/symmetry_operation_cython.pyx", "cython_modules/symmetries/symmetry_operation_cython.pxd"], include_dirs = inc),
               Extension("symmetries.interface_cython", ["cython_modules/symmetries/interface_cython.pyx", "cython_modules/symmetries/interface_cython.pxd"], include_dirs = inc),
               Extension("graph_invariants_cython", ["cython_modules/graph_invariants_cython.pyx", "cython_modules/graph_invariants_cython.pxd"], include_dirs = inc),
               Extension("water_algorithm_cython", ["cython_modules/water_algorithm_cython.pyx", "cython_modules/water_algorithm_cython.pxd"], include_dirs = inc),
               Extension("result_group_cython", ["cython_modules/result_group_cython.pyx", "cython_modules/result_group_cython.pxd"], include_dirs = inc),
               Extension("symmetries.self_symmetry_group_cython", ["cython_modules/symmetries/self_symmetry_group_cython.pyx", "cython_modules/symmetries/self_symmetry_group_cython.pxd"], include_dirs = inc),
               Extension("classification_cython", ["cython_modules/classification_cython.pyx"], include_dirs = inc), 
               Extension("handle_results_cython", ["cython_modules/handle_results_cython.pyx"], include_dirs = inc)]
              
              
# finally, we can pass all this to distutils
setup(
  name = 'icepackage',
  description='Python Distribution Utilities',
  version = "1.0",
  author = "Pauli Parkkinen",
  author_email = "pauli.parkkinen@helsinki.fi",
  cmdclass = {'build_ext': build_ext},
  packages=["ice_package", "ice_package.symmetries"],
  ext_modules = ext_modules,
  package_dir = {'ice_package' : '.', 'ice_package.symmetries' : "./symmetries"},
  py_modules=['ice_package.symmetries', 'ice_package.symmetries.symmetry_operation', 'ice_package.help_methods', 'ice_package.ice_ih', 'ice_package.ice_nt', 'ice_package.handle_results', 'ice_package.energy_commons', 'ice_package.run', 'ice_package.ice_bilayer', 'ice_package.handle_multiple_results', 'ice_package.ice_surfaces', 'ice_package.classification', 'ice_package.water_algorithm', 'ice_package.result_group', 'ice_package.symmetries.interface'],
  include_dirs = [mpi4py.get_include(), np.get_include()],
  scripts = ['icepkg']
)

