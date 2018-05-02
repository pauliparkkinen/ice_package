# encoding: utf-8
from distutils.core import setup
from distutils.extension import Extension

import numpy as np

import sys
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import os

# asdasdasdasdasd
# get the list of extensions
#extNames = scandir("cython_modules")
#print extNames
# and build up the set of Extension objects
#ext_modules = [makeExtension(name, ".", ".") for name in extNames]
#ext_modules = cythonize(["*.pyx", "cython_modules/graph_invariants_cython.pyx", "cython_modules/water_algorithm_cython.pyx"])#["cython_modules/graph_invariants_cython.pyx", "cython_modules/water_algorithm_cython.pyx", "cython_modules/result_group_cython.pyx", "cython_modules/symmetries/symmetry_operation_cython.pyx", "cython_modules/symmetries/interface_cython.pyx", "cython_modules/symmetries/self_symmetry_group_cython.pyx"]) 
cmdclass = { }
try:
    from Cython.Distutils import build_ext
    import mpi4py
except ImportError:
    use_cython = False
else:
    use_cython = True
use_cython = False

if use_cython:
    ids = [".", "ice_package/", "ice_package/cython_modules", mpi4py.get_include(), np.get_include()]
    ext_modules = [Extension("ice_package.symmetries.symmetry_operation_cython", ["ice_package/cython_modules/symmetries/symmetry_operation_cython.pyx", "ice_package/cython_modules/symmetries/symmetry_operation_cython.pxd"], include_dirs = ids),
                Extension("ice_package.symmetries.interface_cython", ["ice_package/cython_modules/symmetries/interface_cython.pyx", "ice_package/cython_modules/symmetries/interface_cython.pxd"], include_dirs = ids),
                Extension("ice_package.graph_invariants_cython", ["ice_package/cython_modules/graph_invariants_cython.pyx", "ice_package/cython_modules/graph_invariants_cython.pxd"], include_dirs = ids),
               Extension("ice_package.water_algorithm_cython", ["ice_package/cython_modules/water_algorithm_cython.pyx", "ice_package/cython_modules/water_algorithm_cython.pxd"], include_dirs = ids),
               Extension("ice_package.result_group_cython", ["ice_package/cython_modules/result_group_cython.pyx", "ice_package/cython_modules/result_group_cython.pxd"], include_dirs = ids),
               
               Extension("ice_package.symmetries.self_symmetry_group_cython", ["ice_package/cython_modules/symmetries/self_symmetry_group_cython.pyx", "ice_package/cython_modules/symmetries/self_symmetry_group_cython.pxd"], include_dirs = ids),
               Extension("ice_package.classification_cython", ["ice_package/cython_modules/classification_cython.pyx"], include_dirs = ids),
               Extension("ice_package.structure_commons_cython", ["ice_package/cython_modules/structure_commons_cython.pyx"], include_dirs = ids), 
               Extension("ice_package.energetics_cython", ["ice_package/cython_modules/energetics_cython.pyx"], include_dirs = ids)]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ids = [".", "/home/pamparkk/lib/python"]
    library_dirs = ["/home/pamparkk/lib/python"]
    ext_modules = [Extension("ice_package.symmetries.symmetry_operation_cython", ["ice_package/cython_modules/symmetries/symmetry_operation_cython.c"], include_dirs=ids ),
                    Extension("ice_package.symmetries.interface_cython", ["ice_package/cython_modules/symmetries/interface_cython.c"], include_dirs=ids ),
                    Extension("ice_package.graph_invariants_cython", ["ice_package/cython_modules/graph_invariants_cython.c"], include_dirs=ids ),
                   Extension("ice_package.water_algorithm_cython", ["ice_package/cython_modules/water_algorithm_cython.c"], include_dirs=ids ),
                   Extension("ice_package.result_group_cython", ["ice_package/cython_modules/result_group_cython.c"], include_dirs=ids ),
                   Extension("ice_package.symmetries.self_symmetry_group_cython", ["ice_package/cython_modules/symmetries/self_symmetry_group_cython.c"], include_dirs=ids ),
                   Extension("ice_package.classification_cython", ["ice_package/cython_modules/classification_cython.c"], include_dirs=ids ),
                   Extension("ice_package.structure_commons_cython", ["ice_package/cython_modules/structure_commons_cython.c"], include_dirs = ids), 
                   Extension("ice_package.energetics_cython", ["ice_package/cython_modules/energetics_cython.c"], include_dirs = ids)]
              
# finally, we can pass all this to distutils
setup(
  name = 'icepackage',
  description='Python Distribution Utilities',
  version = "1.0",
  author = "Pauli Parkkinen",
  author_email = "pauli.parkkinen@helsinki.fi",
  cmdclass = cmdclass,
  ext_modules = ext_modules,
  packages = ['ice_package', 'ice_package.symmetries', 'ice_package.stools', 'ice_package.calculator', 'ice_package.misc', 'test'],
  py_modules=['ice_package.graph_invariants_cython', 'ice_package.symmetries.symmetry_operation', 'ice_package.help_methods', 'ice_package.ice_ih', 'ice_package.run_energy_2', 'ice_package.ice_nt', 'ice_package.handle_results', 'ice_package.energy_commons','ice_package.system_commons', 'ice_package.structure_commons', 'ice_package.run', 'ice_package.handle_multiple_results', 'ice_package.ice_surfaces', 'ice_package.classification', 'ice_package.water_algorithm', 'ice_package.result_group', 'ice_package.symmetries.interface', 'ice_package.energy_commons', 'ice_package.web_interface', 'ice_package.stools.asecoords', 'ice_package.stools', 'ice_package.stools.coords', 'ice_package.stools.cp2ktools', 'ice_package.stools.mycalculators', 'ice_package.stools.transformations', 'ice_package.calculator.calculator', 'ice_package.calculator.calculation', 'ice_package.calculator.cp2k', 'ice_package.misc.cube', 'ice_package.misc.diffraction'],
  package_data={'.': ['install']},
  scripts = ['icepkg'],
  url = "http://www.helsinki.fi/kemia/fysikaalinen/icepackage",
  include_dirs = ids
  
)

