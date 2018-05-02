# encoding: utf-8
import sys
if sys.argv[len(sys.argv)-1] == 'PARALLEL':
    parallel = True
    sys.argv.pop()
else:
    parallel = False

from distutils.core import setup
from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    import Cython.Compiler.Options
except ImportError:
    print "Cython not available, stopping installation."
    exit(1)
    
Cython.Compiler.Options.annotate = True
inc = ["ice_package/", "ice_package/cython_modules", "ice_package/cython_modules/symmetries"]

try:
    import numpy as np
    inc.append(np.get_include())
except ImportError:
    print "Numpy not available, stopping installation."
    exit(1)

compile_time_env = { 'USE_MPI' : False}
if parallel:
    compile_time_env['USE_MPI'] = True
    try:
        import mpi4py
        inc.append(mpi4py.get_include())
    except:
        print "mpi4py not available, stopping installation."
        exit(1)

reload(sys)
sys.setdefaultencoding("utf-8")
import os

# get the list of extensions

ext_modules = [Extension("ice_package.symmetries.symmetry_operation_cython", ["ice_package/cython_modules/symmetries/symmetry_operation_cython.pyx"], include_dirs = inc, cython_compile_time_env = compile_time_env),
               Extension("ice_package.symmetries.interface_cython", ["ice_package/cython_modules/symmetries/interface_cython.pyx"], include_dirs = inc, cython_compile_time_env = compile_time_env),
               Extension("ice_package.graph_invariants_cython", ["ice_package/cython_modules/graph_invariants_cython.pyx"], include_dirs = inc, cython_compile_time_env = compile_time_env),
               Extension("ice_package.water_algorithm_cython", ["ice_package/cython_modules/water_algorithm_cython.pyx"], include_dirs = inc, cython_compile_time_env = compile_time_env),
               Extension("ice_package.result_group_cython", ["ice_package/cython_modules/result_group_cython.pyx"], include_dirs = inc, cython_compile_time_env = compile_time_env),
               Extension("ice_package.symmetries.self_symmetry_group_cython", ["ice_package/cython_modules/symmetries/self_symmetry_group_cython.pyx"], include_dirs = inc, cython_compile_time_env = compile_time_env),
               Extension("ice_package.classification_cython", ["ice_package/cython_modules/classification_cython.pyx"], include_dirs = inc, cython_compile_time_env = compile_time_env), 
               Extension("ice_package.handle_results_cython", ["ice_package/cython_modules/handle_results_cython.pyx"], include_dirs = inc, cython_compile_time_env = compile_time_env), 
               Extension("ice_package.structure_commons_cython", ["ice_package/cython_modules/structure_commons_cython.pyx"], include_dirs = inc, cython_compile_time_env = compile_time_env), 
               Extension("ice_package.energetics_cython", ["ice_package/cython_modules/energetics_cython.pyx"], include_dirs = inc, cython_compile_time_env = compile_time_env)]
              
# finally, we can pass all this to distutils
setup(
  name = 'icepackage',
  description='Python Distribution Utilities',
  version = "1.0",
  author = "Pauli Parkkinen",
  author_email = "pauli.parkkinen@helsinki.fi",
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
  packages=["ice_package", "ice_package.symmetries"],
  package_dir = {'ice_package' : 'ice_package/', 'ice_package.symmetries': 'ice_package/symmetries', 'ice_package.stools':'ice_package/stools', 'ice_package.calculator': 'ice_package/calculator', 'ice_package.misc': 'ice_package/misc'},
  py_modules=['ice_package.symmetries.symmetry_operation', 'ice_package.help_methods', 'ice_package.handle_results', 'ice_package.energy_commons','ice_package.system_commons', 'ice_package.structure_commons', 'ice_package.run', 'ice_package.handle_multiple_results', 'ice_package.classification', 'ice_package.water_algorithm', 'ice_package.result_group', 'ice_package.web_interface', 'ice_package.symmetries.interface', 'ice_package.stools.asecoords', 'ice_package.stools.coords', 'ice_package.stools.cp2ktools', 'ice_package.stools.mycalculators', 'ice_package.stools.transformations', 'ice_package.calculator.calculator', 'ice_package.calculator.calculation', 'ice_package.calculator.cp2k', 'ice_package.misc.cube', 'ice_package.misc.diffraction'],
  include_dirs = inc,
  url = "http://www.helsinki.fi/kemia/fysikaalinen/icepackage",
  scripts = ['icepkg', 'icepkg_energy']
)

