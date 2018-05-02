# encoding: utf-8
import sys
if sys.argv[len(sys.argv)-1] == 'PARALLEL':
    parallel = True
    sys.argv.pop()
else:
    parallel = False

from distutils.core import setup
try:
    from Cython.Distutils.extension import Extension
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    import Cython.Compiler.Options
except ImportError:
    print "Cython not available, stopping installation"
    exit(1)
Cython.Compiler.Options.annotate = True
inc = ["ice_package/"]
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
        print "mpi4py not available, parallel functionality not available"

reload(sys)
sys.setdefaultencoding("utf-8")
import os


# get the list of extensions
ext_modules = [Extension("symmetries.symmetry_operation_cython", ["ice_package/cython_modules/symmetries/symmetry_operation_cython.pyx"], include_dirs = inc, cython_compile_time_env = compile_time_env),
               Extension("symmetries.interface_cython", ["ice_package/cython_modules/symmetries/interface_cython.pyx"], include_dirs = inc, cython_compile_time_env = compile_time_env),
               Extension("graph_invariants_cython", ["ice_package/cython_modules/graph_invariants_cython.pyx"], include_dirs = inc, cython_compile_time_env = compile_time_env),
               Extension("water_algorithm_cython", ["ice_package/cython_modules/water_algorithm_cython.pyx"], include_dirs = inc, cython_compile_time_env = compile_time_env),
               Extension("result_group_cython", ["ice_package/cython_modules/result_group_cython.pyx"], include_dirs = inc, cython_compile_time_env = compile_time_env),
               Extension("symmetries.self_symmetry_group_cython", ["ice_package/cython_modules/symmetries/self_symmetry_group_cython.pyx"], include_dirs = inc, cython_compile_time_env = compile_time_env),
               Extension("classification_cython", ["ice_package/cython_modules/classification_cython.pyx"], include_dirs = inc, cython_compile_time_env = compile_time_env), 
               Extension("handle_results_cython", ["ice_package/cython_modules/handle_results_cython.pyx"], include_dirs = inc, cython_compile_time_env = compile_time_env),
               Extension("structure_commons_cython", ["ice_package/cython_modules/structure_commons_cython.pyx"], include_dirs = inc, cython_compile_time_env = compile_time_env), 
               Extension("energetics_cython", ["ice_package/cython_modules/energetics_cython.pyx"], include_dirs = inc, cython_compile_time_env = compile_time_env)]
              
# finally, we can pass all this to distutils
setup(
  name = 'icepackage',
  description='Python Distribution Utilities',
  version = "1.0",
  author = "Pauli Parkkinen",
  author_email = "pauli.parkkinen@helsinki.fi",
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
  packages=[".", "symmetries"],
  package_dir = {'.' : 'ice_package/', 'symmetries': 'ice_package/symmetries', 'stools': 'ice_package/stools', 'calculator': 'ice_package/calculator', 'misc': 'ice_package/misc'},
  py_modules=['symmetries.symmetry_operation', 'help_methods', 'handle_results', 'energy_commons', 'structure_commons', 'system_commons', 'run',  'handle_multiple_results', 'classification', 'water_algorithm', 'result_group', 'web_interface', 'symmetries.interface', 'stools.asecoords', 'stools.coords', 'stools.cp2ktools', 'stools.mycalculators', 'stools.transformations','calculator.calculator', 'calculator.calculation', 'calculator.cp2k', 'misc.cube', 'misc.diffraction'],
  include_dirs = inc,
  url = "http://www.helsinki.fi/kemia/fysikaalinen/icepackage",
  scripts = ['icepkg', 'icepkg_energy']
)

