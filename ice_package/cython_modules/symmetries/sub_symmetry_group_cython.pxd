from symmetry_operation_cython cimport SymmetryOperation
cimport numpy as np
from cpython cimport bool
ctypedef np.int8_t DTYPE2_t
ctypedef np.uint8_t DTYPE_t

cdef class SubSymmetryGroup:
