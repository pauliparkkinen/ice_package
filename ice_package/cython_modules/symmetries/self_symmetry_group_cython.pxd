from .symmetry_operation_cython cimport SymmetryOperation
cimport numpy as np
from cpython cimport bool

cdef class SelfSymmetryGroup:
    cdef list symmetry_operations
    cdef np.ndarray water_orientations
    cdef bint finalized
    cdef SelfSymmetryGroup parent_self_symmetry_group
    cdef list child_self_symmetry_groups
    cdef list get_all_child_self_symmetry_groups(self)
    cdef np.ndarray[np.int8_t, ndim=1] get_all_water_orientations(self)
    cdef SelfSymmetryGroup get_active_parent_self_symmetry_group(self)
    cdef list get_all_active_symmetry_operations(self)
    cdef tuple get_tested_and_leftover_symmetry_operations(self, list possible_symmetry_operations)
    cdef list get_active_parent_symmetry_operations(self, list possible_symmetry_operations)
