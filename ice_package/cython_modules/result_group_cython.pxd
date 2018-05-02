cimport numpy as np
from .graph_invariants_cython cimport Invariant
from cpython cimport bool

cdef public list merge_groups(list groups)

cdef class ResultGroup:
    cdef public np.int8_t value
    cdef np.int8_t[:, ::1] wos
    cdef list invariant_list, results, bvs, subgroups
    cdef np.uint8_t invariant_list_length 
    cdef np.uint8_t subgroup_count
    cdef Invariant invariant
    cdef public ResultGroup try_subgroups(self, np.ndarray[np.int8_t, ndim=2] bond_variables, signed char destination_level)
    cdef public ResultGroup _try_subgroups(self, np.int8_t[:, ::1] bond_variables, signed char destination_level)
    cdef void add_result(self, np.int8_t[::1] water_orientations, np.int8_t[:, ::1] bond_variables)
    cdef inline bool belongs_to_group(self, signed char value)
    cdef np.int8_t[:, ::1] get_wos(self)
    cdef list get_subgroups_from_level(self, unsigned char level)
    cdef void merge(self, ResultGroup other_group)
    cdef void clear_wos_from_level(self, unsigned char level)
    cdef bool is_equal_with(self, ResultGroup other_group)
    cdef int get_total_number_of_grouped_wos(self, signed char destination_level)

