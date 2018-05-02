cimport numpy as np
from water_algorithm_cython cimport WaterAlgorithm

cdef list remove_equals(list symmetry_operations, bint debug)
cpdef np.ndarray[np.int8_t, ndim=1] get_vector_from_periodicity_axis_number(np.uint8_t number)
cdef np.uint8_t get_opposite_periodicity_axis_number(np.uint8_t number)
cdef np.uint8_t get_periodicity_axis_number_from_vector(vector)
cdef np.int8_t[::1] get_bond_variable_values_from_water_orientation(np.int8_t water_orientation)
cdef list do_symmetry_operation_filtering(list symmetry_operations, np.uint8_t current_molecule_no, np.ndarray[np.uint16_t, ndim = 1] order)
cdef list remove_earlier_found(list symmetry_operations)
cdef void mark_earlier_found(list found_symmetry_operations, list all_symmetry_operations)
cdef void mark_all_not_found(list symmetry_operations)
cdef bint all_found(list symmetry_operations)
cdef list get_sub_symmetry_level_improved(list symmetry_operations, np.uint8_t current_molecule_no, np.ndarray[np.uint16_t, ndim = 1] order)
cdef np.uint8_t get_water_orientation_from_bond_variable_values(np.int8_t[::1] bvvv)
cdef list filter_symmetry_operations_by_dangling_bond_profile(list symmetry_operations, np.int8_t[::1] profile)


cdef class SymmetryOperation:
    cdef public str name
    cdef public np.ndarray molecule_change_matrix, molecule_change_matrix_axis
    cdef public np.ndarray inverse_molecule_change_matrix
    cdef public np.ndarray bond_change_matrix
    cdef public np.ndarray orientation_change_matrix
    cdef public np.ndarray vector, additional_requirements, center
    cdef public np.int8_t  magnitude
    cdef public char* type
    cdef public np.int8_t order_no
    cdef public np.ndarray nn_change_matrix
    cdef public np.ndarray inverse_nn_change_matrix
    cdef public np.ndarray rotation_matrix
    cdef public np.ndarray translation_vector
    cdef public bint has_translate
    cdef int sub_symmetry_level
    cdef public dict symbolic_bond_variable_matrix
    cdef public bint found_earlier
    cdef SymmetryOperation primitive_symmetry_operation
    cdef SymmetryOperation translation_operation
    
    
    cdef public np.int8_t[::1]  apply(self, np.int8_t[::1]  water_orientations)
    cdef public np.int8_t[::1]  apply_for_dangling_bond_profile(self, np.int8_t[::1]  profile)
    cdef public bint are_symmetric(self, np.int8_t[::1] water_orientations, np.int8_t[::1] water_orientations2)
    cdef np.int8_t[::1]  apply_using_orientation_change_matrix(self, np.int8_t[::1] water_orientations)
    cdef bint are_additional_requirements_met(self, np.int8_t[::1] water_orientations)
    cdef bint are_additional_requirements_met_for_dangling_bond_profile(self, np.int8_t[::1] profile)

    cdef np.ndarray[np.int8_t, ndim=2] calculate_orientation_change_matrix(self)
    cdef void calculate_additional_requirements(self, np.int_t[:, :, :] nearest_neighbors_nos, atoms, bint debug) except *
    cdef void calculate_nn_change_matrix(self, np.int_t[:, :, :] nearest_neighbors_nos, atoms, bint debug)
    cdef np.int8_t[:, ::1] get_additional_requirements_for_molecule_no(self, int molecule_no)

    cdef dict get_symbolic_bond_variable_matrix(self, np.ndarray[np.int_t, ndim=3] nearest_neighbors_nos, WaterAlgorithm water_algorithm, bint inverse)
    cdef void initialize(self, np.int_t[:, :, ::1] nearest_neighbors_nos, atoms)
