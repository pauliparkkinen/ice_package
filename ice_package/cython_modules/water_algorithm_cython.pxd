
cimport numpy as np
from .result_group_cython cimport ResultGroup
from .symmetries.symmetry_operation_cython cimport SymmetryOperation
from .symmetries.self_symmetry_group_cython cimport SelfSymmetryGroup
from cpython cimport bool

cdef object comm, comm_p
cdef public unsigned char rank, size, rank_p, size_p
cpdef tuple get_mpi_variables()
cdef inline bint wos_are_equal(np.int8_t[::1] orientation1, np.int8_t[::1] orientation2, int N)
cdef inline bint ice_rules_apply(np.int8_t[:, ::1] bond_variables, np.int_t[:, :, ::1] nearest_neighors_nos)
cdef class WaterAlgorithm:
    cdef bytes filename
    cdef float x_offset, y_offset, z_offset, x_rotation, y_rotation, z_rotation, atom_radius
    cdef bint overwrite 
    cdef bytes folder, additional_atoms
    cdef str image_type
    cdef public float O_H_distance
    cdef list symmetry_operations, primitive_symmetry_operations, translation_operations
    cdef public float O_O_distance
    cdef np.ndarray intermediate_saves, group_saves,  do_symmetry_check, order
    cdef public np.ndarray oxygen_coordinates, nearest_neighbors_nos
    cdef object atoms, cell
    cdef list graph_invariants
    cdef signed char charge
    cdef unsigned char N, dissosiation_count
    cdef bool store_bond_variables
    cdef logfile
    cdef int symmetries_loaded, profile_no, invariant_count
    cdef float time_start, symmetry_total_time, conversion_time, result_group_tryout_time, symmetry_load_time, symmetry_check_time, self_symmetry_time, iteration_time
    cdef bint save_self_symmetry_groups
    cdef dict preset_bond_values, preset_water_orientations
    cdef bool write_geometries
    cdef ResultGroup result_group
    cdef np.uint8_t [:, :, ::1] possible_combinations
    cdef void load_invariants(self)
    cdef void initialize_symmetry_operations(self)
    cdef np.ndarray[np.int8_t, ndim=2] get_symmetries(self, np.ndarray[np.int8_t, ndim=1] result,  np.ndarray[np.int_t, ndim=3] neareast_neighbors_nos, list symmetry_operations)
    cdef bint water_orientation_is_valid(self, signed char water_orientation, np.int8_t[::1] water_orient, np.uint16_t molecule_no)
    cdef np.ndarray[np.int8_t, ndim=2] handle_molecule_algorithm_2(self, np.uint16_t i, np.int8_t[:, ::1] water_orientations, np.uint8_t [:, :, ::1] possible_combinations, np.int_t[::1] nn, bint init, int *discarded)
    cdef public tuple do_initial_grouping(self, np.ndarray[np.int8_t, ndim=2] water_orientations, int depth_limit, bint scatter)
    cdef np.ndarray[np.int8_t, ndim=2] remove_symmetric_results(self, np.int8_t[:, ::1] water_orientations, np.uint16_t molecule_no, list symmetry_operations, int result_group_level, list self_symmetry_groups, bint save_self_symmetry_groups, list pending_self_symmetry_operations, SelfSymmetryGroup parent_self_symmetry_group, int depth_limit, bint scatter)
    cdef bint is_symmetric_with_another_result(self, ResultGroup result_group, np.int8_t[::1] water_orientations, signed int current_no, bint find_single, list symmetry_operations, int result_group_level, list self_symmetry_groups, bint save_self_symmetry_groups, list pending_self_symmetry_operations, SelfSymmetryGroup parent_self_symmetry_group, np.int8_t[:, ::1] bond_variables)
    cdef np.ndarray[np.int8_t, ndim=2] perform_2(self, np.ndarray[np.int8_t, ndim=2] water_orientations, np.uint16_t i, np.ndarray[np.uint16_t, ndim=1] order, np.ndarray[np.uint8_t, ndim=1] do_symmetry_check, signed char group, signed char invariant_level, list self_symmetry_groups)
    
    #cdef np.ndarray[np.float_t, ndim=2] get_single_molecule_hydrogen_coordinates(self, site, water_orientation, i, oxygen_positions,  nearest_neighbors_nos, nn_periodicity, periodicity_axis, cell)
    cdef np.ndarray[np.int8_t, ndim=2] remove_symmetries_no_invariants(self, np.int8_t[:, ::1] water_orientations,  np.int_t[:, :, ::1] nearest_neighbors_nos, np.uint16_t molecule_no, list symmetry_operations, list self_symmetry_groups, bool save_self_symmetry_groups)
    #cdef list handle_self_symmetry_groups(self, np.ndarray[np.int8_t, ndim=2] *water_orientations, np.uint16_t molecule_no, list self_symmetry_groups, list symmetry_operations, int result_group_count)
    cdef inline np.int8_t[:, ::1] remove_symmetries_for_single_group(self, ResultGroup group, np.int8_t[:, ::1] new_water_orientations, int result_group_level, int molecule_no, list symmetry_operations, list self_symmetry_groups, bint save_self_symmetry_groups, list pending_self_symmetry_operations, SelfSymmetryGroup parent_self_symmetry_group, int *count)
    cdef ResultGroup single_wo_initial_grouping(self, np.int8_t[::1] water_orientation, np.int8_t[:, ::1] bond_variables)
    cdef tuple handle_self_symmetry_group(self, SelfSymmetryGroup self_symmetry_group, list symmetry_operations, list new_symmetry_operations, np.int8_t[:, ::1] water_orientations, list new_self_symmetry_groups, int result_group_count, np.uint16_t molecule_no, ResultGroup main_result_group, bint save_self_symmetry_groups, bint all_symmetries_found, np.int_t[::1] nn, int child_level, np.uint8_t [:, :, ::1] possible_combinations)
    cdef tuple handle_self_symmetry_groups(self, np.int8_t[:, ::1] water_orientations, np.uint16_t molecule_no, list self_symmetry_groups, list symmetry_operations, int result_group_count, bint save_self_symmetry_groups, bint all_symmetries_found, np.uint8_t [:, :, ::1] possible_combinations)
    cdef inline int water_orientation_is_valid_using_possible_combinations(self, signed char water_orientation, np.ndarray[np.int8_t, ndim=1] water_orient, np.uint16_t molecule_no)
    cpdef bint additional_requirements_met(self, signed char water_orientation, np.int8_t [::1] water_orient, np.uint16_t molecule_no)
    cdef np.int8_t[::1] get_first_possible_combination(self, np.uint8_t [:, :, :, ::1] possible_combinations)
    cdef np.uint8_t[:, :, ::1] calculate_possible_combinations(self, np.uint16_t molecule_no)
    cdef bint wo_is_allowed_by_preset_bonds(self, np.uint16_t molecule_no,  int water_orientation)
    cdef bint wo_is_equal_with_any_of_previous_wos(self, np.uint16_t molecule_no, int water_orientation)
    cdef int get_coordination_number(self, int i)
    cdef dict add_dangling_bond_profile_to_preset_bond_values(self, np.int8_t[::1] profile, dict preset_bond_values)
    cdef dict add_h3o_to_preset_bond_values(self, int molecule_no, dict preset_bond_values)
    cdef dict add_oh_to_preset_bond_values(self, int molecule_no, dict preset_bond_values)
    cdef np.int8_t[::1] generate_first_result(self)
   
    
    
