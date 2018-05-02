cimport numpy as np

cpdef np.ndarray[np.float_t, ndim=1] get_total_dipole_moments(np.int8_t[:, ::1] wos, np.float_t[:, ::1] oxygen_coordinates, np.int_t[:, :, ::1] nearest_neighbors_nos, np.float_t O_H_distance, np.float_t[:, ::1]  cell)
cpdef np.float_t[::1] get_total_dipole_moment(np.int8_t[::1] water_orientations, oxygen_coordinates, np.int_t[:, :, ::1] nearest_neighbors_nos, np.float_t O_H_distance, np.float_t[:, ::1]  cell)
cpdef np.ndarray[np.float_t, ndim=2] get_single_molecule_hydrogen_coordinates(unsigned int site, np.int8_t water_orientation, unsigned int i, np.float_t[:, ::1] oxygen_positions,  np.int_t[::1] nearest_neighbors_nos, np.int_t[::1] nn_periodicity, np.int_t[::1] periodicity_axis, np.float_t[:, ::1] cell, np.float_t O_H_distance)
cdef np.float_t[::1] get_single_molecule_dipole_moment(unsigned int site, np.int8_t water_orientation, unsigned int i, np.float_t[:, ::1] oxygen_positions, np.int_t[::1] nearest_neighbors_nos, np.int_t[::1] nn_periodicity, np.int_t[::1] nn_periodicity_axis, np.float_t O_H_distance, np.float_t[:, ::1] cell)
cpdef list add_periodic_neighbors(p1, p2, cell, min_distance, max_distance, p2_number, result, periodicities, periodicity_axis, count, sortlist)
cpdef np.ndarray[np.float_t, ndim=3] get_selector_hydrogen_coordinates(np.int_t[::1] coordination_numbers, np.float_t[:, ::1] oxygen_positions, np.int_t[:, :, ::1] nearest_neighbors_nos, np.float_t[:, ::1] cell, np.float_t O_H_distance, dict preset_bond_values)
cpdef tuple get_bond_variables_from_atoms(atoms, float O_O_distance =*, float O_H_distance = *, bint debug = *)
#cdef np.uint8_t get_water_orientation_from_bond_variable_values(np.int8_t[::1] bvvv)
cpdef np.ndarray[np.int8_t, ndim=1] get_vector_from_periodicity_axis_number(np.uint8_t number)

