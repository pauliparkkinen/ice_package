cimport numpy as np
from libc.math cimport pow, sqrt

cdef np.float_t calculate_core_electrostatic_potential_energy(np.int_t[::1] atomic_numbers, np.float_t[:, ::1] atom_positions)
