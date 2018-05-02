#cython: boundscheck=False
#cython: wraparound=False
# cython: nonecheck=False
#cython: infer_types=True


from structure_commons_cython cimport get_total_dipole_moments
from libc.math cimport sqrt, pow

cimport numpy as np
import numpy as np

def get_dipole_moments(np.ndarray[np.int8_t, ndim = 2] wos, np.ndarray[np.float_t, ndim = 2] oxygen_coordinates, np.ndarray[np.int_t, ndim = 3] nearest_neighbors_nos, np.float_t O_H_distance, np.ndarray[np.float_t, ndim = 2] cell):
    return _get_dipole_moments(wos, oxygen_coordinates, nearest_neighbors_nos, O_H_distance, cell)
 
cdef tuple _get_dipole_moments(np.int8_t[:, ::1] wos, np.float_t[:, ::1] oxygen_coordinates, np.int_t[:, :, ::1] nearest_neighbors_nos, np.float_t O_H_distance, np.float_t[:, ::1]  cell):
    cdef np.ndarray[np.float_t, ndim=1] norms, normxys, dipole_moment
    cdef np.ndarray[np.float_t, ndim=2] dipole_moments
    cdef int i, number_of_results = wos.shape[0]
    cdef np.float_t norm, normxy
    if wos is not None:
        norms = np.empty(number_of_results, dtype=np.float)
        normxys = np.empty(number_of_results, dtype=np.float)
        dipole_moments = get_total_dipole_moments(wos, oxygen_coordinates, nearest_neighbors_nos, O_H_distance, cell)
        for i from 0 <= i < number_of_results:
            dipole_moment = dipole_moments[i]
            norm = euler_norm(dipole_moment)
            normxy = euler_norm(dipole_moment[0:2])
            norms[i] = norm
            normxys[i] = normxy

    return norms, normxys

cdef inline np.float_t euler_norm(np.float_t[::1] array):
    cdef np.uint8_t i, size = array.shape[0]
    cdef np.float_t result = 0.0
    for i from 0 <= i < size:
        result += pow(array[i], 2)
    result = sqrt(result)
    return result
