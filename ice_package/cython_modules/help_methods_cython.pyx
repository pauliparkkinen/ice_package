from libc.math cimport sqrt, pow, sin, cos, tan
cimport cython
"""cdef np.ndarray[np.int8_t, ndim=2] merge_water_orientations(list list):
    cdef np.ndarray[np.int8_t, ndim=2] result = None
    cdef np.ndarray[np.int8_t, ndim=2] item
    cdef int i, N = list.shape[0]
    for i from 0 <= i < N:
        item = <np.ndarray[np.int8_t, ndim=2]> list[i]
        if result is None:
            result = item
        else:
            result = np.vstack((result, item))
    return result

cdef list merge(list list):
    cdef list result = []
    cdef int i, N = len(list)
    for i from 0 <= i < N:
        item = list[i]
        result.extend(item)
    return result"""

cdef inline np.float_t euler_norm(np.float_t[::1] array):
    cdef np.uint8_t i, size = array.shape[0]
    cdef np.float_t result = 0.0
    for i from 0 <= i < size:
        result += pow(array[i], 2)
    result = sqrt(result)
    return result

cdef inline np.float_t euler_distance(np.float_t[::1] a, np.float_t[::1] b):
    cdef np.float_t result = sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2) + pow(a[2] - b[2], 2))
    return result

cdef inline np.float_t[:, ::1] periodic_euler_distances(np.float_t[::1] p1, np.float_t[::1] p2, np.float_t[:, ::1] cell):
    cdef np.float_t[:, ::1] result = np.empty((27, 2))
    cdef np.float_t[::1] px = np.empty(3)
    cdef int i, j, k, a, b, c, p_ax, count = 0
    cdef float distance
    for i in range(3):
        a = i - 1
        for j in range(3):
            b = j - 1
            for k in range(3):
                c = k - 1
                px[0] = p2[0] + cell[0, 0] * a + cell[1, 0] * b + cell[2, 0] * c
                px[1] = p2[1] + cell[0, 1] * a + cell[1, 1] * b + cell[2, 1] * c
                px[2] = p2[2] + cell[0, 2] * a + cell[1, 2] * b + cell[2, 2] * c
                p_ax = 9*i + 3*j + k
                distance = euler_distance(p1, px)
                result[count, 0] = distance
                result[count, 1] = <float>p_ax
                count += 1
    return result

cdef inline np.float_t[::1] add_together(np.float_t[::1] array1, np.float_t[::1] array2):
    cdef np.float_t[::1] result = np.ndarray(array1.shape[0], dtype=np.float)
    cdef np.uint8_t i, size = array1.shape[0]
    for i from 0 <= i < size:
        result[i] = array1[i] + array2[i]
    return result

cdef inline np.float_t[::1] reduce(np.float_t[::1] array1, np.float_t[::1] array2):
    cdef np.float_t[::1] result = np.ndarray(array1.shape[0], dtype=np.float)
    cdef np.uint8_t i, size = array1.shape[0]
    for i from 0 <= i < size:
        result[i] = array1[i] - array2[i]
    return result

cdef inline np.float_t[::1] multiply(np.float_t multiplier, np.float_t[::1] array1):
    cdef np.float_t[::1] result = np.ndarray(array1.shape[0], dtype=np.float)
    cdef np.uint8_t i, size = array1.shape[0]
    for i from 0 <= i < size:
        result[i] = multiplier * array1[i]
    return result

@cython.cdivision(True)
cdef inline np.float_t[::1] divide(np.float_t[::1] array1, np.float_t divider):
    cdef np.float_t[::1] result = None
    result = np.ndarray(array1.shape[0], dtype=np.float)
    cdef np.uint8_t i, size = array1.shape[0]
    for i from 0 <= i < size:
        result[i] = array1[i] / divider
    return result

cpdef np.ndarray[np.int8_t, ndim=1] get_vector_from_periodicity_axis_number(np.uint8_t number):
    cdef np.ndarray[np.int8_t, ndim=1] result = np.zeros((3), dtype=np.int8)
    cdef np.int8_t a = number/9 -1 
    cdef np.int8_t b = np.mod(number, 9) / 3 -1 
    cdef np.int8_t c = np.mod(number, 3) -1
    result[0] = a 
    result[1] = b 
    result[2] = c 
    return result

cdef np.uint8_t get_opposite_periodicity_axis_number(np.uint8_t number):
    cdef np.ndarray[np.int8_t, ndim=1] vector = get_vector_from_periodicity_axis_number(number)
    vector *= -1
    cdef np.uint8_t result = get_periodicity_axis_number_from_vector(vector)
    return result

cdef np.uint8_t get_periodicity_axis_number_from_vector(vector):
    cdef np.int8_t a = <np.int8_t> round(vector[0]) + 1
    cdef np.int8_t b = <np.int8_t> round(vector[1]) + 1
    cdef np.int8_t c = <np.int8_t> round(vector[2]) + 1 
    if a < 0 or a > 2 or b < 0 or b > 2 or c < 0 or c > 2:
        return 28
    cdef np.uint8_t result = <np.uint8_t> (a*9 + b*3 + c)
    return result

cdef np.int8_t[::1] get_bond_variable_values_from_water_orientation(np.int8_t water_orientation):
    cdef np.int8_t[::1] result   
    
    if water_orientation == -1:
        result = np.array([0, 0, 0, 0], dtype=np.int8)
    else:
        result = bvv[water_orientation]
    return result

cpdef list get_bond_variable_values_from_water_orientations(water_orientations):
    result = []
    for wo in water_orientations:
        result.append(get_bond_variable_values_from_water_orientation(wo))
    return result

bvv = np.array([[1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1], [-1, 1, 1, -1], [-1, 1, -1, 1], [-1, -1, 1, 1], [1, -1, -1, -1], [-1, 1, -1, -1], [-1, -1, 1, -1], [-1, -1, -1, 1], [-1, 1, 1, 1], [1, -1, 1, 1], [1, 1, -1, 1], [1, 1, 1, -1]], dtype=np.int8)

cdef np.uint8_t get_water_orientation_from_bond_variable_values(np.int8_t[::1] bvvv):
    cdef np.uint8_t i, j, N = bvv.shape[0], M = bvvv.shape[0]   
    cdef bint equals
    for i from 0 <= i < N:
        equals = True
        for j from 0 <= j < M:
            if bvv[i, j] != bvvv[j]:
                equals = False
                break
        if equals:
            return i

"""def np.ndarray[np.uint8_t] get_water_orientations_from_bond_variable_values(np.int8_t[:, ::1] bond_variables):
    cdef int i, N = bond_variables.shape[0]
    cdef np.uint8_t[::1] result = np.zeros(N, dtype=np.uint8)
    for i in range(N):
        result[i] = get_water_orientation_from_bond_variable_values(bond_variables.base[i])
    return result.base
"""
cdef dict combine_symbolic_bond_variable_matrices(matrix_1, matrix_2):
    result = {}
    for molecule_no in matrix_1:
        if molecule_no not in result:
            result[molecule_no] = {}
        for neighbor_no in matrix_1[molecule_no]:
            if neighbor_no not in result[molecule_no]:
                result[molecule_no][neighbor_no] = {}
            for axis in matrix_1[molecule_no][neighbor_no]:
                indeces = matrix_1[molecule_no][neighbor_no][axis]
                result[molecule_no][neighbor_no][axis] = matrix_2[indeces[0]][indeces[1]][indeces[2]]
    return result

cdef inline np.int8_t memoryview_sum(np.int8_t[::1] array, N):
    cdef np.int8_t result = 0
    for i from 0 <= i < N:
        result += array[i]
    return result

cdef np.ndarray[np.float_t, ndim=2] mirror_through_plane(np.ndarray[np.float_t, ndim = 2] positions, np.ndarray[np.float_t, ndim=1] normal_vector, np.ndarray[np.float_t, ndim=1] point, bint debug):
    cdef float L = euler_norm(normal_vector)
    cdef float hl = 1.0+0.00001
    cdef float ll = 1.0-0.00001
    cdef float R, R_2, R_3
    assert L < hl and L > ll
    # center of mass is along the plane always
    #  calculate the constant parameter d of plane equation 
    cdef float d
    if point is None:
        d = 0.0
    else:
        d = -(np.sum(point*normal_vector))
    cdef np.ndarray[np.float_t, ndim=2] result = np.zeros_like(positions)
    cdef int i
    cdef np.ndarray[np.float_t, ndim=1] position
    cdef int N = positions.shape[0]
    
    for i from 0 <= i < N:
        position = positions[i]
        # calculate positions distance R from the plane
        # plane normal vector is normalized (sqrt(a^2+b^2+c^2)=1)
        R = np.abs(np.sum(position*normal_vector)+d)
        # advance - 2 * R along the normal vector (to the other side of the plane)
        result[i] = position - 2*R*normal_vector
        
        R_2 = np.abs(np.sum(result[i]*normal_vector) + d)
        if debug:
            print "-"
            print result[i]
            print -2*R*normal_vector
            print position
            print d
            print R
            print R_2
        if R > R_2+0.00001 or R < R_2-0.00001:
            result[i] = position + 2*R*normal_vector
            R_3 = np.abs(np.sum(result[i]*normal_vector) + d)
            assert (R < R_3+0.00001 and R > R_3-0.00001)
    return result

cdef int atom_at(np.float_t[:, ::1] scaled_positions, np.int8_t[::1] pbc, np.float_t[::1] point, np.float_t max_error, bint convert_to_real_point):
    """
        Checks if there is an atom at point
        point is a relative position
    """
    cdef int i, periodicity_axis
    cdef np.float_t[::1] real_point, relocation
    cdef np.float_t R
    if convert_to_real_point:
        real_point = get_real_point(point, pbc) # , relocation
        #periodicity_axis = get_periodicity_axis_number_from_vector(relocation)
    else:
        real_point = point
        #periodicity_axis = 13
    #print point
    #print real_point
    for i in xrange(scaled_positions.shape[0]):
        R = euler_distance(scaled_positions[i], real_point)
        #print R
        if abs(R) < max_error:
            return i #, periodicity_axis
    return -1 #, 13

cdef np.float_t[::1] get_real_point(np.float_t[::1] point, np.int8_t[::1] pbc):
    if pbc is None:
        pbc = np.array([1, 1, 1], dtype=int)
    cdef np.float_t[::1] result = point.copy()
    cdef np.float_t[::1] rec = np.zeros(3, dtype=float)
    for i in xrange(result.shape[0]):
        if pbc[i]:
            while result[i] < 0:
                rec[i] += 1.0
                result[i] += 1.0
            while result[i] >= 1:
                rec[i] -= 1.0
                result[i] -= 1.0
    #print "Relocated point %s" % rec
    return result#, rec

cdef np.ndarray[np.float_t, ndim = 2] rotate_with_matrix(np.float_t[:, ::1] positions, np.float_t[::1] center, np.float_t[:, ::1] rotation_matrix):
    cdef np.ndarray[np.float_t, ndim = 2] new_positions = np.empty_like(positions.base)
    cdef np.float_t[:, ::1] pos = positions.base - center.base
    cdef int i, N = pos.shape[0]
    for i in range(N):
        new_positions[i] = np.dot(rotation_matrix.base, positions.base[i])
    return new_positions

cdef np.ndarray[np.float_t, ndim = 2]  rotate(np.float_t[:, ::1] positions, np.float_t[::1] axis, float angle, np.float_t[::1] center):
    divide(axis, euler_norm(axis))
    cdef float c = cos(angle)
    cdef float s = sin(angle)
    
    cdef np.ndarray[np.float_t, ndim = 2] p =  positions.base - center.base
    cdef np.ndarray[np.float_t, ndim = 1] ax = axis.base, cen 
    if center is None:
        cen = np.zeros(3)
    else:
        cen = center.base
    cdef int i, N = positions.shape[0]
    cdef np.ndarray[np.float_t, ndim = 2] result = positions.base.copy()
    result[:] = (c * p - np.cross(p, s * ax) +
                     np.outer(np.dot(p, ax), (1.0 - c) * ax) +
                     cen)
    return result

    
    
            
