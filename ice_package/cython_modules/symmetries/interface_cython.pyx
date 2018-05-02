#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: infer_types=True
from ase import Atoms
import numpy as np
cimport numpy as np
from .symmetry_operation_cython cimport SymmetryOperation, remove_equals, get_vector_from_periodicity_axis_number, get_periodicity_axis_number_from_vector
#from symmetry_operation_cython import SymmetryOperation
import math
from ase.visualize import view
from help_methods import change_basis, change_basis_for_vectors, apply_symmetry_operation_to_coordinates, get_d6h_symmetry_operations, rotation_matrix, improper_rotation_matrix, change_basis_for_transformation_matrix
from libc.math cimport pow, sqrt, abs, sin, cos
from time import time
ctypedef np.int8_t DTYPE2_t
ctypedef np.uint8_t DTYPE_t
DTYPE = np.uint8
DTYPE2 = np.int8
cdef float pi2 =  6.283185307

class Degeneracy:
    ASYMMETRIC=0
    SYMMETRIC=1
    SPHERICAL=2 
    

cdef inline np.float_t[:, ::1] positions_related_to_center_of_mass(np.float_t[::1] center_of_mass, np.float_t[:, ::1] positions):
    cdef np.float_t[:, ::1] result = reduce_from_vectors(positions, center_of_mass)
    return result 

cdef inline np.float_t[:, ::1] reduce_from_vectors(np.float_t[:, ::1] vectors, np.float_t[::1] array2):
    cdef np.float_t[:, ::1] result = vectors.copy()
    cdef DTYPE_t i, j, size = vectors.shape[0], size2 = vectors.shape[1]
    for i from 0 <= i < size:
        for j from 0 <= j < size2:
            result[i, j] = vectors[i, j] - array2[j]
    return result
    
cdef inline np.float_t[:, ::1] multiply_vectors_and_float(np.float_t[:, ::1] vectors, float multiplier):
    cdef np.float_t[:, ::1] result = vectors.copy()
    cdef DTYPE_t i, j, size = vectors.shape[0], size2 = vectors.shape[1]
    for i from 0 <= i < size:
        for j from 0 <= j < size2:
            result[i, j] = vectors[i, j]  * multiplier
    return result

cdef inline void divide(np.float_t[::1] dividend, float divider):
    cdef DTYPE_t i, N = dividend.shape[0]
    for i in range(N):
        dividend[i] = dividend[i] / divider
        
cdef inline float sum_vector(np.float_t[::1] vector):
    cdef float result = 0.0
    cdef i, N = vector.shape[0]
    for i in range(N):
        result = result + vector[i]
    return result
    
cdef inline void add_to_vectors(np.float_t[:, ::1] vectors, np.float_t[::1] vector2):
    cdef i, N = vectors.shape[0]
    for i in range(N):
        vectors[i] = add_vectors(vectors[i], vector2)
    
cdef inline np.float_t[::1] add_vectors(np.float_t[::1] vector, np.float_t[::1] vector2):
    cdef np.float_t[::1] result = vector.copy()
    cdef i, N = vector.shape[0]
    for i in range(N):
        result[i] = vector[i] + vector2[i]
    return result

cdef inline np.float_t[::1] cross_product(np.float_t[::1] array1, np.float_t[::1] array2):
    cdef np.float_t[::1] result = array1.copy()
    result[0] = array1[1]*array2[2]-array1[2]*array2[1]
    result[1] = array1[2]*array2[0]-array1[0]*array2[2]
    result[1] = array1[0]*array2[1]-array1[1]*array2[0]
    return result
    
cdef inline np.float_t[::1] add_to_vector(np.float_t[::1] vector, float added):
    cdef np.float_t[::1] result = vector.copy()
    cdef int i, N = vector.shape[0]
    for i in range(N):
        result[i] = vector[i] + added
    return result
     
cdef inline np.float_t[::1] multiply_vectors(np.float_t[::1] multiplicant, np.float_t[::1] multiplier):
    cdef np.float_t[::1] result = multiplicant.copy()
    cdef DTYPE_t i, N = multiplicant.shape[0]
    for i in range(N):
        result[i] = multiplicant[i] * multiplier[i]
    return result
    
cdef inline np.float_t[::1] multiply(np.float_t[::1] multiplicant, float multiplier):
    cdef np.float_t[::1] result = multiplicant.copy()
    cdef DTYPE_t i, N = multiplicant.shape[0]
    for i in range(N):
        result[i] = multiplicant[i] * multiplier
    return result

cdef inline np.float_t[::1] reduce(np.float_t[::1] array1, np.float_t[::1] array2):
    cdef np.float_t[::1] result = array1.copy()
    cdef DTYPE_t i, size = array1.shape[0]
    for i from 0 <= i < size:
        result[i] = array1[i] - array2[i]
    return result

cdef inline np.float_t get_distance(np.float_t[::1] a, np.float_t[::1] b):
    cdef np.float_t result = sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2) + pow(a[2] - b[2], 2))
    return result

cdef inline np.float_t get_length(np.float_t[::1] a):
    cdef np.float_t result = sqrt(pow(a[0], 2) + pow(a[1], 2) + pow(a[2], 2))
    return result

cdef inline np.ndarray[DTYPE2_t, ndim=1] get_equals(np.float_t[::1] center_of_mass, np.float_t[:, ::1] positions, np.float_t[:, ::1] new_positions, DTYPE2_t[::1] atomic_numbers, float error_tolerance, bint debug):
    cdef np.float_t[:, ::1] pos = positions_related_to_center_of_mass(center_of_mass, positions)
    cdef np.float_t[:, ::1] new_pos = positions_related_to_center_of_mass(center_of_mass, new_positions)
    cdef np.ndarray[DTYPE2_t, ndim=1] result = np.empty(positions.shape[0], dtype=DTYPE2)
    result[:] = -1 
    cdef DTYPE2_t l
    cdef int i, j, k, atomic_number_i, atomic_number_j
    cdef int N  = positions.shape[0]
    cdef int M = new_positions.shape[0], O
    cdef np.float_t[::1] position, new_position
    cdef float R
    if debug:
        min_r = np.zeros(N)
        min_r[:] = 10.0
    #if debug:
    #    print new_positions
    #    print positions        
    for i from 0 <= i < N:
        position = pos[i]
        atomic_number_i = atomic_numbers[i]
        for j from 0 <= j < M:
            new_position = new_pos[j]
            R = get_distance(position, new_position)
            if debug and R < min_r[i]:
                min_r[i] = R
            atomic_number_j = atomic_numbers[j]
            if R < error_tolerance and atomic_number_i == atomic_number_j:
                if result[i] == -1:
                    result[i] = j
                else:
                    raise Exception('Error tolerance is too big! (get_equals)')
    O = positions.shape[0]
    for k in range(O):
        l = <DTYPE2_t>result[k]
        if l == -1:
            if debug:
                print min_r
            return None
    if debug:
        print result
    return result

cdef get_closest_position(np.ndarray[np.float_t, ndim=1] position, np.ndarray[np.float_t, ndim=2] new_positions):
    smallest_distance = None
    result = None
    for j, new_position in enumerate(new_positions):
        R = get_distance(position, new_position)
        if smallest_distance == None or R < smallest_distance:
            smallest_distance = R
            result = j
    #print position-new_position
    #print smallest_distance
    return result

def get_closest_positions(positions, new_positions):
    smallest_distance = None
    result = np.zeros(len(positions), dtype=int)
    for i, position in enumerate(positions):
        result[i] = get_closest_position(position, new_positions)
    return result



cdef np.ndarray[DTYPE2_t, ndim=1] get_equals_periodic(np.float_t[:, ::1] positions, np.float_t[:, ::1]  new_positions, np.float_t[:, ::1] cell, DTYPE2_t[::1] pbc, float error_tolerance, bint debug):
    cdef np.ndarray[DTYPE2_t, ndim=1] result = np.zeros(positions.shape[0], dtype=DTYPE2)
    #cdef np.ndarray[DTYPE2_t, ndim=1] periodicity_axes = np.zeros(positions.shape[0], dtype=DTYPE2)
    cdef np.float_t[:, ::1] scaled_positions = get_scaled_positions(cell, positions, pbc), scaled_new_positions = get_scaled_unnormalized_positions(cell, new_positions, pbc)
    cdef DTYPE2_t found_atom, periodicity_axis, l
    cdef int i, j, O
    result[:] = -1 
    #periodicity_axes.fill(13)
    for i in xrange(scaled_new_positions.shape[0]):
        found_atom = atom_at(scaled_positions, pbc, scaled_new_positions[i], 0.01, True) # , periodicity_axis
        if debug:
            for j, pos in enumerate(scaled_positions):
                print positions[i]
                print pos
            print "---*"
            print scaled_new_positions[i]
            print found_atom, periodicity_axis
            print "---"
        
        #found_atom_inverse = inverse_atom_at(new_atoms, atoms.get_scaled_positions()[i], max_error=error_tolerance, symbol=atoms[i].get_symbol(), periodic_pos = periodic_pos)
        if result[i] == -1:
            result[i] = found_atom
            #periodicity_axes[i] = periodicity_axis
        else:
            raise Exception('Error tolerance is too big! (get_equals)')
        #if inverse_result[i] == -1:
        #    inverse_result[i] = found_atom_inverse
        #else:
        #    raise Exception('Error tolerance is too big! (Inverse @ get_equals)')
    if debug:
        print result#, periodicity_axes
    O = positions.shape[0]
    for k in range(O):
        l = <DTYPE2_t>result[k]
        if l == -1:
            return None#, None
    return result#, periodicity_axes

def get_molecule_change_matrix(positions, rotation, translation, cell, centering):
    """
        Note: Calculates new type of molecule change matrix that contains information about in which cell the atom switched in place was of the atom
    """
    cdef np.ndarray[DTYPE2_t, ndim=1] result = np.zeros(positions.shape[0], dtype=DTYPE2)
    result.fill(-1)
    cdef np.ndarray[DTYPE2_t, ndim=2] original_cell_displacement = np.zeros((positions.shape[0], 3), dtype=DTYPE2)
    displacement_vectors = []
    for x in range(-3, 3):
        for y in range(-3, 3):
            for z in range(-3, 3):
                displacement_vectors.append(np.array([x, y, z]))
    for displacement_vector in displacement_vectors:
        if -1 in result:
            displacement = np.dot(displacement_vector, cell) 
            new_positions = apply_symmetry_operation_to_coordinates(positions + displacement, rotation, translation , centering = centering) #np.dot(translation, cell)
            for i, position in enumerate(positions):
                if result[i] == -1:
                    result[i] = equal_position_found(new_positions, position, max_error=0.01)
                    original_cell_displacement[i] = np.array(displacement_vector, dtype=DTYPE2)
    if -1 in result:
        print "ERRORORRR! Replacing atom not found"
        raw_input()
                
       
    #cdef np.ndarray[DTYPE2_t, ndim = 2] returned = np.array([result, original_cell_displacement], dtype=DTYPE2)
    return result, original_cell_displacement

def get_change_matrices(nearest_neighbors_nos, positions, rotation, translation, cell, centering):
    """
        Note: Calculates new type of molecule change matrix that contains information about in which cell the atom switched in place was of the atom
    """
    replacing_molecules = {}
    cdef np.ndarray[DTYPE2_t, ndim=1] periodicity_axis
    cdef np.ndarray[DTYPE2_t, ndim=2] result = np.zeros_like(nearest_neighbors_nos[0])
    cdef np.ndarray[DTYPE2_t, ndim=1] molecule_change_matrix = np.zeros(positions.shape[0], dtype=DTYPE2)
    for molecule_no in range(nearest_neighbors_nos.shape[1]):
        if molecule_no in replacing_molecules:
            new_molecule_no, new_cell_position = replacing_molecules[molecule_no]
        else:
            new_molecule_no, new_cell_position =  get_replacing_molecules_no(positions[molecule_no], positions, rotation, translation, cell, centering)
            replacing_molecules[molecule_no] = [new_molecule_no, new_cell_position]
        molecule_change_matrix[molecule_no] = new_molecule_no
        
        for i, nn in enumerate(nearest_neighbors_nos[0][molecule_no]):
            axis = nearest_neighbors_nos[2][molecule_no][i]
            number = -1
            if axis == 13 and nn in replacing_molecules:
                number, cell_position = replacing_molecules[nn]
            else:
                periodicity_axis = get_vector_from_periodicity_axis_number(axis)
                position = positions[nn] + periodicity_axis
                number, cell_position = get_replacing_molecules_no(position, positions, rotation, translation, cell, centering)
                if axis == 13:
                    replacing_molecules[nn] = [number, cell_position]
                new_axis = get_periodicity_axis_number_from_vector(cell_position - new_cell_position) # POSSIBLE ERROR SOURCE (flip the order to fix if errors found)
                for j, nn_j in enumerate(nearest_neighbors_nos[0][new_molecule_no]):
                    axis_j = nearest_neighbors_nos[2][new_molecule_no][j]
                    if nn_j == number and axis_j == new_axis:
                        result[molecule_no][i] = j 
    return molecule_change_matrix, result
        
                
                
    
    for displacement_vector in displacement_vectors:
        if -1 in result:
            displacement = np.dot(displacement_vector, cell) 
            new_positions = apply_symmetry_operation_to_coordinates(positions + displacement, rotation, translation, centering = centering) #np.dot(translation, cell)
            for i, position in enumerate(positions):
                if result[i] == -1:
                    result[i] = equal_position_found(new_positions, position, max_error=0.01)
                    original_cell_displacement[i] = np.array(displacement_vector, dtype=DTYPE2)
    if -1 in result:
        print "ERRORORRR! Replacing atom not found"
        raw_input()
                
       
    #cdef np.ndarray[DTYPE2_t, ndim = 2] returned = np.array([result, original_cell_displacement], dtype=DTYPE2)
    return result, original_cell_displacement



def get_replacing_molecules_no(position, positions, rotation, translation, cell, centering):
    """
        Note: Calculates new type of molecule change matrix that contains information about in which cell the atom switched in place was of the atom
    """

    displacement_vectors = []
    for x in range(-3, 3):
        for y in range(-3, 3):
            for z in range(-3, 3):
                displacement_vectors.append(np.array([x, y, z]))
    for displacement_vector in displacement_vectors:
        displacement = np.dot(displacement_vector, cell) 
        new_positions = apply_symmetry_operation_to_coordinates(positions + displacement, rotation, translation, centering = centering) #np.dot(translation, cell)
        replacing_molecule = equal_position_found(new_positions, position, max_error=0.01)
        if replacing_molecule != -1:
            return replacing_molecule,  np.array(displacement_vector, dtype=DTYPE2)
                
       
    #cdef np.ndarray[DTYPE2_t, ndim = 2] returned = np.array([result, original_cell_displacement], dtype=DTYPE2)
    return -1, None
        

                    

cdef SymmetryOperation check_inversion_center(np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim=2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers, np.ndarray[np.float_t, ndim = 1] com, float error_tolerance, bint debug):
    cdef np.float_t[:, ::1] new_positions = positions_related_to_center_of_mass(com, positions)
    positions = -positions + com
    cdef np.ndarray[DTYPE2_t, ndim=1] result = get_equals(com, positions, new_positions, atomic_numbers, error_tolerance, debug)
    
    if result == None:
        return None
    print "Inversion center found"
    return SymmetryOperation("i", result, None, type='i', order_no=0, magnitude=1)


cdef SymmetryOperation check_improper_rotation(np.float_t[:, ::1] positions, np.float_t[:, ::1] cell, DTYPE2_t[::1] pbc, DTYPE2_t[::1] atomic_numbers, np.float_t[::1] vector, float angle, np.float_t[::1] center, DTYPE_t n, DTYPE_t order_no, float error_tolerance, bint debug, np.float_t[::1] translation):
    cdef np.float_t[:, ::1] new_positions = rotate(positions, vector, angle, center)
    new_positions = mirror_through_plane(new_positions, vector, center, debug)
    cdef np.ndarray[DTYPE2_t, ndim=1] equals, periodicity_axis = None
    if translation is not None:
        add_to_vectors(new_positions, translation)
        #center += translation
    if not pbc[0] and not pbc[1] and not pbc[2]:
        equals = get_equals(center, positions, new_positions, atomic_numbers, error_tolerance, debug)
    else:
        equals = get_equals_periodic(positions, new_positions, cell, pbc, error_tolerance, debug) # , periodicity_axis
    cdef DTYPE_t on = order_no
    # n = 2 is equal with inversion operator
    if n == 2 or (float(order_no+1) / float(n))  == 0.5:
        return None
    if equals is not None:
        operation_name = "S%i" % n
        while order_no > 0:
            operation_name += "'"
            order_no -= 1
        if translation is not None:
            return SymmetryOperation(operation_name, equals, None,  molecule_change_matrix_axis = periodicity_axis, vector = vector.base, center = center.base, magnitude = n, type='S', order_no=on, rotation_matrix = improper_rotation_matrix(vector.base, angle), translation_vector = translation.base)
        else:
            return SymmetryOperation(operation_name, equals, None,  molecule_change_matrix_axis = periodicity_axis, vector = vector.base, center = center.base, magnitude = n, type='S', order_no=on, rotation_matrix = improper_rotation_matrix(vector.base, angle))

cdef list check_improper_rotations(np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim=2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers,  list rotation_operations, float error_tolerance, bint debug):
    """
        Improper rotational axes must be coincident with existing 
        proper rotational axes and have a degree either equal to
        or twice the degree of coincident proper rotational axis
    """
    cdef list result = []
    cdef float L, angle
    cdef DTYPE_t n, order_no
    cdef np.ndarray[np.float_t, ndim=1] vector
    cdef int i, N = len(rotation_operations)
    cdef SymmetryOperation operation, result_operation

    for i from 0 <= i < N:
        operation = <SymmetryOperation>rotation_operations[i]
        vector = <np.ndarray[np.float_t, ndim=1]> operation.vector.copy()
        L = get_length(vector)
        vector /= L
        angle = pi2/operation.magnitude
        n = operation.magnitude
        center = operation.center

        # one times the proper rotation
        for order_no from 0 <= order_no < n-1:
            result_operation = check_improper_rotation(positions, cell, pbc, atomic_numbers, vector, (order_no+1)*angle, center, n, order_no, error_tolerance, debug, None)
            if result_operation is not None:
                result.append(result_operation)
            result_operation = None
        
        # rotations twice the proper rotation
        angle = angle/2
        for order_no from 0 <= order_no < (2*n-1):
            result_operation = check_improper_rotation(positions, cell, pbc, atomic_numbers, vector, (order_no+1)*angle, center, 2*n, order_no, error_tolerance, debug, None)
            if result_operation is not None:
                result.append(result_operation)
            result_operation = None

    result = remove_equals(result, debug)
    print "%i improper rotations found" % len(result)

    return result

cdef SymmetryOperation check_single_rotation(np.float_t[:, ::1] positions, np.float_t[:, ::1] cell, DTYPE2_t[::1] pbc, DTYPE2_t[::1] atomic_numbers, np.float_t[::1] vector, float angle, np.float_t[::1] center, DTYPE_t n, DTYPE_t order_no, float error_tolerance, bint debug, np.float_t[::1] translation):
    cdef np.float_t[:, ::1] new_positions = rotate(positions, vector, angle, center)
    cdef np.ndarray[DTYPE2_t, ndim=1] equals, periodicity_axis = None
    cdef SymmetryOperation result_operation = None
    if translation is not None:
        add_to_vectors(new_positions, translation)

        #center += translation
    if not pbc[0] and not pbc[1] and not pbc[2]:
        equals = get_equals(center, positions, new_positions, atomic_numbers, error_tolerance, debug)
    else:
        equals = get_equals_periodic(positions, new_positions, cell, pbc, error_tolerance, debug) #, periodicity_axis
    cdef DTYPE_t on = order_no
    if equals is not None:
        operation_name = "C%i" % n
        while order_no > 0:
            operation_name += "'"
            order_no -= 1
        if translation is not None:
            result_operation = SymmetryOperation(operation_name, equals, None, vector = vector.base, center = center.base, magnitude = n, type='C', order_no=on, rotation_matrix = rotation_matrix(vector, angle), translation_vector = translation.base) # , molecule_change_matrix_axis = periodicity_axis
        else:
            result_operation = SymmetryOperation(operation_name, equals, None, vector = vector.base, center = center.base, magnitude = n, type='C', order_no=on, rotation_matrix = rotation_matrix(vector, angle)) # , molecule_change_matrix_axis = periodicity_axis
    return result_operation

cdef list check_rotation(np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim=2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers, np.ndarray[np.float_t, ndim=1] vector, np.ndarray[np.float_t, ndim=1] center, float error_tolerance, bint debug, DTYPE_t n):

    # normalize vector
    cdef float L = get_length(vector)
    vector /= L
    if debug:
        print vector
    cdef list result = []
    cdef DTYPE_t order_no 
    cdef float angle
    cdef SymmetryOperation result_operation = None
    while n >= 2:   
        angle = pi2/n
        for order_no from 0 <= order_no < n-1:
            
            result_operation = check_single_rotation(positions, cell, pbc, atomic_numbers, vector, (order_no+1)*angle, center, n, order_no, error_tolerance, debug, None)
            if result_operation is not None:
                result.append(result_operation)
            result_operation = None
        n -= 1
    
    return result
     

cdef tuple  find_symmetry_operations(atoms, float error_tolerance):
    
    cdef bint debug = False
    cdef list result
    cdef np.ndarray[np.float_t, ndim = 2] cell = np.array(atoms.get_cell(), order = 'C'), positions = np.array(atoms.get_positions(), order = 'C')
    cdef np.ndarray[DTYPE2_t, ndim = 1] pbc = np.zeros(3, dtype=DTYPE2, order = 'C'), atomic_numbers = np.array(atoms.get_atomic_numbers(), dtype=DTYPE2, order = 'C')
    for i, periodic in enumerate(atoms.get_pbc()):
        if periodic:
            pbc[i] = 1
    cdef np.ndarray[np.float_t, ndim = 1] com = np.array(atoms.get_center_of_mass(), order = 'C')
    if error_tolerance is None:
        error_tolerance = 0.05
    if all(atoms.get_pbc()): 
        result, primitive, translations = get_periodic_symmetry_operations(atoms, error_tolerance, debug)
        return result, primitive, translations
    elif pbc.any():
        result = []
        translations = get_pure_translations(atoms, error_tolerance, debug)
        unique_atoms, equivalent_atoms = get_unique_atoms(positions, translations, error_tolerance)
        mirrors = check_mirrors(unique_atoms, positions, cell, pbc, atomic_numbers, com, None, None, error_tolerance, debug)
        horizontal_mirror = get_slab_horizontal_mirror(positions, cell, pbc, atomic_numbers, com, error_tolerance, debug)
        if horizontal_mirror is not None:
            print "Horizontal mirror found"
        rotations = check_slab_rotational_axes(unique_atoms, positions, cell, pbc, atomic_numbers, com, error_tolerance, debug)
        improper_rotations = check_improper_rotations(positions, cell, pbc, atomic_numbers, rotations, error_tolerance, debug)
        result.extend(mirrors)
        result.extend(rotations)
        result.extend(improper_rotations)
        if horizontal_mirror is not None:
            result.append(horizontal_mirror)
            slabha = get_slab_horizontal_rotation_axis(unique_atoms, positions, cell, pbc, atomic_numbers, com, horizontal_mirror.vector.copy(), error_tolerance, debug)
            result.extend(slabha)
        
        result.append(get_identity_operator(positions.shape[0]))
        
        result = final_symmetry_operations_from_symmetry_operations_and_pure_translations(positions, cell, pbc, atomic_numbers, result, translations, error_tolerance, debug)
        print "%i Symmetry operations" % len(result)
        return result, None, None
    else: 
        print "Error Tolerance", error_tolerance
        error_tolerance= 0.6
        moms, vecs = check_symmetry(atoms)
        result = []
        rotations = check_proper_rotational_axes(positions, cell, pbc, atomic_numbers, com, vecs, error_tolerance, debug)
        improper_rotations = check_improper_rotations(positions, cell, pbc, atomic_numbers, rotations, error_tolerance, debug)
        mirrors = check_mirrors(None, positions, cell, pbc, atomic_numbers, com, rotations, vecs, error_tolerance, debug)
        inversion = check_inversion_center(positions, cell, pbc, atomic_numbers, com, error_tolerance, debug)
        if inversion is not None:
            result.append(inversion)
        result.extend(rotations)
        result.extend(improper_rotations)
        result.extend(mirrors)
        result.append(get_identity_operator(positions.shape[0]))
        return result, None, None

cdef SymmetryOperation get_identity_operator(int atom_count):
    cdef np.ndarray[DTYPE2_t, ndim=1] equals = np.arange(0, atom_count, 1, dtype=DTYPE2)
    return SymmetryOperation("Identity", equals, None, type='E')

cdef list check_slab_rotational_axes(list unique_atoms, np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim=2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers, np.ndarray[np.float_t, ndim=1] com, float error_tolerance, bint debug):
    cdef list result = []
    cdef np.ndarray[np.float_t, ndim=2] in_plane_coordinates = get_in_plane_coordinates(unique_atoms, positions[0], positions, cell, pbc) 
    result.extend(remove_equals(check_slab_atom_rotational_axes(positions, cell, pbc, atomic_numbers, com, in_plane_coordinates, error_tolerance, debug), debug))
    result.extend(remove_equals(check_slab_atom_mid_rotational_axes(positions, cell, pbc, atomic_numbers, com, in_plane_coordinates, error_tolerance, debug), debug))
    result.extend(remove_equals(check_slab_three_atom_mid_rotational_axes(positions, cell, pbc, atomic_numbers, com, in_plane_coordinates, error_tolerance, debug), debug))
    
    print "%i proper rotations found" % len(result)
    return result

cdef list check_proper_rotational_axes(np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim=2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers, np.ndarray[np.float_t, ndim = 1] center_of_mass, np.ndarray[np.float_t, ndim=2] vectors_of_inertia, float error_tolerance, bint debug):
    cdef list result = []
    result.extend(remove_equals(check_rotations_along_vectors(positions, cell, pbc, atomic_numbers, center_of_mass, vectors_of_inertia, error_tolerance, debug), debug))
    result.extend(remove_equals(check_com_atom_rotational_axes(positions, cell, pbc, atomic_numbers, center_of_mass, error_tolerance, debug), debug))
    result = remove_equals(result, debug)
    result.extend(remove_equals(check_com_atom_mid_rotational_axes(positions, cell, pbc, atomic_numbers, center_of_mass, error_tolerance, debug), debug))
    result = remove_equals(result, debug)
    result.extend(remove_equals(check_rotations_along_vectors(positions, cell, pbc, atomic_numbers, center_of_mass, vectors_of_inertia, error_tolerance, debug), debug))
    result = remove_equals(result, debug)
    result.extend(remove_equals(check_rotations_along_symmetry_operations(positions, cell, pbc, atomic_numbers, center_of_mass, result, 2, error_tolerance, debug), debug)) 
    result = remove_equals(result, debug)
    result.extend(remove_equals(check_rotations_along_cross_products_of_symmetry_operations(positions, cell, pbc, atomic_numbers, center_of_mass, result, -1, error_tolerance, debug), debug))
    result = remove_equals(result, debug)
    
    print "%i proper rotations found" % len(result)
    return result

   
cdef list check_com_atom_rotational_axes(np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim=2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers, np.ndarray[np.float_t, ndim=1] com, float error_tolerance, bint debug):
    cdef list result = []
    cdef DTYPE_t N = positions.shape[0]
    cdef DTYPE_t i
    for i from 0 <= i < N:
        result.extend(check_rotation(positions, cell, pbc, atomic_numbers, positions[i]-com, com, error_tolerance, debug, 8))
    return result 

cdef list check_slab_atom_rotational_axes(np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim=2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers, np.ndarray[np.float_t, ndim=1] com, np.ndarray[np.float_t, ndim=2] in_plane_coordinates, float error_tolerance, bint debug):
    cdef np.ndarray[np.float_t, ndim = 1] normal_vector = np.zeros(3, dtype=float)
    cdef int i, non_periodic
    cdef bint periodic
    for i, periodic in enumerate(pbc):
        if not periodic:
            normal_vector[i] = 1.0
            non_periodic = i
    cdef list result = []
    cdef DTYPE_t N = in_plane_coordinates.shape[0]
    for i from 0 <= i < N:
        center = in_plane_coordinates[i]
        center[non_periodic] = com[non_periodic]
        result.extend(check_rotation(positions, cell, pbc, atomic_numbers, normal_vector, center, error_tolerance, debug, 6))
    print "Found %i atom-atom rotational axes" % len(result)
    return result

cdef list check_slab_atom_mid_rotational_axes(np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim=2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers, np.ndarray[np.float_t, ndim=1] com, np.ndarray[np.float_t, ndim=2] in_plane_coordinates,  float error_tolerance, bint debug):
    cdef np.ndarray[np.float_t, ndim = 1] normal_vector = np.zeros(3, dtype=float), center
    cdef int i, j, non_periodic
    cdef bint periodic
    for i, periodic in enumerate(pbc):
        if not periodic:
            normal_vector[i] = 1.0
            non_periodic = i
    cdef list result = []
    cdef DTYPE_t N = in_plane_coordinates.shape[0]
    for i from 0 <= i < N:
        for j from 0 <= j < N:
            if i < j:
                center = (in_plane_coordinates[i]+in_plane_coordinates[j])/2.0
                center[non_periodic] = com[non_periodic]
                result.extend(check_rotation(positions, cell, pbc, atomic_numbers, normal_vector, center, error_tolerance, debug, 6))
    print "Found %i atom-atom mid rotational axes" % len(result)
    return result

cdef list check_slab_three_atom_mid_rotational_axes(np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim=2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers, np.ndarray[np.float_t, ndim=1] com, np.ndarray[np.float_t, ndim=2] in_plane_coordinates,  float error_tolerance, bint debug):
    cdef np.ndarray[np.float_t, ndim = 1] normal_vector = np.zeros(3, dtype=float), center
    cdef int i, j, k, non_periodic
    cdef bint periodic
    for i, periodic in enumerate(pbc):
        if not periodic:
            normal_vector[i] = 1.0
            non_periodic = i
    cdef list result = []
    cdef DTYPE_t N = in_plane_coordinates.shape[0]
    for i from 0 <= i < N:
        for j from 0 <= j < N:
            if i < j:
                for k from 0 <= k < N:
                    if k < j:
                        center = (in_plane_coordinates[i]+in_plane_coordinates[j]+in_plane_coordinates[k])/3.0
                        center[non_periodic] = com[non_periodic]
                        result.extend(check_rotation(positions, cell, pbc, atomic_numbers, normal_vector, center, error_tolerance, debug, 6))
    return result

cdef list check_com_atom_mid_rotational_axes(np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim=2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers, np.ndarray[np.float_t, ndim=1] com,  float error_tolerance, bint debug):
    cdef list result = [], add
    cdef np.ndarray[np.float_t, ndim=1] position_i, position_j
    cdef DTYPE_t i, j, atomic_number_i, atomic_number_j
    cdef N = positions.shape[0]
    for i from 0 <= i < N:
        atomic_number_i = atomic_numbers[i]
        position_i = positions[i]
        for j from 0 <= j < N:
            atomic_number_j = atomic_numbers[j]
            position_j = positions[j]
            if atomic_number_i == atomic_number_j:
                p = (position_i + position_j)/2.0
                add = check_rotation(positions, cell, pbc, atomic_numbers, p-com, com, error_tolerance, debug, 8)
                result.extend(add)

                
    return result

cdef public void print_symmetry_operations(list symmetry_operations):
    cdef int i, N = len(symmetry_operations)
    cdef SymmetryOperation symmetry_operation
    for i from 0 <= i < N:
        symmetry_operation = symmetry_operations[i]
        print symmetry_operation

cdef list check_rotations_along_symmetry_operations(np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim=2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers, np.ndarray[np.float_t, ndim=1] com, list operations, DTYPE2_t only_magnitude, float error_tolerance, bint debug):
    cdef list result = []
    cdef DTYPE_t i
    cdef DTYPE_t N = len(operations)
    cdef SymmetryOperation operation
    cdef np.ndarray[np.float_t, ndim=1] operation_vector
    cdef DTYPE2_t operation_magnitude
    
    for i from 0 <= i < N:
        operation = <SymmetryOperation> operations[i]
        operation_vector = <np.ndarray[np.float_t, ndim=1]> operation.vector
        operation_magnitude = operation.magnitude
        if only_magnitude == -1 or operation_magnitude == only_magnitude:
            result.extend(check_rotation(positions, cell, pbc, atomic_numbers, operation_vector, com, error_tolerance, debug, 8))
    return result

cdef list check_rotations_along_cross_products_of_symmetry_operations(np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim=2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers, np.ndarray[np.float_t, ndim=1] com, list operations, DTYPE2_t  only_magnitude, float error_tolerance, bint debug):
    cdef list result = []
    cdef DTYPE_t i, j
    cdef DTYPE_t N = len(operations)
    cdef SymmetryOperation operation, operation2
    cdef DTYPE2_t operation_magnitude, operation_magnitude2
    cdef np.ndarray[np.float_t, ndim=1] operation_vector, operation_vector2, vector
    for i from 0 <= i < N:
        operation = <SymmetryOperation> operations[i]
        operation_magnitude = operation.magnitude
        operation_vector = <np.ndarray[np.float_t, ndim=1]>operation.vector
        for j from 0 <= j < N:
            operation2 = operations[j]
            operation_magnitude2 = operation2.magnitude
            operation_vector2 = operation2.vector
            if i != j and (only_magnitude == -1 or (operation_magnitude2 == only_magnitude and operation_magnitude2 == only_magnitude)):
                vector = np.cross(operation_vector, operation_vector2)
                result.extend(check_rotation(positions, cell, pbc, atomic_numbers, vector, com, error_tolerance, debug, 8))
    return result

cdef list check_rotations_along_vectors(np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim=2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers, np.ndarray[np.float_t, ndim=1] com, np.ndarray[np.float_t, ndim=2] vectors, float error_tolerance, bint debug):
    cdef list result = []
    cdef np.ndarray[np.float_t, ndim=1] vector
    cdef DTYPE_t i, N = len(vectors)
    for i from 0 <= i < N:
        vector = <np.ndarray[np.float_t, ndim=1]>vectors[i]
        result.extend(check_rotation(positions, cell, pbc, atomic_numbers, vector, com, error_tolerance, debug, 8))
    return result

cdef SymmetryOperation check_mirror( np.float_t[:, ::1] positions, np.float_t[:, ::1] cell, DTYPE2_t[::1] pbc, DTYPE2_t[::1] atomic_numbers,  np.float_t[::1] normal_vector, np.float_t[::1] center, bint debug, float error_tolerance, np.float_t[::1] translation):
    # normalize vector
    cdef float L = get_length(normal_vector)
    if L < 0.01:
        return None
    divide(normal_vector, L)
    cdef np.float_t[:, ::1] mirrored = mirror_through_plane(positions.copy(), normal_vector, center, debug)
    if debug:
        print normal_vector
        print positions
        print mirrored
        from ice_package.help_methods import get_oxygens
        import ase
        ase.visualize.view(get_oxygens(mirrored))
        #ase.visualize.view(atoms)
        print normal_vector.base
        print center.base
        raw_input("Continue")
    cdef DTYPE2_t[::1] result #, periodicity_axis = None
    if pbc[0] or pbc[1] or pbc[2]:
        if translation is not None:
            add_to_vectors(mirrored, translation) #np.dot(translation, cell)
        result = get_equals_periodic(positions, mirrored, cell, pbc, error_tolerance, debug) # , periodicity_axis
    else:
        result = get_equals(center, positions, mirrored, atomic_numbers, error_tolerance, debug)
  
    if result is None:
        return None
    else:
        if translation is None:
            return SymmetryOperation('sigma', result.base, None, vector = normal_vector.base, magnitude = 1, center = center.base, order_no = 0, type='sigma', translation_vector = None) # , molecule_change_matrix_axis = periodicity_axis
        else:
            return SymmetryOperation('sigma', result.base, None, vector = normal_vector.base, magnitude = 1, center = center.base, order_no = 0, type='sigma', translation_vector = translation.base)

cdef list check_mirrors(list unique_atoms, np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim = 2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers, np.ndarray[np.float_t, ndim = 1] com,  list rotation_operations, np.ndarray[np.float_t, ndim=2] inertia_axis, float error_tolerance, bint debug):
    cdef list result = []
    cdef np.ndarray[np.float_t, ndim = 2] in_plane_coordinates
    if pbc.any(): 
        in_plane_coordinates = get_in_plane_coordinates(unique_atoms, positions[0], positions, cell, pbc)     
        result.extend(check_atom_atom_mirrors(positions, cell, pbc, atomic_numbers, in_plane_coordinates, error_tolerance, debug))      
        result.extend(check_atom_atom_mid_mirrors(positions, cell, pbc, atomic_numbers, in_plane_coordinates, error_tolerance, debug))
        result.extend(check_atom_mid_atom_mid_mirrors(positions, cell, pbc, atomic_numbers, in_plane_coordinates, error_tolerance, debug))
    else:
        result.extend(check_mirrors_from_previously_found(positions, cell, pbc, atomic_numbers, com, rotation_operations, inertia_axis,  debug, error_tolerance))
        result.extend(check_com_atom_mirrors(positions, cell, pbc, atomic_numbers, com, error_tolerance, debug))
        result.extend(check_com_atom_mid_mirrors(positions, cell, pbc, atomic_numbers, com, error_tolerance, debug ))

    
    result = remove_equals(result, debug)
    print "%i mirrors found" % len(result)
    return result
    
    

cdef np.ndarray[np.float_t, ndim = 2] get_in_plane_coordinates(list unique_atoms, np.ndarray[np.float_t, ndim = 1] atom_position, np.ndarray[np.float_t, ndim = 2] atom_positions, np.ndarray[np.float_t, ndim = 2] cell, np.ndarray[DTYPE2_t] pbc):
    """
        This method is intended for slabs
    """
    cdef int i, j, N = atom_positions.shape[0], non_periodic
    cdef list result = []
    cdef np.ndarray[np.float_t, ndim = 1] slab_periodic_vector
    cdef np.ndarray[np.float_t, ndim = 2] slab_periodic_vectors
    cdef bint periodic
    for i, periodic in enumerate(pbc):
        if not periodic:
            non_periodic = i
    slab_periodic_vectors = get_slab_periodic_vectors(non_periodic)
    
    for i from 0 <= i < 3:
        if i == non_periodic:
            for j in range(N):
                if atom_position[i] == atom_positions[j, i]:
                    result.append(atom_positions[j])
                    
    
    for k in range(slab_periodic_vectors.shape[0]):
        slab_periodic_vector = slab_periodic_vectors[k]
        result.append(atom_position+np.dot(slab_periodic_vector, cell))
    cdef np.ndarray[np.float_t, ndim = 2] result_np = np.array(result, dtype=float)
    return result_np

cdef SymmetryOperation get_slab_horizontal_mirror( np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim = 2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers, np.ndarray[np.float_t, ndim = 1] com,  float error_tolerance, bint debug):
    """
        This method is intended for slabs
    """
    cdef np.ndarray[np.float_t, ndim = 1]  normal_vector = np.zeros(3, dtype=float)
    cdef int i
    cdef bint periodic
    for i, periodic in enumerate(pbc):
        if not periodic:
            normal_vector[i] = 1.0
    cdef SymmetryOperation result = check_mirror(positions, cell, pbc, atomic_numbers, normal_vector, com, debug, error_tolerance, None)
    return result

cdef list get_slab_horizontal_rotation_axis(list unique_atoms, np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim=2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers, np.ndarray[np.float_t, ndim=1] com, np.ndarray[np.float_t, ndim=1] free_axis, float error_tolerance, bint debug):
    """
        This method is intended for slabs
    """
    cdef np.ndarray[np.float_t, ndim = 1] vector, mid_point, slab_periodic_vector, pos 
    cdef np.ndarray[np.float_t, ndim = 2]  rotation_points = np.zeros((0,3), dtype=float), slab_periodic_vectors
    cdef int i, j, k, N = positions.shape[0], M, non_periodic 
    cdef float tolerance = 0.001, vector_length
    cdef bint periodic
    cdef list result = []
    for i, periodic in enumerate(pbc):
        if not periodic:
            non_periodic = i
    slab_periodic_vectors = get_slab_periodic_vectors(non_periodic)
    for i in unique_atoms:
        pos = positions.copy()[i]
        pos[non_periodic] = com[non_periodic]
        rotation_points = np.vstack((rotation_points, pos))
        for k in range(slab_periodic_vectors.shape[0]):
            slab_periodic_vector = slab_periodic_vectors[k]
            rotation_points = np.vstack((rotation_points, pos + np.dot(slab_periodic_vector, cell)))
        for j in unique_atoms:
            if i < j:
                vector = positions[i] - positions[j]
                mid_point =  (positions[i] + positions[j]) / 2.0
                vector /= np.linalg.norm(vector)
                #dotproduct = np.dot(vector, free_axis)
                if mid_point[non_periodic] >= com[non_periodic] - tolerance and mid_point[non_periodic] <= com[non_periodic] + tolerance:
                    rotation_points = np.vstack((rotation_points, mid_point))
                    for k in range(slab_periodic_vectors.shape[0]):
                        slab_periodic_vector = slab_periodic_vectors[k]
                        rotation_points = np.vstack((rotation_points, mid_point + np.dot(slab_periodic_vector, cell)))
        
    s= time()
    M = rotation_points.shape[0]
    for i in range(M):
        for j in range(M):
            if i < j:
                vector = rotation_points[i] - rotation_points[j] 
                vector_length = np.linalg.norm(vector)
                if vector_length > 0.02: # just checking that points are not the same
                    vector /= vector_length
                    result.extend(check_rotation(positions, cell, pbc, atomic_numbers, vector, rotation_points[i], error_tolerance, debug, 2))
    print "Time used in checking %i rotation possibilities: %f s" % (M*M, time()-s) 
    result = remove_equals(result, debug)
    return result

cdef np.ndarray[np.float_t, ndim = 2] get_slab_periodic_vectors(DTYPE2_t non_periodic_axis):
    cdef np.ndarray[np.float_t, ndim = 1] result_vector = np.zeros(3)
    cdef np.ndarray[np.float_t, ndim = 2] result = np.zeros((0, 3))
    cdef int i, j, k
    for i in range(3):
        if i - 1 == 0 or non_periodic_axis != 0:
            result_vector[0] = i -1
            for j in range(3):
                if j - 1 == 0 or non_periodic_axis != 1:
                    result_vector[1] = j - 1
                    for k in range(3):
                        if k - 1 == 0 or non_periodic_axis != 2:
                            result_vector[2] = k - 1
                            if abs(sum(result_vector)) > 0.1: 
                                result = np.vstack((result, result_vector))
    return result 
    

cdef list  check_mirrors_from_previously_found( np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim = 2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers, np.ndarray[np.float_t, ndim = 1] com,  list rotation_operations, np.ndarray[np.float_t, ndim=2] inertia_axis, bint debug, float error_tolerance):
    cdef list result = []
    cdef SymmetryOperation rotation_operation, mirror
    cdef DTYPE_t i, j, N = len(rotation_operations), M = inertia_axis.shape[0]
    cdef np.ndarray[np.float_t, ndim=1] vector, rot_vector, axis
    for i from 0 <= i < N:
        rotation_operation = <SymmetryOperation> rotation_operations[i]
        rot_vector = <np.ndarray[np.float_t, ndim=1]> rotation_operation.vector
        mirror = check_mirror(positions, cell, pbc, atomic_numbers, rot_vector, com, debug, error_tolerance, None)
        if mirror != None:
            result.append(mirror)
        for j from 0 <= j < M:
            axis = inertia_axis[j]
            vector = np.cross(rot_vector, axis)
            if np.any(vector != np.zeros_like(vector)):
                mirror = check_mirror(positions, cell, pbc, atomic_numbers, vector, com, debug, error_tolerance, None)
                if mirror is not None:
                    result.append(mirror)
    return result

cdef list check_com_atom_mirrors( np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim = 2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers, np.ndarray[np.float_t, ndim = 1] com,  float error_tolerance, bint debug):
    cdef list result = []
    cdef np.ndarray[np.float_t, ndim=1] position
    cdef DTYPE_t i, N = positions.shape[0]
    for i from 0 <= i < N:
        position = positions[i]
        if any(position != com): 
            mirror = check_mirror(positions, cell, pbc, atomic_numbers, position-com, com, debug, error_tolerance, None)
            if mirror != None:
                result.append(mirror)
    return result

cdef list check_atom_atom_mirrors( np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim = 2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers,  np.ndarray[np.float_t, ndim = 2] in_plane_atom_positions, float error_tolerance, bint debug):
    cdef list result = []
    cdef DTYPE_t i, j, N = in_plane_atom_positions.shape[0], non_periodic 
    cdef np.ndarray[np.float_t, ndim = 1] position, position_2
    cdef SymmetryOperation mirror
    s = time()
    for i from 0 <= i < N:
        position = in_plane_atom_positions[i]
        for j from 0 <= j < N:
            if j > i:
                position_2 = in_plane_atom_positions[j]
                mirror = check_mirror(positions, cell, pbc, atomic_numbers, position-position_2, position,  debug, error_tolerance, None)
                if mirror is not None:
                    result.append(mirror)
    print "Time used in checking %i mirror possibilities: %f s" % (N*N, time()-s) 
    return result

cdef list check_atom_atom_mid_mirrors( np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim = 2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers,  np.ndarray[np.float_t, ndim = 2] in_plane_atom_positions, float error_tolerance, bint debug):
    cdef list result = []
    cdef DTYPE_t i, j, k, l, N = len(in_plane_atom_positions)
    cdef np.ndarray[np.float_t, ndim = 1] position, position_2, position_3
    s = time()
    for i from 0 <= i < N:
        position = in_plane_atom_positions[i]
        for j from 0 <= j < N:
            if j != i:
                position_2 = in_plane_atom_positions[j]
                for k from 0 <= k < N:
                    if k > j and k != i:
                        position_3 = in_plane_atom_positions[k]
                        atom_mid = (position_2 + position_3) / 2.0
                        mirror = check_mirror(positions, cell, pbc, atomic_numbers, position-atom_mid, position, debug, error_tolerance, None)
                        if mirror is not None:
                            result.append(mirror)
    print "Time used in checking %i mirror possibilities: %f s" % (N*N*N, time()-s) 
    return result

cdef list check_atom_mid_atom_mid_mirrors( np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim = 2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers,  np.ndarray[np.float_t, ndim = 2] in_plane_atom_positions, float error_tolerance, bint debug):
    cdef list result = []
    cdef DTYPE_t i, j, k, l, N = len(in_plane_atom_positions)
    cdef np.ndarray[np.float_t, ndim = 1] position, position_2, position_3, position_4, atom_mid, atom_mid_2
    s = time()
    for i from 0 <= i < N:
        position = in_plane_atom_positions[i]
        for j from 0 <= j < N:
            if j > i:
                position_2 = in_plane_atom_positions[j]
                atom_mid = (position + position_2) / 2.0
                for k from 0 <= k < N:
                    if k != j and k > i:
                        position_3 = in_plane_atom_positions[k]
                        for l from 0 <= l < N:
                            if l != i and l != j and l > k:
                                position_4 = in_plane_atom_positions[k]
                                atom_mid_2 = (position_3 + position_4) / 2.0
                                mirror = check_mirror(positions, cell, pbc, atomic_numbers, atom_mid-atom_mid_2, atom_mid, debug, error_tolerance, None)
                                if mirror is not None:
                                    result.append(mirror)
    print "Time used in checking %i mirror possibilities: %f s" % (N*N*N, time()-s) 
    return result

cdef list check_com_atom_mid_mirrors( np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim = 2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers, np.ndarray[np.float_t, ndim = 1] com,  float error_tolerance, bint debug):
    cdef list result = []
    cdef np.ndarray[np.float_t, ndim=1] p, position1, position2
    cdef DTYPE_t atomic_number1, atomic_number2
    cdef SymmetryOperation mirror
    cdef int i, j
    for i in xrange(positions.shape[0]):
        position1 = <np.ndarray[np.float_t]>positions[i]
        atomic_number1 = atomic_numbers[i]
        for j in xrange(positions.shape[0]):
            position2 = <np.ndarray[np.float_t]>positions[j]
            atomic_number2 = atomic_numbers[j]
            if atomic_number1 == atomic_number2:
                p = (position1 + position2)/2
                if any(p != com): 
                    if debug:
                        print com[2]
                        print p[2]
                        print position1[2]
                    mirror = check_mirror(positions, cell, pbc, atomic_numbers, p-com, com, debug, error_tolerance, None)
                    if mirror is not None:
                        result.append(mirror)
    return result


def check_symmetry(atoms):
    moi, vectors = get_moments_of_inertia(atoms)
    if moi[0] == moi[1] and moi[0] == moi[2]:
        return Degeneracy.SPHERICAL, vectors
    elif moi[0] == moi[1]:
        return Degeneracy.SYMMETRIC, vectors
    elif moi[0] == moi[2]:
        return Degeneracy.SYMMETRIC, vectors
    elif moi[1] == moi[2]:
        return Degeneracy.SYMMETRIC, vectors
    else:
        return Degeneracy.ASYMMETRIC, vectors


def get_moments_of_inertia(atoms):
    '''Get the moments of inertia
   
    ******         
    copied from ASE to get the vectors also
    ******

    The three principal moments of inertia are computed from the
    eigenvalues of the inertial tensor. periodic boundary
    conditions are ignored. Units of the moments of inertia are
    amu*angstrom**2.
    '''
    cdef np.float_t[:, ::1] positions = positions_related_to_center_of_mass(atoms.get_center_of_mass(), atoms.get_positions())
    cdef np.ndarray[np.float_t, ndim=1] masses = atoms.get_masses()
    cdef np.float_t x, y, z, m, I11, I22, I33, I12, I13, I23
    #initialize elements of the inertial tensor
    I11 = I22 = I33 = I12 = I13 = I23 = 0.0
    cdef np.uint8_t i, N = positions.shape[0]
    
    for i from 0 <= i < N:
        x = positions[i, 0]
        y = positions[i, 1]
        z = positions[i, 2]
        m = masses[i]
        I11 += m * (y**2 + z**2)
        I22 += m * (x**2 + z**2)
        I33 += m * (x**2 + y**2)
        I12 += -m * x * y
        I13 += -m * x * z
        I23 += -m * y * z

    cdef np.ndarray[np.float_t, ndim=2] I = np.array([[I11, I12, I13],
                                                      [I12, I22, I23],
                                                      [I13, I23, I33]], order = 'C')

    evals, evecs = np.linalg.eig(I)
    evals = np.ascontiguousarray(evals)
    evecs = np.ascontiguousarray(evecs)
    return evals, evecs

## PYSPGLIB

def get_periodic_symmetry_group(atoms, error_tolerance = 0.01, debug = False):
    from pyspglib import spglib    
    dataset = spglib.get_symmetry_dataset(atoms, symprec=error_tolerance)
    if debug:
        print atoms
        print dataset['equivalent_atoms']
        print dataset['international']
        print dataset['hall']
        print dataset['wyckoffs']
        print dataset['transformation_matrix']
        print "Number of symmetry operations %i"  % len(dataset['rotations'])
    return dataset['international']

"""
cdef list get_periodic_symmetry_operations(atoms, error_tolerance, debug):
    from ase.utils import irotate
    from ase.visualize import view
    from pyspglib import spglib
    #error_tolerance = 0.2
    used_numbers = []
    #symmetry = spglib.get_symmetry(atoms, symprec=1e-5)
    
    dataset = spglib.get_symmetry_dataset(atoms, symprec=1e-5)
    symmetry = spglib.get_symmetry(atoms, symprec=1e-5)
    cdef list result = []
    cdef DTYPE_t i, l, M, N = dataset['rotations'].shape[0]
    cdef np.ndarray[np.float_t, ndim=2]  new_pos
    cdef np.ndarray[np.float_t, ndim=1] pos
    cdef np.ndarray[DTYPE2_t, ndim=1] equals
    cdef SymmetryOperation so
    cdef np.ndarray[DTYPE2_t, ndim=1] identity = np.arange(len(atoms), dtype=DTYPE2)
    if debug:
        cell, scaled_positions, numbers = spglib.find_primitive(atoms, symprec=1e-5)
        #print cell, scaled_positions, numbers
        #lattice, scaled_positions, numbers = spglib.refine_cell(atoms, symprec=1e-5)
        #print lattice, scaled_positions, numbers
        a = Atoms( symbols='O4',
                 cell=cell,
                 scaled_positions=scaled_positions,
                 pbc=True)
        #symmetry = spglib.get_symmetry(a, symprec=1e-2)
        
        #dataset = spglib.get_symmetry_dataset(a, symprec=1e-2)
        #N = dataset['rotations'].shape[0]
        if debug:
            print dataset['number']
            print dataset['equivalent_atoms']
            print dataset['international']
            print dataset['hall']
            print dataset['wyckoffs']
            print dataset['transformation_matrix']
    print "Number of symmetry operations %i"  % len(dataset['rotations'])
    for i from 0 <= i < N:
        new_atoms = atoms.copy()
        test = atoms.copy()
        
        rot = dataset['rotations'][i]
        trans = dataset['translations'][i]
        if debug:
            x, y, z = irotate(rot)
            #print x, y, z
            print "------------------- %i -----------------------" % i
            print "Rotation" 
            print rot
            print "Translation" 
            print trans
        new_pos =  new_atoms.get_scaled_positions()
        M = new_pos.shape[0]
        
        print trans
        print rot
        for l from 0 <= l < M:
            pos = new_pos[l]
            #print new_pos[l]
            new_pos[l] = (np.dot(rot, pos))
            new_pos[l] += trans
            #print new_pos[l]
        new_atoms.set_scaled_positions(new_pos)
        equals = get_equals_periodic(atoms, new_atoms, error_tolerance, debug)
        if equals is not None:
            so = SymmetryOperation(str(i), equals, None, vector = None, magnitude = 1, rotation_matrix=np.array(rot, dtype=float), translation_vector=np.array(trans, dtype=float))
            if np.all(equals == identity):
                so.type = 'E'
            #if debug:
            #    print so
            result.append(so)
            new = equals[0]
            counter = 0
            this = []
            while new != 0:                
                new = equals[new]
                this.append(new)
                if new not in used_numbers:
                    used_numbers.append(new)
                counter += 1
            print "----------"
            print this
            print used_numbers
            print "Smallest subgroup containing 0, length: %i" % counter
        else:
            print "Equivalent not found"
            view(new_atoms)
        
        print equals
        raw_input()

    return result, None, None
"""

def is_multiple_of_primitive_cell(cell1, cell2):
    multiplier = cell1[0][0]/cell2[0][0]
    cell3 = cell1 / multiplier
    for i in range(3):
        for j in range(3):
            if abs(cell3[i][j] - cell2[i][j]) > 0.01:
                return False
    return True

cdef tuple get_periodic_symmetry_operations(atoms, error_tolerance, debug):
    from ase.utils import irotate
    from ase.visualize import view
    from pyspglib import spglib
    
    primitive_cell, scaled_positions, numbers = spglib.find_primitive(atoms, symprec=1e-5)
    
    cdef list primitive_symmetry_operations = None, translation_operations = None
    cdef list result = []
    cdef np.ndarray[np.float_t, ndim = 2] translations, primitive_translations
    cdef np.ndarray[np.float_t, ndim = 3] primitive_rotations
    cdef int i, j
    if primitive_cell is not None: #and not is_multiple_of_primitive_cell(atoms.get_cell(), primitive_cell):
        new_translations = []
        new_rotations = []
        translation_operations = []
        # initialize primitive cell atoms object
        primitive_cell_atoms = Atoms( symbols='O%i' % len(scaled_positions),
                     cell=primitive_cell,
                     scaled_positions=scaled_positions, pbc = atoms.pbc)
        primitive_cell_dataset = spglib.get_symmetry_dataset(primitive_cell_atoms, symprec=1e-2)

        # find pure translations in the cell, i.e., translations that are due to the cell being multiple of primitive cell
        translations = get_pure_translations(atoms, error_tolerance, debug)
        # change pure translations from natural base to the cell's basis 
        natural_basis = np.identity(3)
        translations = change_basis_for_vectors(translations, natural_basis, atoms.get_cell())
        # CONVERT primitive cell symmetry operations to actual cell
        
        primitive_rotations, primitive_translations = change_symmetry_operation_cell(primitive_cell, atoms.get_cell(), primitive_cell_dataset['rotations'], primitive_cell_dataset['translations'])
        #primitive_symmetry_operations = symmetry_operations_from_rotations_and_translations(atoms, primitive_rotations, primitive_translations, error_tolerance)
        norot = np.diag([1, 1, 1])
        for j in range(translations.shape[0]):
            #translation_operation = symmetry_operation_from_rotation_and_translation(atoms, norot, translation, error_tolerance)
            for i, rotation in enumerate(primitive_rotations):
                trans = (primitive_translations[i] + translations[j]) 
                new_rotations.append(rotation)
                new_translations.append(trans)
        result = symmetry_operations_from_rotations_and_translations(atoms, np.array(new_rotations), np.array(new_translations), error_tolerance, debug)
    else:
        if debug:
            print "Primitive cell is None"
        dataset = spglib.get_symmetry_dataset(atoms, symprec=1e-2)
        new_translations = np.array(dataset['translations'], dtype=np.float, order = 'C')
        new_rotations = np.array(dataset['rotations'], dtype=np.float, order = 'C')
        result = symmetry_operations_from_rotations_and_translations(atoms, new_rotations, new_translations, error_tolerance, debug)
    return result, primitive_symmetry_operations, translation_operations

cdef list final_symmetry_operations_from_symmetry_operations_and_pure_translations(np.ndarray[np.float_t, ndim=2] positions, np.ndarray[np.float_t, ndim = 2] cell, np.ndarray[DTYPE2_t] pbc, DTYPE2_t[::1] atomic_numbers, list symmetry_operations, np.ndarray[np.float_t, ndim = 2] translations, float error_tolerance, bint debug):
    cdef list result = []
    cdef int i, j
    cdef SymmetryOperation symmetry_operation, result_operation
    cdef np.ndarray[np.float_t, ndim = 1] center, vector
    for i in range(len(symmetry_operations)):
        symmetry_operation = <SymmetryOperation>symmetry_operations[i]
        for j in range(translations.shape[0]):
            result_operation = None
            if symmetry_operation.vector is None:
                vector = None
            else:
                vector = symmetry_operation.vector.copy()
            if symmetry_operation.center is None:
                center = None
            else:  
                center = symmetry_operation.center.copy()
            t = translations[j].copy()
            if symmetry_operation.type == <bytes> 'C':
                result_operation = check_single_rotation(positions.copy(), cell, pbc, atomic_numbers, vector, (symmetry_operation.order_no + 1) * (pi2 / symmetry_operation.magnitude), center,  symmetry_operation.magnitude, symmetry_operation.order_no, error_tolerance, debug, t)
            elif symmetry_operation.type == <bytes> 'S':
                result_operation = check_improper_rotation(positions.copy(), cell, pbc, atomic_numbers, vector, (symmetry_operation.order_no + 1) * (pi2 / symmetry_operation.magnitude), center, symmetry_operation.magnitude, symmetry_operation.order_no, error_tolerance, debug, t)
            elif symmetry_operation.type == <bytes> 'sigma':
                result_operation = check_mirror(positions.copy(), cell, pbc, atomic_numbers, vector, center, debug, error_tolerance, t)
            elif symmetry_operation.type == <bytes> 'E':
                result_operation = get_translation_operation(positions.copy(), cell, t, pbc, error_tolerance, debug)
            
            if result_operation is not None:
                result.append(result_operation)
            else:
                print "Resulting symmetry operation is None. This should not happen."
                print "Vector:", vector
                print "Center:", center
                print "Type:", symmetry_operation.type, symmetry_operation.magnitude, symmetry_operation.order_no
                print "Translation:", t
                raw_input()
                break
            
    return result
 
def symmetry_operations_from_rotations_and_translations(atoms, np.ndarray[np.float_t, ndim = 3] rotations, np.ndarray[np.float_t, ndim = 2] translations, float error_tolerance, bint debug):
    result = []
    cdef np.ndarray[np.float_t, ndim = 1] translation
    cdef np.ndarray[np.float_t, ndim = 2] rotation
    for i, rotation in enumerate(rotations):
        translation = translations[i]
        so = symmetry_operation_from_rotation_and_translation(atoms, rotation, translation, error_tolerance, debug)
        if so is not None:
            result.append(so)
    return result

def symmetry_operation_from_rotation_and_translation(atoms, np.ndarray[np.float_t, ndim = 2] rotation, np.ndarray[np.float_t, ndim = 1]  translation, float error_tolerance, bint debug):
    new_atoms = atoms.copy()
    cdef np.ndarray[DTYPE2_t, ndim=1] identity = np.arange(len(atoms), dtype=DTYPE2), equals
    debug = False
    new_atoms.set_scaled_positions(apply_symmetry_operation_to_coordinates(atoms.get_scaled_positions(), rotation, translation))
    error_tolerance = 0.01
    cdef np.ndarray[DTYPE2_t, ndim = 1] pbc = np.zeros(3, dtype=DTYPE2, order = 'C')
    for i, periodic in enumerate(atoms.get_pbc()):
        if periodic:
            pbc[i] = 1
    equals = get_equals_periodic(atoms.get_positions(), new_atoms.get_positions(), atoms.get_cell(), pbc, error_tolerance, debug) # , 
    so = None
    # Finally initialize the SymmetryOperation object
    if equals is not None and -1 not in equals:
        so = SymmetryOperation("", equals, None,  vector = None, magnitude = 1, rotation_matrix=np.array(rotation, dtype=float), translation_vector=translation) # molecule_change_matrix_axis = periodicity_axis,
        if np.all(equals == identity) and translation[0] == 0 and translation[1] == 0 and translation[2] == 0:
            so.type = 'E'
    else:
        print error_tolerance
        print rotation
        print translation
        raise Exception("Equivalent not found")
    return so



cdef tuple change_symmetry_operation_cell(primitive_cell, cell, rotations, translations):
    inverse_primary_cell = np.linalg.inv(np.transpose(primitive_cell)) # inv_prim_lat
    
    cdef int i, N = len(rotations)
    cdef np.ndarray[np.float_t, ndim=2] new_translations = change_basis_for_vectors(np.array(translations, dtype=float), np.array(primitive_cell, dtype=float), np.array(cell, dtype=float))
    cdef np.ndarray[np.float_t, ndim=3] new_rotations = np.empty(rotations.shape, dtype=float)
    for i from 0 <= i < N:
        rotation = rotations[i]
        new_rotations[i] = change_basis_for_transformation_matrix(np.array(rotation, dtype=float), np.array(primitive_cell, dtype=float), np.array(cell, dtype=float))
    return new_rotations, new_translations
    
cdef tuple get_unique_atoms(np.float_t[:, ::1] positions, np.float_t[:, ::1] pure_translations, float error_tolerance):
    cdef int i, j, k, N = positions.shape[0], M = pure_translations.shape[0]
    cdef list unique_atoms = [], found_atoms = []
    cdef dict equivalent_atoms = {}
    cdef float distance
    cdef bint found
    for i in range(N):
        found = i in found_atoms
        if not found:
            unique_atoms.append(i)
            equivalent_atoms[i] = []
            for j in range(M):
                for k in range(N):
                    distance = get_distance(positions[k], reduce(positions[i], pure_translations[j]))
                    if k != i and  distance < error_tolerance:
                        found_atoms.append(k)
                        equivalent_atoms[i].append(k)
    return unique_atoms, equivalent_atoms
        

cdef np.ndarray[np.float_t, ndim = 2] get_pure_translations(atoms, float error_tolerance, bint debug):
    cdef list result = []
    cdef int i, j, N = len(atoms)
    cdef np.ndarray[np.float_t, ndim=2] positions = atoms.get_positions(), new_positions, result_np
    cdef np.ndarray[np.float_t, ndim=1] position_i, position_j, translation
    cdef np.ndarray[np.float_t, ndim=2] cell = atoms.get_cell()
    cdef np.ndarray[DTYPE2_t, ndim = 1] pbc = np.zeros(3, dtype=DTYPE2), equals
    for i, periodic in enumerate(atoms.get_pbc()):
        if periodic:
            pbc[i] = 1
    #for i from 0 <= i < N:
    position_i = positions[0]
    for j from 0 <= j < N:
        position_j = positions[j] 
        translation = position_i - position_j
        new_positions = positions + translation
        
        equals = get_equals_periodic(positions, new_positions, cell, pbc, error_tolerance, debug) # , periodicity_axis
        if equals is not None:
            result.append(translation)
    result_np = np.array(result)
    return result_np

cdef SymmetryOperation get_translation_operation(np.ndarray[np.float_t, ndim = 2] positions, np.ndarray[np.float_t, ndim = 2] cell, np.ndarray[np.float_t, ndim = 1] translation, np.ndarray[DTYPE2_t] pbc, float error_tolerance, bint debug):
    cdef list result = []
    cdef np.ndarray[np.float_t, ndim=2] new_positions = positions + translation #np.dot(translation, cell)
    cdef np.ndarray[DTYPE2_t, ndim=1] equals = get_equals_periodic(positions, new_positions, cell, pbc, error_tolerance, debug)
    cdef SymmetryOperation result_operation = None
    if equals is not None:
        if abs(sum(translation)) > 0.001:
            result_operation = SymmetryOperation("T", equals, None, vector = None, magnitude = 1, translation_vector=translation, type="T") #, molecule_change_matrix_axis = periodicity_axis
        else:
            result_operation = SymmetryOperation("E", equals, None, vector = None, magnitude = 1, type="E")
    return result_operation
                        
        
            
        
        
"""
cdef list get_pure_translations(translations, rotations):
    result = []
    for i, translation in enumerate(translations):
        rotation = rotations[i]
        if rotation[0][0] == 1 and rotation[1][1] == 1 and rotation[2][2] == 1:
            result.append(translation)
    return result
"""

cdef list get_unique(input_data):
    result = []
    for data in input_data:
        found = False
        for x, i in enumerate(data):
            data[x] = round(i, 3)  
            if abs(i) < 0.01:
                data[x] = 0.0   
            if data[x] >= 1.0:
                data[x] -= 1
            elif data[x] <= -1.0:
                data[x] += 1      
        for item in result:
            if np.all(item == data):
                found = True
        if not found:
            result.append(data)
    return result



        
        


### Space group determination

def get_lattice_system(lattice_vectors):
    assert len(lattice_vectors) == 3
    a = get_length(lattice_vectors[0])
    b = get_length(lattice_vectors[1])
    c = get_length(lattice_vectors[2])
    alpha = np.arccos(np.dot(lattice_vectors[0], lattice_vectors[1])/ (a*b))
    beta = np.arccos(np.dot(lattice_vectors[0], lattice_vectors[2]) / (a*c))
    gamma = np.arccos(np.dot(lattice_vectors[1], lattice_vectors[2]) / (b*c))
    if alpha == beta and beta == gamma:
        if alpha == math.pi/2.0:
            # alpha, beta and gamma should be 90 degrees
            if a == b and b == c:
                return   6 # CUBIC
            elif a == b or b == c or a == c:
                return  3 # TETRAGONAL
            else:
                assert a != b and b != c and a != c
                return 2 # ORTHORHOMBIC OR TRIGONAL
        else:
            return 4 # RHOMBOHEDRAL
    elif (alpha == beta and alpha == math.pi/2.0 and gamma == pi2/3.0) or (alpha == gamma and alpha == math.pi/2.0 and beta == pi2/3.0) or (beta == gamma and beta == math.pi/2.0 and alpha == pi2/3.0):
        assert (a == b or a == c or b == c) and (a != b or b != c or a != c)
        return 5 # HEXAGONAL 

    elif (alpha == beta and alpha == math.pi/2.0 and gamma != 2*math.pi/2.0) or (alpha == gamma and alpha == math.pi/2.0 and beta != 2*math.pi/2.0) or (beta == gamma and beta == math.pi/2.0 and alpha != 2*math.pi/2.0):
        assert (a != b and a != c and b != c)
        return 1 # MONOCLINIC
    else:
        assert a != b and b != c and a != c and alpha != beta and beta != gamma and alpha != gamma
        return 0 #TRICLINIC

"""
def get_lattice_centering(origo, atoms, lattice_system, lattice_vectors, cell):
    if lattice_system == 0 or lattice_system == 2 or lattice_system == 5:
        return 0 # PRIMITIVE (P)
    else:
        p1 = (origo + lattice_vectors[0]/2.0 + lattice_vectors[1]/2.0)
        p2 = (origo + lattice_vectors[0]/2.0 + lattice_vectors[2]/2.0)
        p3 = (origo + lattice_vectors[0]/2.0 + lattice_vectors[2]/2.0)
        p4 = (origo + lattice_vectors[0]/2.0 + lattice_vectors[1]/2.0 + lattice_vectors[2]/2.0)
        if atom_at(p4, cell)[0] != -1:
            return 1 # BODY CENTERED (I)
        elif atom_at(p1, cell)[0] != -1 and atom_at(p2, cell)[0] != -1 and atom_at(p3, cell)[0] != -1:
            return 2 # FACE CENTERED (F) 
        elif atom_at(p1, cell)[0] != -1:
            return 3 # A
        elif atom_at(p2, cell)[0] != -1:
            return 4 # B
        elif atom_at(p3, cell)[0] != -1:
            return 5 # C
        else:  
            return 0 # PRIMITIVE (P)
"""
        

cdef int atom_at(np.float_t[:, ::1] scaled_positions, DTYPE2_t[::1] pbc, np.float_t[::1] point, np.float_t max_error, bint convert_to_real_point) except *:
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
        R = get_distance(scaled_positions[i], real_point)
        #print R
        if abs(R) < max_error:
            return i #, periodicity_axis
    return -1 #, 13

    

def equal_position_found(positions, point, max_error=0.01):
    """
        Checks if there is an position in positions array that is close enough to the 'point' and if one is found the it's order number is returned
        point is a relative position
    """
    for i, position in enumerate(positions):
        R = get_distance(position, point)
        #print R
        if abs(R) < max_error:
            return i
    return -1


def get_rotation_matrix(axis, angle):
    axis = axis/np.sqrt(np.dot(axis,axis))
    a = np.cos(angle/2)
    b,c,d = -axis*np.sin(angle/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

def get_angle_between_vectors(vector1, vector2):
    l1 = get_length(vector1)
    l2 = get_length(vector2)
    return np.arccos(np.dot(vector1, vector2) / (l1*l2))    


    

        

cdef np.float_t[::1] get_real_point(np.float_t[::1] point, DTYPE2_t[::1] pbc) except *:
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


cdef np.float_t[:, ::1]  mirror_through_plane(np.float_t[:, ::1] positions, np.float_t[::1] normal_vector, np.float_t[::1] point, bint debug) except *:
    cdef float L = get_length(normal_vector)
    cdef float hl = 1.0+0.00001
    cdef float ll = 1.0-0.00001
    cdef float R, R_2, R_3
    
    # center of mass is along the plane always
    #  calculate the constant parameter d of plane equation 
    cdef float d = -sum_vector(multiply_vectors(point, normal_vector))
    cdef np.float_t[:, ::1] result = positions.copy()
    cdef DTYPE_t i
    cdef DTYPE_t N = positions.shape[0]

    if  L > hl or  L < ll:
        return result
    
    for i from 0 <= i < N:
        # calculate positions distance R from the plane
        # plane normal vector is normalized (sqrt(a^2+b^2+c^2)=1)
        R = abs(sum_vector(multiply_vectors(positions[i], normal_vector))+d)
        # advance - 2 * R along the normal vector (to the other side of the plane)
        result[i] = reduce(positions[i], multiply(normal_vector, 2*R))
        
        R_2 = abs(sum_vector(multiply_vectors(result[i], normal_vector)) + d)
        """if debug:
            print "-"
            print result[i]
            print -2*R*normal_vector
            print position
            print d
            print R
            print R_2"""
        if R > R_2+0.001 or R < R_2-0.001:
            result[i] = add_vectors(positions[i], multiply(normal_vector, 2*R))
            R_3 = abs(sum_vector(multiply_vectors(result[i], normal_vector)) + d)
            assert (R < R_3+0.001 and R > R_3-0.001)
    return result

cdef np.ndarray[np.float_t, ndim = 2]  rotate(np.float_t[:, ::1] positions, np.float_t[::1] axis, float angle, np.float_t[::1] center):
    divide(axis, get_length(axis))
    cdef float c = cos(angle)
    cdef float s = sin(angle)
    
    cdef np.ndarray[np.float_t, ndim = 2] p =  positions.base - center.base
    cdef np.ndarray[np.float_t, ndim = 1] ax = axis.base, cen = center.base
    cdef int i, N = positions.shape[0]
    cdef np.ndarray[np.float_t, ndim = 2] result = positions.base.copy()
    result[:] = (c * p - np.cross(p, s * ax) +
                     np.outer(np.dot(p, ax), (1.0 - c) * ax) +
                     cen)
    return result

cdef np.ndarray[np.float_t, ndim = 2] get_scaled_unnormalized_positions(np.float_t[:, ::1] cell, np.float_t[:, ::1] positions,  DTYPE2_t[::1] pbc):
    """Get positions relative to unit cell.

        Atoms outside the unit cell will be wrapped into the cell in
        those directions with periodic boundary conditions so that the
        scaled coordinates are between zero and one.
    """
    #cdef np.ndarray[np.float_t, ndim = 2] scaled = np.empty_like(positions)
    cdef np.ndarray[np.float_t, ndim = 2] scaled = np.ascontiguousarray(np.linalg.solve(cell.base.T, positions.base.T).T, dtype=np.float)
    """
    for i in xrange(positions.shape[0]):
        scaled[i, 0] = positions[i, 0] / cell[0, 0]
        scaled[i, 1] = positions[i, 1] / cell[1, 1]
        scaled[i, 2] = positions[i, 2] / cell[2, 2]
    """
    return scaled

cdef inline np.ndarray[np.float_t, ndim = 2] get_scaled_positions(np.float_t[:, ::1] cell, np.float_t[:, ::1] positions,  DTYPE2_t[::1] pbc):
    """Get positions relative to unit cell.

        Atoms outside the unit cell will be wrapped into the cell in
        those directions with periodic boundary conditions so that the
        scaled coordinates are between zero and one.
    """
    cdef np.ndarray[np.float_t, ndim = 2] scaled = np.ascontiguousarray(np.linalg.solve(cell.base.T, positions.base.T).T,  dtype=np.float)
    #cdef int i
    #cdef np.ndarray[np.float_t, ndim = 2] scaled = np.empty_like(positions)
    #cdef np.ndarray[np.float_t, ndim = 2] scaled = np.linalg.solve(cell.T, positions.T).T
    """
    for i in xrange(positions.shape[0]):
        scaled[i, 0] = positions[i, 0] / cell[0, 0]
        scaled[i, 1] = positions[i, 1] / cell[1, 1]
        scaled[i, 2] = positions[i, 2] / cell[2, 2]
    """
    for i in xrange(3):
        if pbc[i]:
            # Yes, we need to do it twice.
            # See the scaled_positions.py test
            scaled[:, i] %= 1.0
            scaled[:, i] %= 1.0
    return scaled

    
    
        
        
    
