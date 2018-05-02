#cython: boundscheck=False
#cython: wraparound=False
#cython: infer_types=True
#cython: none_check=False
#cython: c_line_in_traceback=False
 
import numpy as np
cimport numpy as np
np.import_array()
import ase
np.uint8 = np.uint8
import math
from ..water_algorithm_cython  cimport WaterAlgorithm
from help_methods import change_basis
cdef list remove_equals(list symmetry_operations, bint debug):
    cdef list result = []
    cdef SymmetryOperation sym_o, sym_b
    cdef int i, j, M, N = len(symmetry_operations) 
    cdef bint found_eq
    if debug:
        print "Symmetry operations before: %i" % N 
    for i from 0 <= i < N:
        sym_o = symmetry_operations[i]
        found_eq = False
        M = len(result)
        for j from 0 <= j < M:
            sym_b = result[j]
            if sym_b == sym_o:
                if debug:
                    print "%s and %s are equal" % (sym_b, sym_o) 
                found_eq = True
                break
            #else:
            #    print "%s and %s are not equal" % (sym_b, sym_o) 
        if not found_eq:
            result.append(sym_o)
    return result

cdef list do_symmetry_operation_filtering(list symmetry_operations, np.uint8_t current_molecule_no, np.ndarray[np.uint16_t, ndim = 1] order):
    """ 
        Remove symmetry operations from list that can't be possible at this point of water algorithm
        Also removes the identity operation
    """
    cdef list result = [], sub_symmetry_improved = []
    cdef SymmetryOperation symmetry_operation
    cdef int i, j, k, l, N = len(symmetry_operations), M = order.shape[0], O
    cdef np.ndarray[np.uint16_t, ndim = 1] inverse_order = np.zeros(M, dtype=np.uint16)
    cdef np.int8_t[::1] molecule_change_matrix
    cdef np.int8_t[:, ::1] additional_requirements
    cdef np.uint8_t n, current_molecule_order_no
    cdef bint success
    # inverse order means that the list has the order number of the molecule i with index i
    for k from 0 <= k < M:
        no = order[k]
        inverse_order[no] = k
        if no == current_molecule_no:
            current_molecule_order_no = k
    
    for i from 0 <= i < N:
        symmetry_operation = symmetry_operations[i]
        molecule_change_matrix = symmetry_operation.molecule_change_matrix
        success = True
        if molecule_change_matrix is None:
            result.append(symmetry_operation)
            continue
        if symmetry_operation.type is not <bytes> 'E':
            # check that previously handled molecules 
            for j from 0 <= j <= current_molecule_order_no:
                if inverse_order[molecule_change_matrix[order[j]]] > current_molecule_order_no:
                    success = False
                    break
            if success and current_molecule_order_no != M-1:
                for j from current_molecule_order_no < j < M:
                    if inverse_order[molecule_change_matrix[order[j]]] <= current_molecule_order_no:
                        success = False
                        break
            if success and current_molecule_order_no != M-1:
                additional_requirements = symmetry_operation.get_additional_requirements_for_molecule_no(current_molecule_no)
                O = additional_requirements.shape[0]
                #if current_molecule_order_no > 2:
                #    success = False
                #if symmetry_operation.additional_requirements is not None:
                #    success = False
    
                for l in range(O):
                    if additional_requirements[l, 0] != current_molecule_no and inverse_order[additional_requirements[l, 0]] > current_molecule_order_no:
                        success = False
                        #print "Denied by additional requirements"
                    if additional_requirements[l, 1] != current_molecule_no and inverse_order[additional_requirements[l, 1]] > current_molecule_order_no:
                        success = False
                        #print "Denied by additional requirements"
            if success:
                result.append(symmetry_operation)

    return result

cdef list filter_symmetry_operations_by_dangling_bond_profile(list symmetry_operations, np.int8_t[::1] profile):
    cdef np.int8_t[::1] profile_2
    cdef SymmetryOperation symmetry_operation
    cdef i, N = len(symmetry_operations)
    cdef list result = []
    for i in range(N):
        symmetry_operation = <SymmetryOperation>symmetry_operations[i]
        if symmetry_operation.type == <bytes> 'E':
            continue
        profile_2 = symmetry_operation.apply_for_dangling_bond_profile(profile.copy())
        if profile_2 is not None and profiles_are_equal(profile, profile_2, profile.shape[0]):
            result.append(symmetry_operation)
    print "Filtered %i operations" % (len(symmetry_operations) - len(result))
    return result

def print_profile(np.int8_t[::1] profile):
    cdef int i, N = profile.shape[0]    
    result = ""
    for i from 0 <= i < N:
        result += "%i " % profile[i]
    print result

cdef inline bint profiles_are_equal(np.int8_t[::1] orientation1, np.int8_t[::1] orientation2, int N):
    """
        Checks if two water orientation lists are equal
        Parameter:
            orientation1 : first orientation list
            orientation2 : second orientation list
            N            : The number of molecules in raft
        Returns
            0 if are not equal
            1 if they are equal
    """
    cdef int i
    for i from 0 <= i < N:
        if orientation1[i] != orientation2[i]:
            return 0
    return 1

cdef list get_sub_symmetry_level_improved(list symmetry_operations, np.uint8_t current_molecule_no, np.ndarray[np.uint16_t, ndim = 1] order):
    """ 
        Remove symmetry operations from list that can't be possible at this point of water algorithm
        Also removes the identity operation from the execution
    """
    cdef list result = [], sub_symmetry_improved = []
    cdef SymmetryOperation symmetry_operation
    cdef int i, j, k, N = len(symmetry_operations), M = order.shape[0], sub_symmetry_level, current_level
    cdef np.ndarray[np.uint16_t, ndim = 1] inverse_order = np.zeros(M, dtype=np.uint16)
    cdef np.int8_t[::1] molecule_change_matrix
    cdef np.uint8_t n, current_molecule_order_no, next_molecule_no
    cdef bint success
    for k from 0 <= k < M:
        no = order[k]
        inverse_order[no] = k
        if no == current_molecule_no:
            current_molecule_order_no = k
    
    for i from 0 <= i < N:
        symmetry_operation = symmetry_operations[i]
        sub_symmetry_level = symmetry_operation.sub_symmetry_level
        current_level = 0
        molecule_change_matrix = symmetry_operation.molecule_change_matrix
        if molecule_change_matrix is None:
            continue
        if symmetry_operation.type is not <bytes> 'E':
            for j from 0 <= j <= current_molecule_order_no:
                k = 0                
                next_molecule_no = order[j]
                while True:
                    k += 1
                    if inverse_order[next_molecule_no] > current_molecule_order_no :
                        break
                    next_molecule_no = molecule_change_matrix[next_molecule_no]
                    if next_molecule_no == order[j] or k > current_molecule_order_no:
                        current_level += 1
                        break;
            """if success and current_molecule_no != M-1:
                for j from current_molecule_order_no < j < M:
                    if inverse_order[molecule_change_matrix[order[j]]] <= current_molecule_o rder_no:
                        success = False"""
            symmetry_operation.sub_symmetry_level = current_level
            if sub_symmetry_level == 0 and current_level > sub_symmetry_level and current_level != current_molecule_order_no +1 and not symmetry_operation.found_earlier:
                result.append(symmetry_operation)

    return result



cdef bint all_found(list symmetry_operations):
    cdef SymmetryOperation symmetry_operation
    cdef int i, N = len(symmetry_operations)
    cdef list result = []
    if symmetry_operations is None or len(symmetry_operations) == 0:
        return True
    for i from 0 <= i < N:
        symmetry_operation = symmetry_operations[i]
        if symmetry_operation.type is not <bytes> 'E' and not symmetry_operation.found_earlier:
            return False
    return True

cdef void mark_earlier_found(list found_symmetry_operations, list all_symmetry_operations):
    cdef SymmetryOperation symmetry_operation
    cdef int i, N = len(all_symmetry_operations)
    for i from 0 <= i < N:
        symmetry_operation = <SymmetryOperation>all_symmetry_operations[i]
        if symmetry_operation in found_symmetry_operations:
            symmetry_operation.found_earlier = True

cdef void mark_all_not_found(list symmetry_operations):
    cdef SymmetryOperation symmetry_operation
    cdef int i, N = len(symmetry_operations)
    for i from 0 <= i < N:
        symmetry_operation = <SymmetryOperation>symmetry_operations[i]
        symmetry_operation.found_earlier = False
 
cdef list remove_earlier_found(list symmetry_operations):
    cdef SymmetryOperation symmetry_operation
    cdef int i, N = len(symmetry_operations)
    cdef list result = []
    for i from 0 <= i < N:
        symmetry_operation = symmetry_operations[i]
        if not symmetry_operation.found_earlier:
            result.append(symmetry_operation)
    return result

cdef class SymmetryOperation:
    def __init__(self, char* name, np.ndarray[np.int8_t, ndim=1] molecule_change_matrix, np.ndarray[np.uint8_t, ndim=2]  bond_change_matrix, np.ndarray[np.int8_t, ndim=1] molecule_change_matrix_axis = None, np.ndarray[np.int8_t, ndim=2]  orientation_change_matrix = None, np.ndarray[np.float_t, ndim=1]  vector = None, np.ndarray[np.float_t, ndim=1] center = None, np.int8_t magnitude = 1, char* type='C', np.int8_t order_no=0, np.ndarray[np.float_t, ndim=2] rotation_matrix=None, np.ndarray[np.float_t, ndim=1] translation_vector=None, np.ndarray[np.int8_t, ndim=2] nn_change_matrix = None, dict symbolic_bond_variable_matrix = None, bint has_translate = False, bint found_earlier = False, np.ndarray[np.int8_t, ndim=1] inverse_molecule_change_matrix = None, np.ndarray[np.int8_t, ndim=2] inverse_nn_change_matrix = None, SymmetryOperation primitive_symmetry_operation = None, SymmetryOperation translation_operation = None, np.ndarray[np.int8_t, ndim=2] additional_requirements = None, sub_symmetry_level = 0):
        i = 2
        while i <= 5:
            if order_no != 0 and magnitude != 0 and np.mod(order_no+1, i) == 0 and np.mod(magnitude, i) == 0:
                order_no = order_no / i 
                magnitude = magnitude / i
                break
            i += 1
        self.name = "%s%i (%i) - %s" % (type, magnitude, order_no, molecule_change_matrix)
        #self.name = name
        self.molecule_change_matrix = molecule_change_matrix
        self.molecule_change_matrix_axis = molecule_change_matrix_axis
        self.bond_change_matrix = bond_change_matrix
        self.orientation_change_matrix = orientation_change_matrix
        self.vector = vector
        self.magnitude = magnitude
        self.type = type
        self.center = center
        
        self.order_no = order_no
        self.nn_change_matrix = nn_change_matrix
        self.symbolic_bond_variable_matrix = symbolic_bond_variable_matrix
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector
        self.has_translate = has_translate
        self.found_earlier = found_earlier
        self.inverse_molecule_change_matrix = inverse_molecule_change_matrix
        self.inverse_nn_change_matrix = inverse_nn_change_matrix 
        self.primitive_symmetry_operation = primitive_symmetry_operation
        self.translation_operation = translation_operation
        self.additional_requirements = additional_requirements
        self.sub_symmetry_level = sub_symmetry_level

    def __reduce__(self):
        return (SymmetryOperation, # constructor to be called
        (self.name, self.molecule_change_matrix, self.bond_change_matrix, self.orientation_change_matrix, self.vector, self.magnitude, self.type, self.order_no, self.rotation_matrix, self.translation_vector, self.nn_change_matrix, self.symbolic_bond_variable_matrix, self.has_translate, self.found_earlier, self.inverse_molecule_change_matrix, self.inverse_nn_change_matrix, self.primitive_symmetry_operation, self.translation_operation, self.additional_requirements, self.sub_symmetry_level), # arguments to constructor 
        None, None, None)


    """
    def get_molecule_change_matrix(self, positions, rotation, translation, cell, centering):
        #
        #    Note: Calculates new type of molecule change matrix that contains information about in which cell the atom switched in place was of the atom
        #
        cdef np.ndarray[np.int8_t, ndim=1] result = np.zeros(positions.shape[0], dtype=np.int8)
        result.fill(-1)
        cdef np.ndarray[np.int8_t, ndim=2] original_cell_displacement = np.zeros((positions.shape[0], 3), dtype=np.int8)
        displacement_vectors = []
        for x in range(-3, 3):
            for y in range(-3, 3):
                for z in range(-3, 3):
                    displacement_vectors.append(np.array([x, y, z]))
        for displacement_vector in displacement_vectors:
            if -1 in result:
                displacement = np.dot(displacement_vector, cell) 
                new_positions = apply_symmetry_operation_to_coordinates(positions + displacement, rotation, np.dot(translation, cell), centering = centering) 
                for i, position in enumerate(positions):
                    if result[i] == -1:
                        result[i] = equal_position_found(new_positions, position, max_error=0.01)
                        original_cell_displacement[i] = np.array(displacement_vector, dtype=np.int8)
        if -1 in result:
            print "ERRORORRR! Replacing atom not found"
            raw_input()
                    
           
        #cdef np.ndarray[np.int8_t, ndim = 2] returned = np.array([result, original_cell_displacement], dtype=np.int8)
        return result, original_cell_displacement
    """

    cdef void calculate_additional_requirements(self, np.int_t[:, :, :] nearest_neighbors_nos, atoms, bint debug) except *:
        cdef bint periodic = atoms.get_pbc() is not None and any(atoms.get_pbc())
        # relevant only for periodic systems 
        if not periodic:
            return
            
        cdef list result = []
        cdef np.ndarray[np.int8_t, ndim=1] molecule_change_matrix, mcm = None
        cdef np.ndarray[np.int8_t, ndim=2] nn_change_matrix, ncm = None
        cdef np.ndarray[np.float_t, ndim=2] positions = atoms.get_positions(), original_positions = atoms.get_positions(), cell = atoms.get_cell()
        cdef np.ndarray[np.int8_t, ndim = 1] pbc = np.zeros(3, dtype=np.int8)
        angle = (self.order_no + 1) * (2*math.pi / self.magnitude)
        cdef int i, j, N
        for i, periodic in enumerate(atoms.get_pbc()):
            if periodic:
                pbc[i] = 1
        if all(pbc):
            nn_change_matrix, molecule_change_matrix = get_periodic_nn_change_matrix_from_rotation_and_translation(positions, original_positions, cell, pbc, nearest_neighbors_nos, self.rotation_matrix, self.translation_vector, debug)
        else:
            nn_change_matrix, molecule_change_matrix = get_periodic_nn_change_matrix(positions, original_positions, cell, pbc, nearest_neighbors_nos, angle, self.vector, self.center, self.translation_vector, self.type, debug, False)
        N = molecule_change_matrix.shape[0] 
        
        
        while mcm is None or not profiles_are_equal(molecule_change_matrix, mcm, molecule_change_matrix.shape[0]):
            original_positions -= cell[0]
            if all(pbc):
                ncm, mcm = get_periodic_nn_change_matrix_from_rotation_and_translation(positions, original_positions, cell, pbc, nearest_neighbors_nos, self.rotation_matrix, self.translation_vector, debug)
            else:
                 ncm, mcm = get_periodic_nn_change_matrix(positions, original_positions, cell, pbc, nearest_neighbors_nos, angle, self.vector, self.center, self.translation_vector, self.type, debug, False)
            
            for i in range(N):
                for j in range(4):
                    if mcm[i] != molecule_change_matrix[i] or nn_change_matrix[i, j] != ncm[i, j]:
                        result.append([molecule_change_matrix[i], mcm[i], nn_change_matrix[i, j], ncm[i, j]])
        mcm = None     
        original_positions = atoms.get_positions()          
        while mcm is None or not profiles_are_equal(molecule_change_matrix, mcm, molecule_change_matrix.shape[0]):
            original_positions -= cell[1]
            if all(pbc):
                ncm, mcm = get_periodic_nn_change_matrix_from_rotation_and_translation(positions, original_positions, cell, pbc, nearest_neighbors_nos, self.rotation_matrix, self.translation_vector, debug)
            else:
                 ncm, mcm = get_periodic_nn_change_matrix(positions, original_positions, cell, pbc, nearest_neighbors_nos, angle, self.vector, self.center, self.translation_vector, self.type, debug, False)
            for i in range(N):
                for j in range(4):
                    if mcm[i] != molecule_change_matrix[i] or nn_change_matrix[i, j] != ncm[i, j]:
                        result.append([molecule_change_matrix[i], mcm[i], nn_change_matrix[i, j], ncm[i, j]])
        
        
        """if self.molecule_change_matrix is not None and self.inverse_molecule_change_matrix is not None:
            for i, molecule_no_i in enumerate(self.molecule_change_matrix):
                inverse_molecule_no_i = self.inverse_molecule_change_matrix[i]
                for j, molecule_no_j in enumerate(self.molecule_change_matrix):
                    inverse_molecule_no_j = self.inverse_molecule_change_matrix[j]
                    if i < j and molecule_no_i == molecule_no_j:
                        for k, new_index_i in enumerate(self.nn_change_matrix[i]):
                            for l, new_index_j in enumerate(self.nn_change_matrix[j]):
                                if new_index_i == new_index_j:
                                    result.append([i, j, k, l])
                    if i < j and inverse_molecule_no_i == inverse_molecule_no_j:
                        for k, new_index_i in enumerate(self.inverse_nn_change_matrix[i]):
                            for l, new_index_j in enumerate(self.inverse_nn_change_matrix[j]):
                                if new_index_i == new_index_j:
                                    result.append([i, j, k, l])"""
        if len(result) > 0:
            self.additional_requirements = np.array(result, dtype=np.int8)
            #print "--------------------"
            #print self.molecule_change_matrix, self.magnitude, self.type, self.center, self.translation_vector, self.vector
            #print "Additional requirements:", self.additional_requirements
            #raw_input()
        else:
            self.additional_requirements = None

    cdef bint are_additional_requirements_met_for_dangling_bond_profile(self, np.int8_t[::1] profile):
        cdef np.int8_t[:, ::1] additional_requirements = self.additional_requirements
        cdef np.int8_t profile_value_i, profile_value_j
        cdef int i, N 
        if self.additional_requirements is not None:
            N = additional_requirements.shape[0]
            for i in range(N):
                profile_value_i = profile[additional_requirements[i, 0]]
                profile_value_j = profile[additional_requirements[i, 1]]
                if profile_value_i != 0 and profile_value_j != 0 and profile_value_i != profile_value_j:
                    return False
                    
        return True
    
    cdef np.int8_t[:, ::1] get_additional_requirements_for_molecule_no(self, int molecule_no):
        cdef np.int8_t[:, ::1] additional_requirements = self.additional_requirements
        cdef np.int8_t[:, ::1] result = None
        cdef int i, N, result_count = 0
        if additional_requirements is not None:
            result = additional_requirements.copy()
            N = additional_requirements.shape[0]
            for i in range(N):
                if additional_requirements[i, 0] == molecule_no or additional_requirements[i, 1] == molecule_no:
                    result[result_count] = additional_requirements[i]
                    result_count += 1
            result = result[:result_count]
        return result

    cdef bint are_additional_requirements_met(self, np.int8_t[::1] water_orientations):
        cdef np.int8_t[:, ::1] additional_requirements = self.additional_requirements
        cdef np.int8_t[::1] bond_variables_i, bond_variables_j
        cdef int i, N
        
        if water_orientations is not None and additional_requirements is not None:
            N = additional_requirements.shape[0]
            for i in range(water_orientations.shape[0]):
                if water_orientations[i] == -1:
                    return False

            for i in range(N):
                bond_variables_i = get_bond_variable_values_from_water_orientation(water_orientations[additional_requirements[i, 0]])
                bond_variables_j = get_bond_variable_values_from_water_orientation(water_orientations[additional_requirements[i, 1]])
                #if bond_variables_i[additional_requirements[i, 2]] == 0 and bond_variables_j[additional_requirements[i, 3]] == 0:
                #    continue
                if bond_variables_i[additional_requirements[i, 2]] == 0 or bond_variables_j[additional_requirements[i, 3]] == 0 or bond_variables_i[additional_requirements[i, 2]] != bond_variables_j[additional_requirements[i, 3]]:
                    return False
                    
        return True
                
            
                    
                     
            
        
    cdef np.ndarray[np.int8_t, ndim=2] calculate_orientation_change_matrix(self):
        cdef np.ndarray[np.int8_t, ndim=2] orientation_change_matrix = np.empty((self.nn_change_matrix.shape[0], 12), dtype=np.int8)
        cdef np.ndarray[np.int8_t, ndim=2] nn_change_matrix = self.nn_change_matrix
        cdef int i, j, N = self.nn_change_matrix.shape[0]
        cdef np.int8_t[::1] current_bond_variables, new_bond_variables
        cdef np.ndarray[np.int8_t, ndim=1] nn_matrix
        cdef np.uint8_t M = nn_change_matrix.shape[1]
        for orientation from 0 <= orientation < 12:
            current_bond_variables = get_bond_variable_values_from_water_orientation(orientation)
            for i from 0 <= i < N:
                nn_matrix = nn_change_matrix[i]
                new_bond_variables = np.empty(4, dtype=np.int8)
                for j from 0 <= j < M:
                    new_bond = nn_matrix[j]
                    new_bond_variables[j] = current_bond_variables[new_bond]
                orientation_change_matrix[i, orientation] = get_water_orientation_from_bond_variable_values(new_bond_variables)
        
        return orientation_change_matrix
    
    
    cdef void calculate_nn_change_matrix(self, np.int_t[:, :, :] nearest_neighbors_nos, atoms, bint debug) except *:   
        cdef np.ndarray[np.int8_t, ndim=2] result = np.empty((nearest_neighbors_nos.shape[1], nearest_neighbors_nos.shape[2]), dtype=np.int8)
        cdef np.ndarray[np.int8_t, ndim=2] result_inverse = np.empty((nearest_neighbors_nos.shape[1], nearest_neighbors_nos.shape[2]), dtype=np.int8) 
        cdef np.ndarray[np.float_t, ndim=2] positions, original_positions, cell
        cdef np.ndarray[np.int8_t, ndim=1] molecule_change_matrix = self.molecule_change_matrix, inverse_molecule_change_matrix = self.inverse_molecule_change_matrix
        cdef int i, j, new_no, N = molecule_change_matrix.shape[0], M = nearest_neighbors_nos.shape[2], ij_periodic, axis, new_axis, k,  nn_n, nn_s, new_inverse_no, nn_b, nn_c
        cdef np.ndarray[np.float_t, ndim=2] rotation_matrix = self.rotation_matrix
        cdef np.ndarray[np.float_t, ndim=1] translation_vector = self.translation_vector
        cdef np.ndarray[np.int8_t, ndim=1] a
        cdef bint periodic = any(atoms.get_pbc())
        cdef np.int8_t[::1] unique_molecules
        cdef np.ndarray[np.int8_t, ndim = 1] pbc
        result.fill(-1)
        result_inverse.fill(-1)
            # Result size should be 20 x 4 in Ice 20:s case (20 molecules and 4 neighbors)
            #  The result consist of a matrix that has the order number of the bond of new_
        if debug:
            print "------------------------------------"
            print self.rotation_matrix
            print self.translation_vector
            print self.molecule_change_matrix
        if periodic:        
            angle = (self.order_no + 1) * (2*math.pi / self.magnitude)
            positions = atoms.get_positions()
            original_positions = positions.copy()
            cell = atoms.get_cell()
            pbc = np.zeros(3, dtype=np.int8)
            for i, periodic in enumerate(atoms.get_pbc()):
                if periodic:
                    pbc[i] = 1
            if all(pbc):
                self.nn_change_matrix, self.molecule_change_matrix = get_periodic_nn_change_matrix_from_rotation_and_translation(positions, original_positions, cell, pbc, nearest_neighbors_nos, rotation_matrix, translation_vector, debug)
            else:
                self.nn_change_matrix, self.molecule_change_matrix = get_periodic_nn_change_matrix(positions, original_positions, cell, pbc, nearest_neighbors_nos, angle, self.vector, self.center, translation_vector, self.type, debug, False)
            try_count = 0
            unique = molecule_change_matrix_is_unique(self.molecule_change_matrix)
            while not unique and try_count < 10:
                unique_molecules = molecule_change_matrix_unique_molecules(self.molecule_change_matrix)
                for i in range(N):
                    if not unique_molecules[i]:
                        original_positions[i] -= cell[0]
                if all(pbc):
                    self.nn_change_matrix, self.molecule_change_matrix = get_periodic_nn_change_matrix_from_rotation_and_translation(positions, original_positions, cell, pbc, nearest_neighbors_nos, rotation_matrix, translation_vector, debug)
                else:
                    self.nn_change_matrix, self.molecule_change_matrix = get_periodic_nn_change_matrix(positions, original_positions, cell, pbc, nearest_neighbors_nos, angle, self.vector, self.center, translation_vector, self.type, debug, False)
                try_count += 1
                unique = molecule_change_matrix_is_unique(self.molecule_change_matrix)
                if try_count == 10:
                    print "Limit reached"
            if all(pbc):
                self.inverse_nn_change_matrix, self.inverse_molecule_change_matrix = get_periodic_nn_change_matrix_from_rotation_and_translation(positions, original_positions, cell, pbc, nearest_neighbors_nos, self.rotation_matrix, self.translation_vector, debug)
            else:
                self.inverse_nn_change_matrix, self.inverse_molecule_change_matrix = get_periodic_nn_change_matrix(positions, original_positions, cell, pbc, nearest_neighbors_nos, angle, self.vector, self.center, translation_vector, self.type, debug, True)
        else:
            for i from 0 <= i < N:
                new_no = molecule_change_matrix[i]
                #new_inverse_no = inverse_molecule_change_matrix[i]
                
                # go through all the nearest neighbors of currently selected molecule
                for j from 0 <= j < M:
                    nn_s = nearest_neighbors_nos[0, i, j]
                    
                    #nn_b = nearest_neighbors_nos[0, new_inverse_no, j]
                    #print "HeI, J %i, %i" % (i, j)
                    ij_periodic = nearest_neighbors_nos[1, i, j]
                    #new_axis = apply_symmetry_operation_to_periodicity_axis(i, nn_s, new_no, molecule_change_matrix[nn_s], rotation_matrix, translation_vector, nearest_neighbors_nos[2, i, j], water_algorithm, debug)
                    #print_str = "Setting (%i - %i)_%i" % (i, nn_s, nearest_neighbors_nos[2][i][j])
                        
                    #print "Length of nn %i" % len(nearest_neighbors_nos[0][new_no])
                    
                    #if not periodic:
                    a = np.array([new_no, molecule_change_matrix[nn_s], 13], dtype=np.int8)
                    #else:
                    #    angle = (self.order_no + 1) * (2*math.pi / self.magnitude)
                    #    a  = get_nn_change_for_single_bond(i, nn_s, angle, self.vector, translation_vector, self.center, self.type, nearest_neighbors_nos[2, i, j], atoms, debug, False)
                    #    b = get_nn_change_for_single_bond(i, nn_s, -angle, self.vector, translation_vector, self.center, self.type, nearest_neighbors_nos[2, i, j], atoms, debug, True)
                    #    inverse_molecule_change_matrix[i] = b[0]

                    
                    for k from 0 <= k < M:
                        nn_n = nearest_neighbors_nos[0, new_no, k]
                        jk_periodic = nearest_neighbors_nos[1, new_no, k]
                        # if the new position of current neighbor is a neighbor of the new position
                        #  then set the neighbors order number to the result
                        if a[1] == nn_n:
                            axis = nearest_neighbors_nos[2, new_no, k]
                            if axis == a[2]:
                                #assert result[i, j] == -1
                                #print_str += " to (%i, %i)_%i" % (new_no, nn_n, axis)
                                result[i, j] = k
                                #break
                        """if translation_vector is not None:     
                            nn_n = nearest_neighbors_nos[0, b[0], k]
                            if b[1] == nn_n:
                                axis = nearest_neighbors_nos[2, b[0], k]
                                if axis == b[2]:
                                    result_inverse[i, j] = k
                        """
                   
                        
                    
                    """if a[0] != new_no or result[i, j] == -1:
                        get_nn_change_for_single_bond(i, nn_s, angle, self.vector, translation_vector, self.center, self.type, nearest_neighbors_nos[2, i, j], atoms, True, False)
                        print i, nn_s, nearest_neighbors_nos[2, i, j], "new_bond: ", a
                        print new_no,"nn: ", nearest_neighbors_nos.base[0, new_no], nearest_neighbors_nos.base[2, new_no]
                        print "Result: ", result[i, j]
                        print "MCM: ", molecule_change_matrix
                        print "Type: ", self.type, self.magnitude, self.order_no
                        print "Center: ", self.center
                        print "Vector: ", self.vector
                        print "Translation: ", self.translation_vector
                        raw_input()"""
                                 
                    #assert result[i, j] != -1
                    """ 
                        if result[i, j] == -1:
                            print self.rotation_matrix
                            print self.translation_vector
                            print self.molecule_change_matrix
                            print i, nn_s, molecule_change_matrix[i], molecule_change_matrix[nn_s]
                            print nearest_neighbors_nos[2, i, j]
                            print new_axis
                            print result[i, j] = get_nn_change_for_single_bond(i, nn_s, rotation_matrix, translation_vector, nearest_neighbors_nos[2, i, j], water_algorithm, debug)
                            print "Equivalent not found"
                            
                            raw_input()
                            #print "---------"get_nn_change_for_single_bond(atom_no, neighbor_no, rotation, translation, np.uint8_t periodicity_axis, WaterAlgorithm water_algorithm, debug)
                            #raw_input()
                            # THROW away the correct periodicity requirement
                            # EXPERIMENTAL: may cause errors
                            # if the new position of current neighbor is a neighbor of the new position
                            #  then set the neighbors order number to the result
                            for k from 0 <= k < M:
                                nn_n = nearest_neighbors_nos[0, new_no, k]
                                jk_periodic = nearest_neighbors_nos[1, new_no, k]
                                
                                # if the new position of current neighbor is a neighbor of the new position
                                #  then set the neighbors order number to the result
                                if molecule_change_matrix[nn_s] == nn_n:
                                    if result[i, j] != -1:
                                        print i, new_no, nn_s, molecule_change_matrix[nn_s]
                                        print self.molecule_change_matrix
                                        print ij_periodic != jk_periodic 
                                        print "Several possibilities"
                                        if ij_periodic == jk_periodic:
                                             result[i, j] = k
                                        raw_input()
                                    #print_str += " to (%i, %i)_%i" % (new_no, nn_n, axis)
                                    else:
                                        result[i, j] = k"""
                    """
                        if new_axis != 13:
                            # SEEK for periodic -> periodic match
                            for k, nn_n in enumerate(nearest_neighbors_nos[0][new_no]):
                                jk_periodic = nearest_neighbors_nos[1][new_no][k]
                                # if the new position of current neighbor is a neighbor of the new position
                                #  then set the neighbors order number to the result
                                if self.molecule_change_matrix[nn_s] == nn_n :
                                    axis = nearest_neighbors_nos[2][new_no][k]
                                    if axis == new_axis:
                                        assert result[i][j] == -1
                                        print_str += " to (%i, %i)_%i" % (new_no, nn_n, axis)
                                        result[i][j] = k
                                        break

                            if result[i][j] == -1:
                                # SEEK periodic -> normal match
                                for k, nn_n in enumerate(nearest_neighbors_nos[0][new_no]):
                                    if self.molecule_change_matrix[nn_s] == nn_n :
                                        axis = nearest_neighbors_nos[2][new_no][k]
                                        if axis == 13:
                                            print_str += " to (%i, %i)_%i" % (new_no, nn_n, axis)
                                            result[i][j] = k
                                            break
                            assert result[i][j] != -1 """
                """for i, new_no in enumerate(self.molecule_change_matrix):
                    # go through all the nearest neighbors of currently selected molecule
                    for j, nn_s in enumerate(nearest_neighbors_nos[0][i]):
                        ij_periodic = nearest_neighbors_nos[1][i][j]
                        new_axis = apply_symmetry_operation_to_periodicity_axis(i, nn_s, new_no, self.molecule_change_matrix[nn_s], self.rotation_matrix, self.translation_vector, nearest_neighbors_nos[2][i][j], water_algorithm)
                        print_str = "Setting (%i - %i)_%i" % (i, nn_s, nearest_neighbors_nos[2][i][j])
                        # SEEK normal - normal match
                        count = 0
                        if result[i][j] == -1:
                            for k, nn_n in enumerate(nearest_neighbors_nos[0][new_no]):
                                if self.molecule_change_matrix[nn_s] == nn_n and k not in result[i]:                        
                                    axis = nearest_neighbors_nos[2][new_no][k]   
                                    print_str += " to (%i, %i)_%i" % (new_no, nn_n, axis)
                                    result[i][j] = k
                                    break

                                 
                        # Assert that there is no empty spots in the result
                        assert result[i][j] != -1
                    assert 0 in result[i] and 1 in result[i] and 2 in result[i] and 3 in result[i]"""
            #if translation_vector is not None:
            #    assert -1 not in result_inverse
            self.inverse_nn_change_matrix = result_inverse 
            self.nn_change_matrix = result
            #self.name = result + " %i" % result[0]W
             
         
    def __str__(self):
        if self.vector != None:
            return   "%s%i_%i ** Vector: %f, %f, %f Center: %f, %f, %f MCM: %s" % (self.type, self.magnitude, self.order_no, self.vector[0], self.vector[1], self.vector[2], self.center[0], self.center[1], self.center[2], self.molecule_change_matrix)
        elif self.primitive_symmetry_operation is not None:
            
            result = str(self.primitive_symmetry_operation)
            result += "\nTranslation\n"
            result += str(self.translation_operation)
            return result
        else:
            result = str(self.translation_vector) +"\n"
            result = str(self.rotation_matrix) +"\n"
            return result +  self.name + " %s" % self.has_translate

    def __richcmp__(self, SymmetryOperation other, int op):
        if op == 2:
            return self.eq(other)
    
    def eq(self, other):
        cdef bint matrices_agree = np.all(np.equal(self.molecule_change_matrix, other.molecule_change_matrix)), nns_agree
        if not matrices_agree:
            return False
        if self.nn_change_matrix is not None and other.nn_change_matrix is not None:
            nns_agree = np.all(np.equal(self.nn_change_matrix, other.nn_change_matrix)) 
            return matrices_agree and nns_agree
        else:
            return <bytes> self.type == <bytes> other.type 
            
        cdef bint rotations_agree = (self.rotation_matrix is None and other.rotation_matrix is None) or ( self.rotation_matrix is not None and  other.rotation_matrix is not None and (self.rotation_matrix - other.rotation_matrix < 0.01).all() )
        cdef bint translations_agree = (self.translation_vector is None and other.translation_vector is None) or ( self.translation_vector is not None and other.translation_vector is not None and ((self.translation_vector - other.translation_vector) < 0.01).all())
       
        if matrices_agree and not (rotations_agree and translations_agree):
            print "MATRICES AGREE"
            print self.rotation_matrix, other.rotation_matrix, self.translation_vector, other.translation_vector, rotations_agree, translations_agree
            if  self.rotation_matrix is not None and other.rotation_matrix is not None:
                print (self.rotation_matrix - other.rotation_matrix < 0.01).all()
            raw_input()
        return matrices_agree and  rotations_agree and translations_agree and self.type == other.type
        if self.vector == None and other.vector == None:
            return self.name  == other.name
        else:
            return all(self.molecule_change_matrix == other.molecule_change_matrix)
            #diff = np.sum(np.abs(self.vector-other.vector))
            #diff2 = np.sum(np.abs(self.vector+other.vector))
            #return (diff < 10**-2 or diff2 < 10**-2) and self.order_no == other.order_no and self.type == other.type and (self.magnitude == other.magnitude or (self.magnitude < other.magnitude and float(self.order_no) / float(self.magnitude) == float(other.order_no) / float(other.magnitude)))

    cdef public bint are_symmetric(self, np.int8_t[::1] water_orientations, np.int8_t[::1] water_orientations2):
        cdef bint requirements_met = self.are_additional_requirements_met(water_orientations)
        if not requirements_met:
            return False
        cdef np.int8_t[::1] molecule_change_matrix = self.molecule_change_matrix
        cdef np.int8_t new_no
        cdef np.uint8_t i, j, N = molecule_change_matrix.shape[0]
        cdef np.int8_t current_orientation, new_orientaiton
        cdef np.int8_t[:, ::1] orientation_change_matrix = self.orientation_change_matrix
        for i from 0 <= i < N:
            new_no = molecule_change_matrix[i]
            current_orientation = water_orientations[new_no]
            if current_orientation == -1 or water_orientations[i] == -1:
                new_orientation = -1
            else:
                new_orientation = orientation_change_matrix[i, current_orientation]
            if water_orientations2[i] != new_orientation:
                return False
        return True
    
    cdef public np.int8_t[::1]  apply(self, np.int8_t[::1]  water_orientations):
        #if self.primitive_symmetry_operation is not None:
        #    return self.translation_operation.apply(self.primitive_symmetry_operation.apply(water_orientations, water_algorithm, nearest_neighbors_nos, inverse), water_algorithm, nearest_neighbors_nos, inverse) 
        
        #    #print self.orientation_change_matrix
        #    #raw_input()
        cdef np.int8_t[::1] result     
        cdef bint requirements_met = self.are_additional_requirements_met(water_orientations)
        if not requirements_met:
            result = None
        else:
            result = self.apply_using_orientation_change_matrix(water_orientations) 
        #print "---------"
        
        #print water_orientations
        #result = self.apply_using_nn_change_matrix(water_algorithm, water_orientations, nearest_neighbors_nos, inverse)
        #print result
        #if np.any(result != self.apply_using_nn_change_matrix(water_algorithm, water_orientations, nearest_neighbors_nos)):
        #    print self.molecule_change_matrix
        #    print self.nn_change_matrix
        #    print self.orientation_change_matrix
        return result

    cdef public np.int8_t[::1]  apply_for_dangling_bond_profile(self, np.int8_t[::1]  profile):
        cdef np.int8_t[::1] molecule_change_matrix = self.molecule_change_matrix
        cdef np.uint8_t i, N = molecule_change_matrix.shape[0]
        cdef np.int8_t[::1] result = profile.copy()
        cdef np.int8_t current_orientation, new_no
        for i from 0 <= i < N:
            new_no = molecule_change_matrix[i]
            current_orientation = profile[new_no]
            result[i] = current_orientation
        if not self.are_additional_requirements_met_for_dangling_bond_profile(profile):
            return None
        return result
 
    cdef void initialize(self, np.int_t[:, :, ::1] nearest_neighbors_nos, atoms) except *:
        self.calculate_nn_change_matrix(nearest_neighbors_nos, atoms, False)
        self.calculate_additional_requirements(nearest_neighbors_nos, atoms, False)
        self.orientation_change_matrix = self.calculate_orientation_change_matrix() 
        
        
    cdef np.int8_t[::1]  apply_using_orientation_change_matrix(self, np.int8_t[::1] water_orientations):
        cdef np.int8_t[::1] molecule_change_matrix = self.molecule_change_matrix
        cdef np.int8_t new_no
        cdef np.uint8_t i, j, N = molecule_change_matrix.shape[0]
        cdef np.int8_t current_orientation
        cdef np.int8_t[:, ::1] orientation_change_matrix = self.orientation_change_matrix
        cdef np.int8_t[::1] result = water_orientations.copy()
        for i from 0 <= i < N:
            new_no = molecule_change_matrix[i]
            current_orientation = water_orientations[new_no]
            if current_orientation == -1 or water_orientations[i] == -1:
                result[i] = -1
            else:
                result[i] = orientation_change_matrix[i, current_orientation]
        return result
                
    """
    cdef np.ndarray[np.int8_t, ndim=1] apply_using_nn_change_matrix(self, atoms, np.ndarray[np.int8_t, ndim=1] water_orientations, np.ndarray[np.int_t, ndim=3] nearest_neighbors_nos, bint  inverse):
        cdef np.uint8_t i, j, new_no, N = water_orientations.shape[0]
        cdef np.ndarray[np.int8_t, ndim=1] molecule_change_matrix
        cdef np.int8_t current_orientation, water_orientation, new_bond
        cdef np.ndarray[np.int8_t, ndim=1] current_bond_variables, new_bond_variables
        
        # Lazy load for nn change matrix
        if self.nn_change_matrix is None:
            self.calculate_nn_change_matrix(nearest_neighbors_nos, atoms, False)
            self.calculate_additional_requirements()
        cdef np.ndarray[np.int8_t, ndim=2] nn_change_matrix
        cdef np.ndarray[np.int8_t, ndim=1] nn_matrix
        cdef np.ndarray[np.int8_t, ndim=1] result = np.empty(N, dtype=np.int8)
        if not self.are_additional_requirements_met(water_orientations):
            result.fill(-1)
            return result
        
        
        if inverse and self.inverse_molecule_change_matrix is not None:
            nn_change_matrix = self.inverse_nn_change_matrix
            molecule_change_matrix = self.inverse_molecule_change_matrix
        else:
            nn_change_matrix = self.nn_change_matrix # size (N, M)
            molecule_change_matrix = self.molecule_change_matrix
            
        cdef np.uint8_t M = nn_change_matrix.shape[1]

        
        for i from 0 <= i < N:
    
            new_no = molecule_change_matrix[i]
            #assert new_no >= 0
            nn_matrix = nn_change_matrix[i]
            #print "New no %i vs len(water_orientations): %i " % (new_no,  len(water_orientations)-1)
            #if new_no > N-1: #cannot be symmetric
            #    return None
            current_orientation = water_orientations[new_no]
            if current_orientation == -1 or water_orientations[i] == -1:
                result[i] = -1
                continue
            current_bond_variables = get_bond_variable_values_from_water_orientation(current_orientation)
            new_bond_variables = np.empty_like(current_bond_variables)
            for j from 0 <= j < M:
                new_bond = nn_matrix[j]
                new_bond_variables[j] = current_bond_variables[new_bond]
                
            #print "new_bond_variables"
            #print new_bond_variables
            #print get_water_orientation_from_bond_variable_values(new_bond_variables)
            result[i] = get_water_orientation_from_bond_variable_values(new_bond_variables)
        return result
        
    cdef np.ndarray[np.int8_t, ndim=1] apply_the_hard_way(self, WaterAlgorithm water_algorithm, np.ndarray[np.int8_t, ndim=1] water_orientations):
        cdef np.ndarray[np.int8_t, ndim=1] result = np.zeros(len(water_orientations), dtype=np.int8)
        cdef np.uint8_t i,  N, new_no = water_orientations.shape[0]
        cdef np.int8_t current_orientation, water_orientation
        cdef np.ndarray[np.int8_t, ndim=1] current_bond_variables, new_bond_variables
        for i from 0 <= i < N:
            water_orientation = water_orientations[i]
            new_no = np.abs(self.molecule_change_matrix[i])
            #print "New no %i vs len(water_orientations): %i " % (new_no,  len(water_orientations)-1)
            if new_no > N-1: #cannot be symmetric
                return None
            current_orientation = water_orientations[new_no]
            if current_orientation == -1:
                result[i] = -1
                continue
            current_bond_variables = get_bond_variable_values_from_water_orientation(current_orientation)
            new_bond_variables = self.apply_to_bond_variables(water_algorithm,  new_no,  current_bond_variables)
            #print "new_bond_variables"
            #print new_bond_variables
            #print get_water_orientation_from_bond_variable_values(new_bond_variables)
            result[i] = get_water_orientation_from_bond_variable_values(new_bond_variables)
        return result
             
    cdef np.ndarray[np.int8_t, ndim=1] apply_to_bond_variables(self, WaterAlgorithm water_algorithm, np.uint8_t molecule_no, np.ndarray[np.int8_t, ndim=1]  current_bond_variables):
        cdef np.uint8_t s = water_algorithm.get_water_molecule_class(molecule_no)
        cdef np.uint8_t N = current_bond_variables.shape[0]
        cdef np.ndarray[np.int8_t, ndim=1] result = np.zeros(N, dtype=np.int8)
        for i from 0 <= i < N:
            result[i] = current_bond_variables[self.bond_change_matrix[s][i]]
        return result
    """

    
        
        
    cdef dict get_symbolic_bond_variable_matrix(self, np.ndarray[np.int_t, ndim=3] nearest_neighbors_nos, WaterAlgorithm water_algorithm, bint inverse):
        if self.symbolic_bond_variable_matrix is None:
            if self.bond_change_matrix is not None:
                self.symbolic_bond_variable_matrix = get_symbolic_bond_variable_matrix(self.molecule_change_matrix,  self.bond_change_matrix, nearest_neighbors_nos)
            else:
                if self.primitive_symmetry_operation is not None:
                    primitive = self.primitive_symmetry_operation.get_symbolic_bond_variable_matrix(nearest_neighbors_nos, water_algorithm, inverse)
                    translation = self.translation_operation.get_symbolic_bond_variable_matrix(nearest_neighbors_nos, water_algorithm, inverse)
                    self.symbolic_bond_variable_matrix = combine_symbolic_bond_variable_matrices(primitive, translation)
                else:
                    
                    if inverse:
                        self.symbolic_bond_variable_matrix = get_symbolic_bond_variable_matrix_from_nn(self.inverse_molecule_change_matrix, self.inverse_nn_change_matrix, nearest_neighbors_nos)
                    else:
                        self.symbolic_bond_variable_matrix = get_symbolic_bond_variable_matrix_from_nn(self.molecule_change_matrix, self.nn_change_matrix, nearest_neighbors_nos)
        return self.symbolic_bond_variable_matrix
        

cdef inline bint molecule_change_matrix_is_unique(np.int8_t[::1] molecule_change_matrix):
    cdef int i, j, N = molecule_change_matrix.shape[0]   
    for i in range(N):
        for j in range(N):
            if i != j and molecule_change_matrix[i] == molecule_change_matrix[j]:
                return False
    return True      

cdef inline np.int8_t[::1] molecule_change_matrix_unique_molecules(np.int8_t[::1] molecule_change_matrix):
    cdef int i, j, N = molecule_change_matrix.shape[0]   
    cdef np.int8_t[::1] result = molecule_change_matrix.copy() 
    for i in range(N):
        result[i] = 1
        for j in range(i):
            if i != j and molecule_change_matrix[i] == molecule_change_matrix[j]:
                result[i] = 0
    return result  

cdef dict get_symbolic_bond_variable_matrix_from_nn(np.ndarray[np.int8_t, ndim=1] molecule_change_matrix, np.ndarray[np.int8_t, ndim=2] nn_change_matrix, np.ndarray[np.int_t, ndim=3] nearest_neighbors_nos):
    """
        Result is a dict that contains the changes in bonds

        Each matrix element contains:
            0: The number of the first molecule that is switched to its place
            1: The number of the other molecule switched to its place
            2: Periodicity axis (13 if not periodic)
             

        N is the number of molecules in system        
    """
    cdef int new_molecule_no, bond_no, molecule_no, M, molecule_count = molecule_change_matrix.shape[0], neighbor_no, periodicity_axis, nmno
    cdef np.ndarray[np.int8_t, ndim=1] matrix
    cdef np.int8_t new_bond_no
    cdef dict result = {}
    for molecule_no from 0 <= molecule_no < molecule_count:
        new_molecule_no = molecule_change_matrix[molecule_no]
        if molecule_no not in result:
            result[molecule_no] = {}
        matrix = nn_change_matrix[molecule_no]
        M = matrix.shape[0]
        for bond_no from 0 <= bond_no < M:
            new_bond_no = matrix[bond_no]
            neighbor_no = nearest_neighbors_nos[0][molecule_no][bond_no] 
            periodicity_axis = nearest_neighbors_nos[2][molecule_no][bond_no]
            if neighbor_no not in result[molecule_no]:
                result[molecule_no][neighbor_no] = {}
            if periodicity_axis not in result[molecule_no][neighbor_no]:
                result[molecule_no][neighbor_no][periodicity_axis] = np.array([new_molecule_no, nearest_neighbors_nos[0, new_molecule_no, new_bond_no], nearest_neighbors_nos[2, new_molecule_no, new_bond_no]], dtype = np.int8)

    return result

def print_symbolic_bond_variable_matrix(sbvm):
    for sbvmp in sbvm:
        print "--------------------------------"
        for i,  sbvmr in enumerate(sbvmp):
            row = "%i: " % i
            for sbv in sbvmr:
                if sbv[0] != -1:
                    str = " "
                    if sbv[2] == 1:
                        str = "*"
                    row += " %i, %i%s |" % (sbv[0],  sbv[1], str)
                else:
                    row += " -, -  |"
            print row
    

def get_symbolic_bond_variable_matrix(molecule_change_matrix,  bond_change_matrix,  nearest_neighbors_nos):
        molecule_count = len(molecule_change_matrix)
        result = np.empty((molecule_count,  molecule_count,  3))
        result.fill(-1)
        result_periodic = np.empty((molecule_count,  molecule_count,  3))
        result_periodic.fill(-1)
        for molecule_no,  new_molecule_no in enumerate(molecule_change_matrix):
            change_matrix_mul = np.mod(molecule_no,  2) 
            if change_matrix_mul >= len(bond_change_matrix):
                change_matrix_mul = 0
            for bond_no,  new_bond_no in enumerate(bond_change_matrix[change_matrix_mul]):
                if nearest_neighbors_nos[1][molecule_no][bond_no]:
                    res = result_periodic
                else:
                    res = result
                neighbor_no = nearest_neighbors_nos[0][molecule_no][bond_no] 
                res[molecule_no][neighbor_no][0] = np.abs(new_molecule_no)
                nmno = np.abs(new_molecule_no)
                bonded_molecule_no = nearest_neighbors_nos[0][nmno][new_bond_no]
                res[molecule_no][neighbor_no][2] =  nearest_neighbors_nos[2][nmno][new_bond_no]
                res[molecule_no][neighbor_no][1] = bonded_molecule_no
        return np.array([result,  result_periodic])
                #print normal

cdef np.uint8_t apply_symmetry_operation_to_periodicity_axis(atom_no, neighbor_no, new_atom_no, new_neighbor_no, rotation, translation, np.uint8_t periodicity_axis, WaterAlgorithm water_algorithm, debug):
    if rotation == None and translation == None:
        assert periodicity_axis == 13
        return 13 
    scaled_positions = water_algorithm.atoms.get_scaled_positions()
    cdef np.ndarray[np.int8_t, ndim=1] periodicity_vector = get_vector_from_periodicity_axis_number(periodicity_axis)
    real_position = scaled_positions[neighbor_no] + periodicity_vector
    pos = (np.dot(rotation, real_position)) + translation
    nppos = (np.dot(rotation, scaled_positions[neighbor_no])) + translation
    napos = (np.dot(rotation, scaled_positions[atom_no])) + translation
    # NEW periodicity vector is extracted from the rotated and translated real_position a

    cdef np.ndarray[np.double_t, ndim=1] new_periodicity_vector = pos - (scaled_positions[new_neighbor_no]) - (napos - scaled_positions[new_atom_no]) #scaled_positions[new_neighbor_no] - (nppos - scaled_positions[new_neighbor_no])  - (napos - scaled_positions[new_atom_no])   
    #for i in range(3):
    #    if abs(new_periodicity_vector[i]) < 0.99:
    #        new_periodicity_vector[i] = 0.0
    new_number = get_periodicity_axis_number_from_vector(new_periodicity_vector)
    
    #print new_number
    #raw_input()
    if debug : 
        print "********************"
        print napos - scaled_positions[new_atom_no]
        print pos - scaled_positions[new_neighbor_no]
        print periodicity_vector
        print new_periodicity_vector
        print new_number
        print "********************"
    
    return new_number

cdef tuple get_periodic_nn_change_matrix_from_rotation_and_translation(np.ndarray[np.float_t, ndim=2] all_positions, np.ndarray[np.float_t, ndim=2] original_positions, np.ndarray[np.float_t, ndim=2] cell, np.int8_t[::1] pbc, np.int_t[:, :, :] nearest_neighbors_nos, np.ndarray[np.float_t, ndim=2] rotation_matrix, np.ndarray[np.float_t] translation, bint debug):
    cdef float error_tolerance = 0.05
    cdef int i, j, neighbor_no, periodicity_axis, N = nearest_neighbors_nos.shape[1]
    cdef np.ndarray[np.int16_t, ndim = 2] indeces = np.empty((N, 4), dtype=np.int16)
    
    # create indeces table to map the positions with the original
    for i in range(N):
        for j in range(4):
            neighbor_no = nearest_neighbors_nos[0, i, j] 
            periodicity_axis = nearest_neighbors_nos[2, i, j] 
            indeces[i, j] = original_positions.shape[0]
            #print i, j, neighbor_no, periodicity_axis, ": ", indeces[i,j]
            #print all_positions[neighbor_no], (all_positions[i] - original_positions[i])
            original_positions = np.vstack((original_positions, all_positions[neighbor_no] - (all_positions[i] - original_positions[i]) + np.dot(get_vector_from_periodicity_axis_number(periodicity_axis), cell)))
            #else:
            #    indeces[i, j] = neighbor_no   
    # make a copy of the original positions to be freely altered in next step
    cdef np.ndarray[np.float_t] center = np.zeros(3)
    cdef np.ndarray[np.float_t, ndim = 2] original_scaled_positions = get_scaled_positions(cell, original_positions) 
    cdef np.ndarray[np.float_t, ndim = 2] scaled_positions =  rotate_with_matrix(original_scaled_positions, center, rotation_matrix)
    # assumes that that translation is scaled
    scaled_positions += translation

    cdef np.ndarray[np.float_t, ndim = 2] all_scaled_positions = get_scaled_positions(cell, all_positions)
    try:
        return _get_periodic_nn_change_matrix(scaled_positions, original_scaled_positions, all_scaled_positions, cell, pbc, nearest_neighbors_nos, error_tolerance, indeces)
    except:
        print "R:", rotation_matrix 
        print "T:", translation
        import traceback
        traceback.print_exc()
        #print i, nearest_neighbors_nos[0, i, j], nearest_neighbors_nos[2, i, j], "-->", new_atom_no, new_neighbor_no, periodicity_vector_no, scaled_positions[i], all_positions[new_atom_no], scaled_positions[indeces[i, j]], all_positions[new_neighbor_no], original_positions[indeces[i, j]]
        ase.visualize.view(ase.Atoms("O%i"% N, scaled_positions[:N, :], cell = cell))
        raw_input()
        return None

cdef tuple _get_periodic_nn_change_matrix(np.ndarray[np.float_t, ndim = 2] scaled_positions, np.ndarray[np.float_t, ndim = 2] original_scaled_positions, np.ndarray[np.float_t, ndim = 2] all_scaled_positions, np.ndarray[np.float_t, ndim=2] cell, np.int8_t[::1] pbc, np.int_t[:, :, :] nearest_neighbors_nos, float error_tolerance, np.ndarray[np.int16_t, ndim = 2] indeces):
    cdef int i, N = nearest_neighbors_nos.shape[1]
    cdef np.ndarray[np.int8_t, ndim = 2] result = np.empty((N, 4), dtype=np.int8)
    cdef np.ndarray[np.int8_t] molecule_change_matrix = np.empty(N, dtype=np.int8)
    cdef np.ndarray[np.double_t, ndim=1] new_periodicity_vector
    cdef np.int16_t new_neighbor_no, new_atom_no, periodicity_vector_no
    cdef np.ndarray[np.float_t, ndim = 1] pos_neighbor, pos_atom
    for i in range(N):
        pos_atom = scaled_positions[i]
        new_atom_no = atom_at(all_scaled_positions, pbc, pos_atom,  error_tolerance, True)
        molecule_change_matrix[i] = new_atom_no 
        for j in range(4):
            pos_neighbor = scaled_positions[indeces[i, j]]
            new_neighbor_no = atom_at(all_scaled_positions, pbc, pos_neighbor,  error_tolerance, True)
            if new_neighbor_no == -1:
                raw_input("new_neighbor_no == -1")
            new_periodicity_vector = (pos_neighbor - all_scaled_positions[new_neighbor_no]) - (pos_atom - all_scaled_positions[new_atom_no])
            periodicity_vector_no = get_periodicity_axis_number_from_vector(new_periodicity_vector)
            found = False
            for k in range(4):
                if nearest_neighbors_nos[0, new_atom_no, k] == new_neighbor_no and nearest_neighbors_nos[2, new_atom_no, k] == periodicity_vector_no:
                    result[i, j] = k
                    found = True
            if not found:
                print new_periodicity_vector
                print all_scaled_positions  
                raise Exception("SYSTEM ERROR in initializing periodic nn change matrix.")
                
    return result, molecule_change_matrix


    

cdef tuple get_periodic_nn_change_matrix(np.ndarray[np.float_t, ndim=2] all_positions, np.ndarray[np.float_t, ndim=2] original_positions, np.ndarray[np.float_t, ndim=2] cell, np.int8_t[::1] pbc, np.int_t[:, :, :] nearest_neighbors_nos, float angle, np.ndarray[np.float_t] axis, np.ndarray[np.float_t] center, np.ndarray[np.float_t] translation, char* operation_type, bint debug, bint inverse):
    # all_positions map the positions inside the cell
    # original positions can contain positions outside the cell
    cdef float error_tolerance = 0.05
    cdef int i, j, k, neighbor_no, periodicity_axis, N = nearest_neighbors_nos.shape[1]
    cdef np.ndarray[np.int16_t, ndim = 2] indeces = np.empty((N, 4), dtype=np.int16)
    
    # create indeces table to map the positions with the original
    for i in range(N):
        for j in range(4):
            neighbor_no = nearest_neighbors_nos[0, i, j] 
            periodicity_axis = nearest_neighbors_nos[2, i, j] 
            indeces[i, j] = original_positions.shape[0]
            #print i, j, neighbor_no, periodicity_axis, ": ", indeces[i,j]
            #print all_positions[neighbor_no], (all_positions[i] - original_positions[i])
            original_positions = np.vstack((original_positions, all_positions[neighbor_no] - (all_positions[i] - original_positions[i]) + np.dot(get_vector_from_periodicity_axis_number(periodicity_axis), cell)))
            #else:
            #    indeces[i, j] = neighbor_no   
    # make a copy of the original positions to be freely altered in next step
    cdef np.ndarray[np.float_t, ndim=2] positions = original_positions.copy()
    # apply the operation for the positions 
    if inverse:
        if translation is not None:    
            positions -= translation
        if operation_type == <bytes>'S' or operation_type == <bytes>'sigma':
            positions = mirror_through_plane(positions, axis, center, debug)
        if operation_type == <bytes>'S' or operation_type == <bytes>'C':
            positions = rotate(positions, axis, -angle, center)
    else:
        if operation_type == <bytes>'S' or operation_type == <bytes>'C':
            positions = rotate(positions, axis, angle, center)
        if operation_type == <bytes>'S' or operation_type == <bytes>'sigma':
            positions = mirror_through_plane(positions, axis, center, debug)
        if translation is not None:    
            positions += translation
    cdef np.ndarray[np.float_t, ndim = 2] scaled_positions = get_scaled_positions(cell, positions), original_scaled_positions = get_scaled_positions(cell, original_positions), all_scaled_positions = get_scaled_positions(cell, all_positions)

    try:
        return _get_periodic_nn_change_matrix(scaled_positions, original_scaled_positions, all_scaled_positions, cell, pbc, nearest_neighbors_nos, error_tolerance, indeces)
    except:
        print operation_type, angle, center, translation, axis
        #print i, nearest_neighbors_nos[0, i, j], nearest_neighbors_nos[2, i, j], "-->", new_atom_no, new_neighbor_no, periodicity_vector_no, positions[i], all_positions[new_atom_no], positions[indeces[i, j]], all_positions[new_neighbor_no], original_positions[indeces[i, j]]
        ase.visualize.view(ase.Atoms("O%i"% N, positions[:N, :], cell = cell))
        raw_input()
    
       
            
            
    return None, None 

"""cdef np.ndarray[np.int8_t, ndim=1] get_nn_change_for_single_bond(np.ndarray[np.float_t, ndim=2] all_scaled_positions, np.ndarray[np.float_t, ndim=2] original_scaled_positions, np.ndarray[np.float_t, ndim=2] scaled_positions, atom_no, neighbor_no, angle, axis, translation, center, operation_type, np.uint8_t periodicity_axis, atoms, bint debug, bint inverse):    
    error_tolerance = 0.05
   
    cdef np.ndarray[np.float_t, ndim = 1] pos_neighbor = scaled_positions[1]
    cdef np.ndarray[np.float_t, ndim = 1] pos_atom = scaled_positions[0]
    cdef np.int8_t new_neighbor_no = atom_at(atoms, pos_neighbor, max_error=error_tolerance, symbol='O', scaled_positions = True)
    cdef np.int8_t new_atom_no = atom_at(atoms, pos_atom, max_error=error_tolerance, symbol='O', scaled_positions = True)
    cdef np.ndarray[np.double_t, ndim=1] new_periodicity_vector = (pos_neighbor - all_scaled_positions[new_neighbor_no]) - (pos_atom - all_scaled_positions[new_atom_no])
    
    if debug:
        print "Pos_atom, Pos_neighbor: ", pos_atom, pos_neighbor
        print "Minuses:", (pos_neighbor - all_scaled_positions[new_neighbor_no]), (pos_atom - all_scaled_positions[new_atom_no])
        print "Result:", new_periodicity_vector
    natural_basis = np.diag([1, 1, 1])
    #print inverse
    #print rotation
    #print "nos from %i, %i to %i, %i" % (atom_no, neighbor_no, new_atom_no, new_neighbor_no)
    #print "NPV"
    #print new_periodicity_vector
    #new_periodicity_vector = change_basis(new_periodicity_vector, natural_basis, water_algorithm.atoms.get_cell())
    #print new_periodicity_vector
    #if new_neighbor_no == -1:
    #    print translation, angle, axis, center, operation_type
    #    ase.visualize.view(new_atoms)
    #    raw_input()
    return np.array([new_atom_no, new_neighbor_no, get_periodicity_axis_number_from_vector(new_periodicity_vector)], dtype=np.int8)
"""    
    

cdef np.ndarray[np.float_t, ndim = 2] get_scaled_positions(np.ndarray[np.float_t, ndim = 2] cell, np.ndarray[np.float_t, ndim = 2] positions):
    """Get positions relative to unit cell.

        Atoms outside the unit cell will be wrapped into the cell in
        those directions with periodic boundary conditions so that the
        scaled coordinates are between zero and one.
    """
    # this has a potential of going wrong, as apparently some fortran library is called behind the scene
    cdef np.ndarray[np.float_t, ndim = 2] scaled = np.ascontiguousarray(np.linalg.solve(cell.T, positions.T).T, dtype=np.float)
    return scaled

include "../help_methods_cython.pyx"

    
    


