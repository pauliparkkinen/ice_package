#cython: wraparound=False
#cython: nonecheck=False
#cython: infer_types=True

cimport cython
import numpy as np
cimport numpy as np
from help_methods import get_oxygens, get_periodic_distance, get_periodic_distances, get_oxygen_indeces, get_only_oxygens
import math

 
def get_water_orientations_from_bond_variables_and_molecule_change_matrix(foreign_nearest_neighbors_nos, nearest_neighbors_nos, np.int8_t[:, ::1] bond_variables,  equals = None):
    """
        fits the bond variables of foreign nearest neighbors nos (probably obtained from 'get_bond_varibles_from_atoms' of structure_commons.py) to 
        nearest_neighbors_nos and gets the bond_variables
    """
    cdef np.int8_t[::1] bvv
    cdef int j, k, molecule_no, N = bond_variables.shape[0], M = bond_variables.shape[1]
    cdef np.int8_t[::1] bond_variable_set
    cdef np.ndarray[np.int8_t, ndim = 1] result = np.empty(N, dtype=np.int8)
    if N != nearest_neighbors_nos.shape[1]:
        raise Exception("The structure has a wrong number of oxygens (has %i, should have %i). As a result, this cannot be a proton configuration of the handled oxygen raft." % (N, nearest_neighbors_nos.shape[1]))
    for molecule_no in range(N):
        bvv = np.array([-1, -1, -1, -1], dtype=np.int8)
        bond_variable_set = np.zeros(M, dtype=np.int8)
        for j, new_molecule_no in enumerate(foreign_nearest_neighbors_nos[0, molecule_no]):
            for k, tryout_new_molecule_no in enumerate(nearest_neighbors_nos[0, molecule_no]):
                if new_molecule_no == tryout_new_molecule_no and foreign_nearest_neighbors_nos[2, molecule_no, j] == nearest_neighbors_nos[2, molecule_no, k] and bond_variable_set[k] == 0:
                    bond_variable_set[k] = 1
                    bvv[k] = bond_variables[molecule_no, j]
                    break
        for k, is_set in enumerate(bond_variable_set):
            if is_set == 0:
                print "NNs for molecule %i:" % molecule_no
                print nearest_neighbors_nos[0, molecule_no], nearest_neighbors_nos[2, molecule_no]
                print foreign_nearest_neighbors_nos[0, molecule_no], foreign_nearest_neighbors_nos[2, molecule_no]
                raise Exception("Local nearest neighbor %i-%i does not have correspondent in foreign nearest neighbors. This could be due to significant altering of the oxygen structure." % (molecule_no, nearest_neighbors_nos[0, molecule_no, k]))
                    
        result[molecule_no] = get_water_orientation_from_bond_variable_values(bvv)
        if result[molecule_no] > 5:
            print "Molecule %i" % molecule_no
            print bvv.base
            print bond_variable_set[0], bond_variable_set[1], bond_variable_set[2], bond_variable_set[3]  
            print bond_variables.base[molecule_no]
            print foreign_nearest_neighbors_nos[0, molecule_no]
            raw_input()
 
    return result

cpdef list add_periodic_neighbors(p1, p2, cell, min_distance, max_distance, p2_number, result, periodicities, periodicity_axis, count, sortlist):
    distances, axis = get_periodic_distances(p1, p2, cell)
    cdef np.ndarray[np.int8_t, ndim=1] vec
    for distance, p_ax in zip(distances, axis):
        if distance > min_distance and distance < max_distance and p_ax != 13:
            result = np.append(result, p2_number)
            periodicities = np.append(periodicities, True)
            periodicity_axis.append(p_ax)
            vec = get_vector_from_periodicity_axis_number(p_ax)
            sortlist = np.vstack((sortlist,  p2+vec*cell))
            count += 1
    return [result, periodicities, periodicity_axis, count, sortlist]

@cython.boundscheck(False)
cdef inline void add_periodic_neighbors2(np.float_t[::1] p1, np.float_t[::1] p2, np.float_t[:, ::1] cell, float min_distance, float max_distance, int p1_number, int p2_number, np.int_t[:, :, ::1] result, int *count):
    cdef np.float_t[:, ::1] distances = periodic_euler_distances(p1, p2, cell)
    cdef float distance
    cdef int i, N = distances.shape[0], p_ax
    for i in range(N):
        distance = distances[i, 0]
        p_ax = int(distances[i, 1])
        if distance > min_distance and distance < max_distance and p_ax != 13:
            result[0, p1_number, count[0]] = p2_number
            result[1, p1_number, count[0]] = 1
            result[2, p1_number, count[0]] = p_ax
            count[0] += 1

cpdef np.int_t[::1] get_coordination_numbers(np.int_t[:, :, ::1] nearest_neighbors_nos):
    cdef int N = nearest_neighbors_nos.shape[1], M = 4
    cdef np.int_t[::1] result = np.empty(N, dtype=np.int)
    cdef int molecule_type, molecule_no, j, coordination_number
    for molecule_no from 0 <= molecule_no < N:
        coordination_number = 0
        for bond_no from 0 <= bond_no < M:
            if nearest_neighbors_nos[0, molecule_no, bond_no] != molecule_no or nearest_neighbors_nos[2, molecule_no, bond_no] != 13:
                coordination_number += 1
        result[molecule_no] = coordination_number
    return result

@cython.boundscheck(False)
cdef np.float_t[::1] get_single_molecule_dipole_moment(unsigned int site, np.int8_t water_orientation, unsigned int i, np.float_t[:, ::1] oxygen_positions, np.int_t[::1] nearest_neighbors_nos, np.int_t[::1] nn_periodicity, np.int_t[::1] nn_periodicity_axis, np.float_t O_H_distance, np.float_t[:, ::1] cell):
    cdef np.float_t[:, ::1] h_coordinates = get_single_molecule_hydrogen_coordinates(site, water_orientation, i, oxygen_positions, nearest_neighbors_nos, nn_periodicity, nn_periodicity_axis,  cell,  O_H_distance) 
    cdef np.float_t[::1] h_coordinate,  oxygen_position = oxygen_positions[i]
    cdef np.float_t[::1] dipole_moment = np.zeros(3, dtype=np.float)
    cdef unsigned int j, N = h_coordinates.shape[0]
  
    for j from 0 <= j < N:
        add_to_dipole_moment_estimate(h_coordinates[j], oxygen_position, O_H_distance, dipole_moment)
            
    return dipole_moment

@cython.cdivision(True)
cdef inline void add_to_dipole_moment_estimate(np.float_t[::1] h_coordinate, np.float_t[::1] o_coordinate, np.float_t O_H_distance, np.float_t[::1] dipole_moment):
    cdef int i, N = 3
    for i from 0 <= i < N:
        dipole_moment[i] += ( h_coordinate[i] - o_coordinate[i] ) / (O_H_distance * 2)

@cython.boundscheck(False)
cpdef np.ndarray[np.float_t, ndim=2] get_single_molecule_hydrogen_coordinates(unsigned int coordination_number, np.int8_t water_orientation, unsigned int i, np.float_t[:, ::1] oxygen_positions,  np.int_t[::1] nearest_neighbors_nos, np.int_t[::1] nn_periodicity, np.int_t[::1] periodicity_axis, np.float_t[:, ::1] cell, np.float_t O_H_distance):
    cdef np.int8_t[::1] bvv = get_bond_variable_values_from_water_orientation(water_orientation)
    cdef np.float_t[:, ::1] result
       
    cdef np.float_t[::1] add, real_position, oxygen_position = oxygen_positions.base[i], neighbor_oxygen_position, com
    cdef np.float_t[::1] vector, normal
    cdef np.ndarray[np.float_t, ndim = 1] position
    cdef np.float_t vector_length, normal_length
    cdef np.float_t distance
    cdef int n, counter = 0, N = nearest_neighbors_nos.shape[0], db_count = 0, index = 0, x
    coordination_number = 0
    if water_orientation == -1:
        result = np.zeros((0, 3), dtype=np.float)
        return result.base
    if water_orientation < 6:
        result = np.zeros((2, 3), dtype=np.float)
    elif water_orientation > 9:
        result = np.zeros((3, 3), dtype=np.float)  
    else:
        result = np.zeros((1, 3), dtype=np.float)    

    # determine dangling bond count, coordination_number
    for n from 0 <= n < N:
        x  = nearest_neighbors_nos[n]
        if bvv[n] == 1 and i == x and not nn_periodicity[n]:
            db_count += 1
        if i != x or nn_periodicity[n]:
            coordination_number += 1 
            
    for n from 0 <= n < N:
        x  = nearest_neighbors_nos[n] 
        if bvv[n] == 1:
            position = get_single_bond_hydrogen_coordinates(oxygen_position, oxygen_positions[x], i, x, periodicity_axis[n], oxygen_positions, nearest_neighbors_nos, periodicity_axis, &counter, db_count, coordination_number, O_H_distance, cell)
            result[index, 0] = position[0]
            result[index, 1] = position[1]
            result[index, 2] = position[2]
            index += 1
    return result.base 
    
@cython.boundscheck(False)
cpdef np.ndarray[np.float_t, ndim=3] get_selector_hydrogen_coordinates(np.int_t[::1] coordination_numbers, np.float_t[:, ::1] oxygen_positions, np.int_t[:, :, ::1] nearest_neighbors_nos, np.float_t[:, ::1] cell, np.float_t O_H_distance, dict preset_bond_values):
    cdef int donor_molecule_number, acceptor_molecule_number, periodicity_axis, i,  N = nearest_neighbors_nos.shape[1], dangling_bond_counter
    cdef dict molecule_preset_bond_values, pair_preset_bond_values
    cdef bint allowed
    cdef np.float_t[:, :, ::1] result = np.zeros((N, 4, 3), dtype = np.float)  
            
    for donor_molecule_number in range(N):
        if preset_bond_values is not None and donor_molecule_number in preset_bond_values:
            molecule_preset_bond_values = preset_bond_values[donor_molecule_number]
        else:
            molecule_preset_bond_values = None
        for i in range(4):
            # if this bond is set to -1, then write nan's instead of the possible location
            dangling_bond_counter = 0
            allowed = True
            acceptor_molecule_number = nearest_neighbors_nos[0, donor_molecule_number, i]
            
            # check if the bond is allowed
            if molecule_preset_bond_values is not None and acceptor_molecule_number in molecule_preset_bond_values:
                pair_preset_bond_values = molecule_preset_bond_values[acceptor_molecule_number]
                if periodicity_axis in pair_preset_bond_values and pair_preset_bond_values[periodicity_axis] == -1:
                    allowed = False
                else:
                    allowed = True
                    
            # do the writing to the array    
            if allowed:
                result.base[donor_molecule_number, i] = get_single_bond_hydrogen_coordinates(oxygen_positions[donor_molecule_number], oxygen_positions[acceptor_molecule_number], donor_molecule_number, acceptor_molecule_number, nearest_neighbors_nos[2, donor_molecule_number, i], oxygen_positions, nearest_neighbors_nos[0, donor_molecule_number], nearest_neighbors_nos[2, donor_molecule_number], &dangling_bond_counter, 4-coordination_numbers[donor_molecule_number], coordination_numbers[donor_molecule_number], O_H_distance, cell)
            else:
                result[donor_molecule_number, i, 0] = np.nan
                result[donor_molecule_number, i, 1] = np.nan
                result[donor_molecule_number, i, 2] = np.nan
                    
    return result.base 
 
@cython.boundscheck(False)
cdef np.ndarray[np.float_t, ndim=1] get_single_bond_hydrogen_coordinates(np.float_t[::1] donor_oxygen_position, np.float_t[::1] acceptor_oxygen_position, int donor_molecule_number, int acceptor_molecule_number, int periodicity_axis, np.float_t[:, ::1] oxygen_positions,  np.int_t[::1] nearest_neighbors_nos, np.int_t[::1] periodicity_axises, int  *dangling_bond_counter, int dangling_bond_count, int coordination_number, np.float_t O_H_distance, np.float_t[:, ::1] cell):
    cdef np.float_t[::1] add, real_position, com
    cdef np.float_t[::1] vector, normal
    cdef np.float_t vector_length, normal_length, distance
    cdef bint periodic = periodicity_axis != 13
    cdef bint dangling_bond = not periodic and donor_molecule_number == acceptor_molecule_number
    
    if not dangling_bond:
        # the bond is a normal hydrogen bond between two molecules
        if periodic:
            distance,  real_position = get_periodic_distance(donor_oxygen_position, acceptor_oxygen_position, cell, periodicity_axis)
        else:
            distance = euler_norm(reduce(donor_oxygen_position, acceptor_oxygen_position))
            real_position = acceptor_oxygen_position
        # oxygen_positions -  (O_H_distance * (oxygen_positions - real_position) / distance)
        return reduce(donor_oxygen_position, divide(multiply(O_H_distance, reduce(donor_oxygen_position, real_position)), distance)).base

    else:
        # the bond is a  dangling bond
        dangling_bond_counter[0] += 1
        if dangling_bond_count == 2 or (coordination_number == 1 and dangling_bond_count == 1):
            if coordination_number == 0:
                vector = np.array([0, 1, 0], dtype=np.float)
                normal = np.array([1, 0, 0], dtype=np.float)
            else:
                # get center of mass of nearest neighbors
                com = get_nn_com(donor_molecule_number, nearest_neighbors_nos, oxygen_positions, periodicity_axises,  cell)
                       
                vector = reduce(donor_oxygen_position, com)
                # get normal for the plane defined by nearest neighbors
                normal = get_nn_normal(donor_molecule_number, coordination_number, nearest_neighbors_nos, oxygen_positions, periodicity_axises, cell)
                # normalize the vector  
                vector_length = euler_norm(vector)
                vector = divide(vector, vector_length)
                # normalize the normal  
                normal_length = euler_norm(normal)
                normal = divide(normal, normal_length)
                     
            #if db_count == 2:  
            # multiply with tan(52.2*2.0*math.pi/360.) to get the correct angle 104.4 between dangling bonds
            if dangling_bond_count == 2:
                normal = multiply(tan(52.2*2.0*math.pi/360.), normal)
            else:
                normal = multiply(tan(75.6*2.0*math.pi/360.), normal)
                normal = multiply(-1.0, normal)
            # multiply with tan((180-52.2)*2.0*math.pi/360.)
            #normal = multiply(math.tan((360-104.4)*2.0*math.pi/360.), normal)
                    
            if dangling_bond_counter[0] == 2:
                # if counter is 2 let's go to negative direction along the normal
                normal = multiply(-1.0, normal)
            # add vector and normal together and normalize   
            vector = add_together(vector, normal)
            vector_length = euler_norm(vector)
            vector = divide(vector, vector_length)
            multiply(O_H_distance, vector)
        else:
            # get center of mass of nearest neighbors
            com = get_nn_com(donor_molecule_number, nearest_neighbors_nos, oxygen_positions, periodicity_axises,  cell)
            vector = reduce(donor_oxygen_position, com)
            # normalize the vector  
            vector_length = euler_norm(vector)
                        
            vector = divide(vector, vector_length)
        # the dangling hydrogen is along this vector
        return add_together(donor_oxygen_position, multiply(O_H_distance, vector)).base
        

@cython.boundscheck(False)
cpdef np.ndarray[np.float_t, ndim=1] get_total_dipole_moments(np.int8_t[:, ::1] wos, np.float_t[:, ::1] oxygen_coordinates, np.int_t[:, :, ::1] nearest_neighbors_nos, np.float_t O_H_distance, np.float_t[:, ::1]  cell):
    cdef np.int8_t[::1] wo
    cdef unsigned int i, N = wos.shape[0]
    cdef np.float_t[:, ::1] total_dipole_moments = np.ndarray((N, 3), dtype=np.float)
    for i from 0 <= i < N:
        wo = wos[i]
        total_dipole_moments[i] = get_total_dipole_moment(wo, oxygen_coordinates, nearest_neighbors_nos, O_H_distance, cell)
        #if i % 1000 == 0:
        #    print "Cython %i / %i" % (i, N)
    return total_dipole_moments.base
      
@cython.boundscheck(False)      
cpdef np.float_t[::1] get_total_dipole_moment(np.int8_t[::1] water_orientations, oxygen_coordinates, np.int_t[:, :, ::1] nearest_neighbors_nos, np.float_t O_H_distance, np.float_t[:, ::1]  cell):
    cdef np.float_t[::1] total_dipole_moment = np.array([0, 0, 0], dtype=np.float)
    cdef unsigned int i, site, N = water_orientations.shape[0]
    cdef unsigned int water_orientation
    cdef np.float_t[:, ::1] oxygen_positions = oxygen_coordinates
    for i from 0 <= i < N:
        water_orientation = water_orientations[i]
        site = i % 4
        total_dipole_moment = add_together(total_dipole_moment, get_single_molecule_dipole_moment(site, water_orientation, i, oxygen_positions, nearest_neighbors_nos[0, i], nearest_neighbors_nos[1, i], nearest_neighbors_nos[2, i], O_H_distance, cell))
    return total_dipole_moment

cdef np.ndarray[np.float_t, ndim=1] get_nn_com(int molecule_no, np.int_t[::1] nearest_neighbors_nos, np.float_t[:, ::1] oxygen_positions, np.int_t[::1] nn_periodicity_axis, np.float_t[:, ::1] cell):
    cdef list com_coordinates = []
    cdef int n, N = nearest_neighbors_nos.shape[0], x
    for n from 0 <= n < N:
        x = nearest_neighbors_nos[n]
        if molecule_no != x:
            if nn_periodicity_axis[n] == 13:
                com_coordinates.append(oxygen_positions.base[x])
            else:
                position = oxygen_positions.base[x] + np.dot(cell, get_vector_from_periodicity_axis_number(nn_periodicity_axis[n]))
                com_coordinates.append(position) 
            
    return get_oxygens(np.array(com_coordinates)).get_center_of_mass()  

cdef np.ndarray[np.float_t, ndim=1] get_nn_normal(int molecule_no, int coordination_number, np.int_t[::1] nearest_neighbors_nos, np.float_t[:, ::1] oxygen_positions, np.int_t[::1] nn_periodicity_axis, np.float_t[:, ::1] cell):
    cdef np.ndarray[np.float_t, ndim = 2] vectors = np.zeros((coordination_number, 3), dtype=float)
    cdef int n, N = nearest_neighbors_nos.shape[0], counter = 0, x
    
    for n from 0 <= n < N: 
        x = nearest_neighbors_nos[n]
        if molecule_no != x:
            if nn_periodicity_axis[n] == 13:
                vectors[counter] = oxygen_positions.base[x] - oxygen_positions.base[molecule_no]
            else:
                vector = oxygen_positions.base[x] + np.dot(cell, get_vector_from_periodicity_axis_number(nn_periodicity_axis[n])) - oxygen_positions.base[molecule_no]
                vectors[counter] = vector
            counter += 1
    if coordination_number == 2:  
        return np.cross(vectors[0], vectors[1])
    else: # coordination number = 1 
        return random_orthogonal(vectors[0])

def random_orthogonal(original):
    if original[0] == 0.0:
        return np.array([1., 0., 0.])
    elif original[0] != 0.0 and original[1] == 0.0:
        return np.array([0., 1., 0.])
    else:
        return np.array([-original[1], original[0], 0.0])

def find_nearest_neighbors_nos(atom_positions, O_O_distance, periodic=False, cell=None):
    if type(periodic) == list or type(periodic) == np.ndarray:
        periodic = any(periodic)
    return _all_nearest_neighbors_no(atom_positions, O_O_distance, periodic, cell).base

@cython.boundscheck(False)
cdef np.int_t[:, :, ::1] _all_nearest_neighbors_no(np.float_t[:, ::1] atom_positions, float O_O_distance, bint periodic, np.float_t[:, ::1] cell):
    cdef int i, count = 0, N = atom_positions.shape[0]
    cdef np.int_t[:, :, ::1] result = np.empty((3, N, 4), dtype=np.int_)
    result[:] = -1
    
    for i in range(N):
        nearest_neighbors_no(O_O_distance,  i,  atom_positions[i], atom_positions, periodic, cell, result)
    add_dangling_bonds_to_nearest_neighbors(result)
    return result

@cython.boundscheck(False)
cdef inline void add_dangling_bonds_to_nearest_neighbors(np.int_t[:, :, ::1] nearest_neighbors_nos):
    cdef int N = nearest_neighbors_nos.shape[1]
    for i in range(N):
        for j in range(4):
            if nearest_neighbors_nos[0, i, j] == -1:
                nearest_neighbors_nos[0, i, j] = i
                nearest_neighbors_nos[1, i, j] = 0
                nearest_neighbors_nos[2, i, j] = 13

@cython.boundscheck(False)
cdef inline void nearest_neighbors_no(float O_O_distance,  int atom_no,  np.float_t[::1] atom_position, np.float_t[:, ::1] atom_positions, bint periodic, np.float_t[:, ::1] cell, np.int_t[:, :, ::1] nearest_neighbors_nos):
    cdef int i, N = atom_positions.shape[0], count = 0
    cdef float min_distance = O_O_distance * 0.85
    cdef float max_distance = O_O_distance * 1.15
    cdef float distance

    for i in range(N):
        distance = euler_distance(atom_position,  atom_positions[i])
        if (distance > min_distance and distance < max_distance):
            nearest_neighbors_nos[0, atom_no, count] = i
            nearest_neighbors_nos[1, atom_no, count] = 0
            nearest_neighbors_nos[2, atom_no, count] = 13
            count = count + 1
        if periodic:
            add_periodic_neighbors2(atom_position, atom_positions[i], cell, min_distance, max_distance, atom_no, i, nearest_neighbors_nos, &count) 
     
@cython.boundscheck(False)       
cpdef tuple get_bond_variables_from_atoms(atoms, float O_O_distance = 2.7, float O_H_distance = 1.0, bint debug = False):  
    """
        Parses nearest neighbors nos, bond variables and the elongated bonds of the ice structure
    """  
    oxygens = get_only_oxygens(atoms)
    cdef np.float_t[:, ::1] oxygen_positions = oxygens.get_positions(), positions = atoms.get_positions()
    cdef int[::1] oxygen_indeces = np.array(get_oxygen_indeces(atoms), dtype=int)
    
    
    cdef np.float_t smallest_distance, distance
    cdef int smallest_distance_oxygen, axis_to_closest_oxygen, i, j, k, l, N_oxygens = oxygen_positions.shape[0], N = positions.shape[0], oxygen_index, second_closest_oxygen, axis_to_second_closest_oxygen, axis, neighbor_no, hydrogen_no, N_hydrogens, oxygen_index_2
    periodic = atoms.get_pbc()
    cdef np.float_t[:, ::1] cell = atoms.get_cell()
    cdef np.int_t[::1] atomic_numbers = atoms.get_atomic_numbers()
    cdef np.int_t[:, ::1] nearest_hydrogens = np.empty((N_oxygens, 4), dtype=np.int_),  nearest_hydrogens_axis = np.empty((N_oxygens, 4), dtype=np.int_)
    nearest_hydrogens[:] = -1
    nearest_hydrogens_axis[:] = -1
    cdef bint seek_periodic = (type(periodic) == list and 1 in periodic) or (type(periodic) == np.ndarray and 1 in periodic) or (type(periodic) == bool and periodic)
    cdef list nearest_hydrogens_o
    cdef bint elongated_bonds = False
    cdef bint elongated_hydrogen_bonds = False
    cdef bint allow_nonperiodic
    cdef np.int_t[:, :, ::1] nearest_neighbors_nos
    cdef np.int_t[:, ::1] nearest_neighbors_hydrogens
    cdef np.int8_t[:, ::1] bond_variables
    cdef np.float_t[:, ::1] periodic_distances
    
    # Find the oxygens that are closest to each hydrogen and store the closest hydrogens indeces for
    # each oxygen atom
    for i in range(N):
        if atomic_numbers[i] == 1: # 'Hydrogen'
            smallest_distance = 0.0
            smallest_distance_oxygen = -1
            axis_to_closest_oxygen = -1
            # go through all oxygens and find the closest oxygen to the handled hydrogen
            for j in range(N_oxygens):
                allow_nonperiodic = i != j
                _get_smallest_distance_and_axis(positions[i], oxygen_positions[j], cell, seek_periodic, allow_nonperiodic, &distance, &axis)
                if i != j  and (smallest_distance_oxygen == -1 or distance < smallest_distance):
                    smallest_distance = distance
                    smallest_distance_oxygen = j
                    axis_to_closest_oxygen = 13
                            
            # check if the bond is considered elongated
            if smallest_distance > 1.15:
                elongated_bonds = True
            # store for the smallest_distance oxygen
            for j in range(4):
                if nearest_hydrogens[smallest_distance_oxygen, j] == -1:
                    nearest_hydrogens[smallest_distance_oxygen, j]  = i
                    nearest_hydrogens_axis[smallest_distance_oxygen, j] = axis_to_closest_oxygen
                    break

    # Get the bond variables by first finding the second closest oxygen to the closest hydrogen,
    # i.e., by finding the nearest neighbors nos
    nearest_neighbors_nos = _all_nearest_neighbors_no(oxygen_positions, O_O_distance, seek_periodic, cell)
    
    # indeces for hydrogens in the atoms object
    nearest_neighbors_hydrogens = np.empty((N_oxygens, 4), dtype=np.int_)
    nearest_neighbors_hydrogens[:] = -1
    bond_variables = np.empty((N_oxygens, 4), dtype=np.int8)
    bond_variables[:] = 0
    # find the second closest oxygens, i.e., acceptors for bonds
    for i in range(N_oxygens):
        oxygen_index = oxygen_indeces[i]
        # Find the second closest oxygens:
        for l in range(4):
            hydrogen_no = nearest_hydrogens[i, l]
            if hydrogen_no != -1:
                axis_to_closest_oxygen = nearest_hydrogens_axis[i, l]    
                smallest_distance = 0.0 # to second closest oxygen
                second_closest_oxygen = -1
                axis_to_second_closest_oxygen = -1
                
                for j in range(4):
                    neighbor_no = nearest_neighbors_nos[0, i, j]
                    # check that we are not comparing oxygen with itself 
                    allow_nonperiodic = i != neighbor_no
                    _get_smallest_distance_and_axis(positions[hydrogen_no], oxygen_positions[neighbor_no], cell, seek_periodic, allow_nonperiodic, &distance, &axis)
                    if (i != neighbor_no or axis != axis_to_closest_oxygen) and (second_closest_oxygen == -1 or distance < smallest_distance):
                        axis_to_second_closest_oxygen = nearest_neighbors_nos[2, i, j]
                        smallest_distance = distance
                        second_closest_oxygen = j
                
                # If there is a second smallest distance oxygen
                if second_closest_oxygen != -1:
                    if smallest_distance < 2.3:
                        if smallest_distance > 0.74 * O_O_distance:
                            elongated_hydrogen_bonds = True
                            if debug: 
                                print "Second closest oxygen is at %f Ang. LC is %f. O-H is %f" % (smallest_distance, O_O_distance, O_H_distance)
                        oxygen_index_2 = nearest_neighbors_nos[0, i, second_closest_oxygen]
                        bond_variables[i, second_closest_oxygen] = 1
                        nearest_neighbors_hydrogens[i, second_closest_oxygen] = hydrogen_no
                        
                        # set the opposite bond
                        for j in range(nearest_neighbors_nos.shape[2]):
                            neighbor_no = nearest_neighbors_nos[0, oxygen_index_2, j]
                            axis = nearest_neighbors_nos[2, oxygen_index_2, j]
                            if neighbor_no == i and axis == get_opposite_periodicity_axis_number(axis_to_second_closest_oxygen):
                                bond_variables[oxygen_index_2, j] = -1
                                break
                        # IMPORTANT: the oxygen periodicity must be found again, as the hydrogen bonded to donor molecule
                        # can be a periodic image in the structure
                        #allow_nonperiodic = i != second_closest_oxygen
                        #_get_smallest_distance_and_axis(oxygen_positions[i], oxygen_positions[second_closest_oxygen], cell, seek_periodic, allow_nonperiodic, &distance, &axis)
                        # Set the bond
                        #  -Iterate over the nearest neighbors until empty one (-1) is found
                        #for j in range(nearest_neighbors_nos.shape[2]):
                        #    neighbor_no = nearest_neighbors_nos[0, i, j]
                        #    if neighbor_no == -1:
                        #        nearest_neighbors_nos[0, i, j] = second_closest_oxygen
                        #        nearest_neighbors_nos[2, i, j] = axis
                        #        nearest_neighbors_nos[1, i, j] = axis != 13
                        #        nearest_neighbors_hydrogens[i, j] = hydrogen_no
                        #        bond_variables[i, j] = 1
                        #        break
                        # Set the opposite bond:
                        #  -Iterate over the nearest neighbors of second closest oxygen until empty one (-1) is found
                        #for j in range(nearest_neighbors_nos.shape[2]):
                        #    neighbor_no = nearest_neighbors_nos[0, second_closest_oxygen, j]
                        #    if neighbor_no == -1:
                        #        nearest_neighbors_nos[0, second_closest_oxygen, j] = i 
                        #        nearest_neighbors_nos[2, second_closest_oxygen, j] = get_opposite_periodicity_axis_number(axis)
                        #        nearest_neighbors_nos[1, second_closest_oxygen, j] = nearest_neighbors_nos[2, second_closest_oxygen, j] != 13
                        #        bond_variables[second_closest_oxygen, j] = -1
                        #        break
                    else:  # No second closest oxygen found -> dangling hydrogen bond
                        for j in range(nearest_neighbors_nos.shape[2]):
                            neighbor_no = nearest_neighbors_nos[0, i, j]
                            axis = nearest_neighbors_nos[2, i, j]
                            if neighbor_no == i and axis == 13 and bond_variables[i, j] == 0:
                                nearest_neighbors_hydrogens[i, j] = hydrogen_no
                                bond_variables[i, j] = 1
                                break
                        #for j in range(nearest_neighbors_nos.shape[2]):
                        #    neighbor_no = nearest_neighbors_nos[0, i, j]
                        #    if neighbor_no == -1:
                        #        nearest_neighbors_nos[0, i, j] = i
                        #        nearest_neighbors_nos[1, i, j] = 0
                        #        nearest_neighbors_nos[2, i, j] = 13
                        #        nearest_neighbors_hydrogens[i, j] = hydrogen_no
                        #        bond_variables[i, j] = 1
                        #        break
     
    # Include dangling oxygen bonds to nearest_neighbors_nos
    for i in range(bond_variables.shape[0]):
        for j in range(bond_variables.shape[1]):
            if bond_variables[i, j] == 0:
                bond_variables[i, j] = -1
                

    
    return nearest_neighbors_nos, bond_variables, elongated_bonds, elongated_hydrogen_bonds, nearest_neighbors_hydrogens, oxygen_indeces

@cython.boundscheck(False)
cdef inline void _get_smallest_distance_and_axis(np.float_t[::1] position1, np.float_t[::1] position2, np.float_t[:, ::1] cell, bint seek_periodic, bint allow_nonperiodic, np.float_t *smallest_distance, int *smallest_distance_axis):
    """
        Sets the smallest distance between two positions to variable smallest_distance, and the corresponding periodicity axis to smallest_distance_axis
    """
    cdef np.float_t[:, ::1] periodic_distances
    cdef int axis
    if not seek_periodic:
        smallest_distance[0] = euler_distance(position1, position2)
        smallest_distance_axis[0] = 13 
    else:   
        periodic_distances = periodic_euler_distances(position1, position2, cell)                  
        for k in range(periodic_distances.shape[0]):
            axis = int(periodic_distances[k, 1])
            if (allow_nonperiodic or axis != 13) and (k == 0 or periodic_distances[k, 0] < smallest_distance[0]):
                smallest_distance_axis[0] = axis
                smallest_distance[0] = periodic_distances[k, 0]

include "help_methods_cython.pyx"
