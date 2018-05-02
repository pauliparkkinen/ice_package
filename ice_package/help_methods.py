import numpy as np
import math, ase
from symmetries.symmetry_operation import get_bond_variable_values_from_water_orientation, remove_equals, get_vector_from_periodicity_axis_number, get_opposite_periodicity_axis_number

bvv = np.array([[1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1], [-1, 1, 1, -1], [-1, 1, -1, 1], [-1, -1, 1, 1], [1, -1, -1, -1], [-1, 1, -1, -1], [-1, -1, 1, -1], [-1, -1, -1, 1], [-1, 1, 1, 1], [1, -1, 1, 1], [1, 1, -1, 1], [1, 1, 1, -1]], dtype=np.int8)

def get_water_orientation_from_bond_variable_values(bvvv):
    N = bvv.shape[0]
    M = bvvv.shape[0]  
    for i in range(N):
        equals = True
        for j in range(M):
            if bvv[i, j] != bvvv[j]:
                equals = False
                break
        if equals:
            return i

def get_water_orientations_from_bond_variable_values(bond_variables):
    N = bond_variables.shape[0]
    result = np.zeros(N, dtype=np.int8)
    for i in range(N):
        result[i] = get_water_orientation_from_bond_variable_values(bond_variables[i])
    return result

def remove_hydrogens(atoms):
    new_atoms = ase.Atoms(cell=atoms.cell, pbc = atoms.pbc)
    for atom in atoms:
        if atom.get_symbol() != 'H':
            new_atoms.append(atom)
    return new_atoms

def get_only_oxygens(atoms):
    new_atoms = ase.Atoms(cell=atoms.cell, pbc = atoms.pbc)
    for atom in atoms:
        if atom.get_symbol() == 'O':
            new_atoms.append(atom)
    return new_atoms

def remove_all_except(atoms, indeces):
    """
       Removes all atoms except atoms with indeces in indeces
        -used in hydrogen bond charge transfer calculations
    """
    result = atoms.copy()
    start = -1
    end = len(atoms) - 1
    # Iterate from end to start so that all the right atoms are removed
    for i in range(end, start, -1):
        if i not in indeces:
            result.pop(i = i)
    return result
    
def remove_indeces(atoms, indeces):
    """
       Removes atoms with indeces in indeces
        -used in hydrogen bonding energetic calculations
    """
    result = atoms.copy()
    start = -1
    end = len(atoms) - 1
    # Iterate from end to start so that all the right atoms are removed
    for i in range(end, start, -1):
        if i in indeces:
            result.pop(i = i)
    return result


def get_oxygen_indeces(atoms):
    result = []
    for i, atom in enumerate(atoms):
        if atom.get_symbol() == 'O':
            result.append(i)
    return result

def rotate(atoms, rotation_matrix):
    new_atoms = atoms.copy()
    new_positions = np.zeros(atoms.get_positions().shape)
    com = atoms.get_center_of_mass()
    pos = atoms.get_positions() - com
    for i, position in enumerate(pos):
        new_positions[i] = np.dot(rotation_matrix, position)
    new_atoms.set_positions(new_positions)
    return new_atoms 

def normalize(vector):
    result = np.array(vector)
    result /= np.linalg.norm(result)
    return result

def rotate_to(atoms, vector_1, vector_2):
    """
        Rotate atoms so that vector_1 and vector_2 are equal
    """
    vectorr_1 = normalize(vector_1.copy())
    vectorr_2 = normalize(vector_2.copy())
    vector_3 = np.cross(vectorr_1, vectorr_2)
    vector_4 = np.cross(vector_3, vectorr_1)
    M1 = [vectorr_1, vector_4, vector_3]

    cos = np.dot(vectorr_2, vectorr_1)
    sin = np.dot(vectorr_2, vector_4)
       
    M2 = np.array([[ cos,   sin,    0 ],
                   [ -sin,  cos,    0 ],
                   [ 0,     0,      1 ]])
    rotation_matrix = np.dot(np.dot(np.linalg.inv(M1), M2), M1)
    return rotate(atoms, rotation_matrix)

def get_coordinates_in_basis(positions, basis):
    new_positions = np.zeros_like(positions)
    invbasis = np.linalg.inv(basis)
    for i, position in enumerate(positions):
        new_positions[i] = np.dot(invbasis, position)
    return new_positions    

def are_equal(number1, number2, error_tolerance=0.1):
    return number2 > number1 -error_tolerance and  number2 < number1 + error_tolerance

def get_average_oxygen_distance(oxygen_positions, nearest_neighbors_nos):
    total = 0.0
    count = 0
    for i in range(len(nearest_neighbors_nos)):
        for j in range(4):
            if i != nearest_neighbors_nos[0, i, j] and nearest_neighbors_nos[2, i, j] == 13: 
                total += abs(np.linalg.norm(oxygen_positions[i] -  oxygen_positions[nearest_neighbors_nos[0, i, j]]))
                count += 1
    return float(total) / float(count)
    
def get_oxygens(oxygen_positions, periodic = False, slab = False, cell = None):
    from ase import Atom,  Atoms
    
    atoms = []
    for i in range(len(oxygen_positions)):
        O = Atom('O',(oxygen_positions[i][0],oxygen_positions[i][1], oxygen_positions[i][2]))
        atoms.append(O)
    atoms = Atoms(atoms)  
    atoms.pbc = periodic
    if cell != None:
        atoms.set_cell(cell)
    return atoms


def get_hydrogens(hydrogen_positions):
    from ase import Atom,  Atoms
    atoms = []
    print hydrogen_positions
    for i in range(len(hydrogen_positions)):
        H = Atom('H',(hydrogen_positions[i][0], hydrogen_positions[i][1], hydrogen_positions[i][2]))
        atoms.append(H)
    return Atoms(atoms)

def merge(list):
    result = []
    for item in list:
        result.extend(item)
    return result

def merge_water_orientations(list):
    result = None
    for item in list:
        if result == None:
            result = item
        else:
            result = np.vstack((result, item))
    return result
        
def split_list(alist, wanted_parts=1):
    if alist is None:
        return None
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

def get_periodic_distances(p1, p2, cell):
    result = []
    axis = []
    for i in range(3):
        a = i - 1
        for j in range(3):
            b = j - 1
            for k in range(3):
                c = k - 1
                px = p2 + (cell[0] * a + cell[1] * b + cell[2] * c)
                p_ax = 9*i + 3*j + k
                distance = get_distance(p1, px)
                result.append(distance)
                axis.append(p_ax)
    return result, axis
    

def get_periodic_distance(p1, p2, cell, periodicity_axis):
    px = p2 + np.dot(get_vector_from_periodicity_axis_number(periodicity_axis), cell)
    distance = get_distance(p1, px)
    return distance, px



def print_(text, file = None):
    if file is not None:
        print >> file, text

def rmse(true_values, predictions):
    assert len(true_values) == len(predictions) 
    return  np.sum(np.sqrt((true_values - predictions)**2)) / len(true_values)

def get_distance(p1, p2, periodic=False, cell=None):
    distance = math.sqrt(math.pow(p1[0]-p2[0], 2)+math.pow(p1[1]-p2[1], 2)+math.pow(p1[2]-p2[2], 2))
    if periodic:
        periodic_distance,  real_position = get_periodic_distance(p1,p2,cell)
        return min([distance, periodic_distance])
    else:
        return distance

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def change_basis_for_vectors(vectors, original_basis, new_basis):
    result = np.empty_like(vectors) 
    change_of_basis_matrix_1 = original_basis.T # from original basis  to natural basis 
    change_of_basis_matrix_2 = np.linalg.inv(new_basis.T) # from natural basis to new basis
    for i, vector in enumerate(vectors):
        result[i] = np.dot(change_of_basis_matrix_1, vector)
        result[i] = np.dot(result[i], change_of_basis_matrix_2) 
    return result

def change_basis_for_transformation_matrix(transformation_matrix, original_basis, new_basis = None):
    # let's go from original basis to natural basis
    change_of_basis_matrix = original_basis.T
    inv_change_of_basis_matrix = np.linalg.inv(change_of_basis_matrix)
    natural_basis_transformation_matrix = np.dot(change_of_basis_matrix, np.dot(transformation_matrix, inv_change_of_basis_matrix))
    # if the new basis is none, we will return transformation to natural basis
    if new_basis is None:
        return natural_basis_transformation_matrix
    # if not, we will go from     
    else:
        change_of_basis_matrix = new_basis.T
        inv_change_of_basis_matrix = np.linalg.inv(change_of_basis_matrix)
        return np.dot(inv_change_of_basis_matrix, np.dot(natural_basis_transformation_matrix, change_of_basis_matrix))

def change_basis(vector, original_basis, new_basis = None):
    change_of_basis_matrix_1 = original_basis.T # from original basis  to natural basis 
    if new_basis != None:
        change_of_basis_matrix_2 = np.linalg.inv(new_basis.T) # from natural basis to new basis
    else:
        change_of_basis_matrix_2 = None
    result = np.dot(change_of_basis_matrix_1, vector)
    if new_basis == None:
        return result
    else:
        return np.dot(change_of_basis_matrix_2, result)
    
def rotation_matrix(axis,theta):
    axis = axis/np.sqrt(np.dot(axis,axis))
    a = np.cos(theta/2)
    b,c,d = -axis*np.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

def reflection_matrix_through_origin(plane_vector):
    a = plane_vector[0]
    b = plane_vector[1]
    c = plane_vector[2]
    return np.array([[1 - 2 * a * a,   -2 * a * b,   - 2 * a * c ],
                     [  - 2 * a * b, 1 -2 * b * b,   - 2 * b * c ],
                     [   -2 * a * c,   -2 * b * c, 1 - 2 * c * c ]])

def improper_rotation_matrix(axis, theta):
    return np.dot(rotation_matrix(axis, theta), reflection_matrix_through_origin(axis))

def apply_symmetry_operation_to_coordinates(coordinates, rotation_matrix, translation_vector = None, centering = None):
    new_pos = np.zeros_like(coordinates)
    for j, pos in enumerate(coordinates):
        if centering is not None:
            new_pos[j] = np.dot(rotation_matrix, pos+centering)
        else:
            new_pos[j] = np.dot(rotation_matrix, pos)
        if translation_vector is not None:
            new_pos[j] += translation_vector
        if centering is not None:
            new_pos[j] -= centering
    return new_pos

def get_rotations(max_rotation, axis):
    angles = []
    for i in range(0, max_rotation):
        angles.append(i*2*math.pi/max_rotation)
    result = []
    for angle in angles:
        result.append(rotation_matrix(axis, angle))
    return result

def get_main_axis_mirrors(max_rotation):
    angles = []
    for i in range(0, max_rotation):
        angles.append(i*math.pi/max_rotation)
    result = []
    for angle in angles:
        result.append(reflection_matrix_through_origin([math.cos(angle), math.sin(angle), 0]))
    return result

def get_side_axis_rotations(max_rotation):
    angles = []
    for i in range(0, max_rotation):
        angles.append(i*math.pi/max_rotation)
    result = []
    for angle in angles:
        result.append(rotation_matrix([math.cos(angle), math.sin(angle), 0], math.pi))
    return result

def get_d6h_symmetry_operations(glide = True):
    result = []
    pure_rotations = get_rotations(6, [0, 0, 1])
    main_axis_mirrors = get_main_axis_mirrors(6)
    side_axis_rotations = get_side_axis_rotations(6)
    horizontal_mirror = reflection_matrix_through_origin([0, 0, 1])
    translations = []
    transformation_matrices = []
    for i, pure_rotation in enumerate(pure_rotations):
        if glide and i % 2 == 1:
            translations.append([0.0, 0.0, 0.5])
        else:
            translations.append([0.0, 0.0, 0.0])
        transformation_matrices.append(pure_rotation)

    # improper rotations (including inversion and horizontal mirror)
    improper_translations = []
    improper_matrices = []
    for i, translation in enumerate(translations):
        transformation_matrix = transformation_matrices[i]
        improper_matrices.append(np.dot(transformation_matrix, horizontal_mirror))
        improper_translations.append(translation)
    translations.extend(improper_translations)
    transformation_matrices.extend(improper_matrices)
    # side axis rotations, 0 is through two atoms, 1 is between atoms
    for i, side_rotation in enumerate(side_axis_rotations):
        if glide and i % 2 == 0:
            translations.append([0.0, 0.0, 0.5])
        else:
            translations.append([0.0, 0.0, 0.0])
        transformation_matrices.append(side_rotation)
    # Main axis mirrors side axis rotations, 0 is through two atoms, 1 is between atoms
    for i, mirror in enumerate(main_axis_mirrors):
        if glide and i % 2 == 1:
            translations.append([0.0, 0.0, 0.5])
        else:
            translations.append([0.0, 0.0, 0.0])
        transformation_matrices.append(mirror)
    return transformation_matrices, translations
    
def get_equals_periodic(atoms, new_atoms, error_tolerance, debug, periodic_pos):
    result = np.zeros((len(atoms)), dtype=int)
    inverse_result = np.zeros((len(atoms)), dtype=int)
    result.fill(-1) 
    inverse_result.fill(-1)
    for i, position in enumerate(new_atoms.get_scaled_positions()):
        found_atom = atom_at(atoms, position, max_error=error_tolerance, symbol=atoms[i].get_symbol())
        found_atom_inverse = inverse_atom_at(new_atoms, atoms.get_scaled_positions()[i], max_error=error_tolerance, symbol=atoms[i].get_symbol(), periodic_pos = periodic_pos)
        if result[i] == -1:
            result[i] = found_atom
        else:
            raise Exception('Error tolerance is too big! (get_equals)')
        if inverse_result[i] == -1:
            inverse_result[i] = found_atom_inverse
        else:
            raise Exception('Error tolerance is too big! (Inverse @ get_equals)')
    if debug:
        print result
    for l in result:
        if l == -1:
            return None, None
    #view(new_atoms)
    #view(atoms)
    #print result, inverse_result
    #raw_input()
    return result, inverse_result

def atom_at(atoms, point, max_error=0.01, symbol=None, scaled_positions = True):
    """
        Checks if there is an atom at point
        point is a relative position
    """
    
    #print point
    #print real_point
    if scaled_positions:
        positions = atoms.get_scaled_positions()
        real_point = get_real_point(point, None)
    else:
        positions = atoms.get_positions()
        real_point = get_real_point(point, atoms.get_cell())
    for i, position in enumerate(positions):
        R = np.linalg.norm(position-real_point)
        #print R
        if abs(R) < max_error and (symbol == None or symbol == atoms.get_chemical_symbols()[i]):
            return i
    return -1

def get_real_positions(relative_positions):
    result = np.zeros_like(relative_positions)
    for i, position in enumerate(relative_positions):
        result[i] = get_real_point(position)
    return result

def get_real_point(point, cell = None):
    assert len(point) == 3
    result = point.copy()
    if cell is not None:
        natural_basis = np.diag([1, 1, 1])
        result = change_basis(result, natural_basis, cell)
    rec = np.zeros(3, dtype=int)
    for i, l in enumerate(result):
        
        while result[i] < 0:
            rec[i] += 1.0
            result[i] += 1.0
        while result[i] >= 1:
            rec[i] -= 1.0
            result[i] -= 1.0
    #print "Relocated point %s" % rec
    if cell is not None:
        result = change_basis(result, cell)
    return result

def inverse_atom_at(atoms, point, max_error=0.01, symbol=None, periodic_pos = None):
    """
        Checks if there is an atom at point
        point is a relative position
    """
    real_positions = get_real_positions(atoms.get_scaled_positions())
    
    #print point
    #print real_point
    for i, position in enumerate(real_positions):
        R = np.linalg.norm(position-point)
        #print R
        if abs(R) < max_error and (symbol == None or symbol == atoms.get_chemical_symbols()[i]):
            return i
    if periodic_pos is not None:
        for positions in periodic_pos:
            for i, position in enumerate(get_real_positions(positions)):
                R = np.linalg.norm(position-point)
                #print R
                if abs(R) < max_error and (symbol == None or symbol == atoms.get_chemical_symbols()[i]):
                    return i
    
    return -1

def handle_periodic_input(periodic):
    if type(periodic) == list or type(periodic) == bool:
        return periodic
    else:
        if periodic == 'x-wire':
            return [True, False, False]
        elif periodic == 'y-wire':
            return [False, True, False]
        elif periodic == 'z-wire':
            return [False, False, True]
        elif periodic == 'slab':
            return [True, True, False]
        elif periodic == 'bulk':
            return True
        else:
            return False



   
"""
from ice_surfaces import *
import ase
from ase.visualize import view
#from symmetries.interface_cython import get_equals_periodic



rotations, translations = get_d6h_symmetry_operations()
cell = get_hexagonal_cell(1, 1, 1, 2.7, True, False, False)
coords = get_hexagonal_ice_coordinates(1, 1, 1, 2.7, orthogonal_cell = True, slab=False, misc = False)     
a = ase.Atoms("O%i" % 8, positions=coords, cell=cell)                 
atoms = a.copy()
error_tolerance = 0.1
centering = np.array([-1.102, -(7.649+2.558)/2, 0])
debug = False
for i, rot in enumerate(rotations):

    new_pos = apply_symmetry_operation_to_coordinates(a.get_positions(), rot, np.dot(translations[i],cell), centering) 
    atoms.set_positions(new_pos)
    equals, inverse = get_equals_periodic(a, atoms, error_tolerance, debug, None)
    print equals
    if equals == None:
        print i
        print rot
        print translations[i]
        view(atoms)
        raw_input()
    #view(atoms)
    #raw_input()"""

