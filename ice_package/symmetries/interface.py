from ase import Atoms
import numpy
from symmetry_operation import SymmetryOperation, remove_equals
import math

class Degeneracy:
    ASYMMETRIC=0
    SYMMETRIC=1
    SPHERICAL=2 
    

def positions_related_to_center_of_mass(atoms, positions):
    com = atoms.get_center_of_mass()
    return positions - com

def get_equals(atoms, new_positions, error_tolerance=0.5, debug=False):
    positions = positions_related_to_center_of_mass(atoms, atoms.get_positions())
    new_positions = positions_related_to_center_of_mass(atoms, new_positions)
    result = numpy.zeros((len(atoms)), dtype=int)
    result.fill(-1) 
    for i, position in enumerate(positions):
        for j, new_position in enumerate(new_positions):
            R = numpy.linalg.norm(position-new_position)
            if R < error_tolerance and atoms[i].get_atomic_number() == atoms[j].get_atomic_number():
                if result[i] == -1:
                    result[i] = j
                else:
                    raise Exception('Error tolerance is too big! (get_equals)')
    if debug:
        print result
    for l in result:
        if l == -1:
            return None
    return result

def get_closest_position(position, new_positions):
    smallest_distance = None
    result = None
    for j, new_position in enumerate(new_positions):
        R = numpy.linalg.norm(position-new_position)
        if smallest_distance == None or R < smallest_distance:
            smallest_distance = R
            result = j
    #print position-new_position
    #print smallest_distance
    return result

def get_closest_positions(positions, new_positions):
    smallest_distance = None
    result = numpy.zeros(len(positions), dtype=int)
    for i, position in enumerate(positions):
        result[i] = get_closest_position(position, new_positions)
    return result



def get_equals_periodic(atoms, new_atoms, error_tolerance=0.03, debug=False):
    result = numpy.zeros((len(atoms)), dtype=int)
    result.fill(-1) 
    for i, position in enumerate(new_atoms.get_scaled_positions()):
        found_atom = atom_at(atoms, position, max_error=error_tolerance, symbol=atoms[i].get_symbol())
        if result[i] == -1:
            result[i] = found_atom
        else:
            raise Exception('Error tolerance is too big! (get_equals)')
    if debug:
        print result
    for l in result:
        if l == -1:
            return None
    return result
                    

def check_inversion_center(atoms):
    positions = positions_related_to_center_of_mass(atoms, atoms.get_positions())
    positions = -positions + atoms.get_center_of_mass()
    result = get_equals(atoms, positions)
    if result == None:
        return None
    print "Inversion center found"
    return SymmetryOperation("i", result, None, type='i', order_no=0, magnitude=1)

def mirror_through_plane(atoms, normal_vector, debug = False):
    L = numpy.linalg.norm(normal_vector)
    assert L < 1.0+10**-6 and L > 1.0-10**-6
    com = atoms.get_center_of_mass()
    # center of mass is along the plane always
    #  calculate the constant parameter d of plane equation 
    d = -(numpy.sum(com*normal_vector))
    result = numpy.zeros_like(atoms.get_positions())
    for i, position in enumerate(atoms.get_positions()):
        # calculate positions distance R from the plane
        # plane normal vector is normalized (sqrt(a^2+b^2+c^2)=1)
        R = numpy.abs(numpy.sum(position*normal_vector)+d)
        # advance - 2 * R along the normal vector (to the other side of the plane)
        result[i] = position - 2*R*normal_vector
        
        R_2 = numpy.abs(numpy.sum(result[i]*normal_vector) + d)
        if debug:
            print "-"
            print result[i]
            print -2*R*normal_vector
            print position
            print d
            print R
            print R_2
        if R > R_2+10**-6 or R < R_2-10**-6:
            result[i] = position + 2*R*normal_vector
            R_3 = numpy.abs(numpy.sum(result[i]*normal_vector) + d)
            assert (R < R_3+10**-6 and R > R_3-10**-6)
    return result

def check_improper_rotation(atoms, vector, angle, center, n, result, order_no=0):
    a = atoms.copy()
    a.rotate(vector, a=angle, center=center)
    positions = mirror_through_plane(a, vector)
    equals = get_equals(atoms, positions)
    on = order_no
    # n = 2 is equal with inversion operator
    if n == 2 or (float(order_no+1) / float(n))  == 0.5:
        return
    if equals != None:
        operation_name = "S%i" % n
        while order_no > 0:
            operation_name += "'"
            order_no -= 1
        result.append(SymmetryOperation(operation_name, equals, None, vector = vector, magnitude = n, type='S', order_no=on))

def check_improper_rotations(atoms, rotation_operations):
    """
        Improper rotational axes must be coincident with existing 
        proper rotational axes and have a degree either equal to
        or twice the degree of coincident proper rotational axis
    """
    result = []

    
    com = atoms.get_center_of_mass()
    for operation in rotation_operations:
        vector = operation.vector.copy()
        L = numpy.linalg.norm(vector)
        vector /= L
        angle = (2*math.pi)/operation.magnitude
        n = operation.magnitude
        center = com  

        # one times the proper rotation
        for order_no in range(n-1):
            check_improper_rotation(atoms, vector, (order_no+1)*angle, center, n, result, order_no=order_no)
        
        # rotations twice the proper rotation
        angle = angle/2
        for order_no in range(2*n-1):
            check_improper_rotation(atoms, vector, (order_no+1)*angle, center, 2*n, result, order_no=order_no)

    result = remove_equals(result)
    print "%i improper rotations found" % len(result)

    return result

def check_single_rotation(atoms, vector, angle, center, n, result, order_no=0):
    a = atoms.copy()
    a.rotate(vector, a=angle, center=center)
    positions = a.get_positions()
    equals = get_equals(atoms, positions)
    on = order_no
    if equals != None:
        operation_name = "C%i" % n
        while order_no > 0:
            operation_name += "'"
            order_no -= 1
        result.append(SymmetryOperation(operation_name, equals, None, vector = vector, magnitude = n, type='C', order_no=on))

def check_rotation(atoms, vector):
    com = atoms.get_center_of_mass()
    # normalize vector
    L = numpy.linalg.norm(vector)
    vector /= L
    result = []
    n = 6
    while n >= 2:
        angle = (2*math.pi)/n
        for order_no in range(n-1):
            check_single_rotation(atoms, vector, (order_no+1)*angle, com, n, result, order_no=order_no)
        n -= 1

    return result
    

def find_symmetry_operations(atoms):
    if any(atoms.pbc) == True:
        return get_periodic_symmetry_operations(atoms)
    else: 
        moms, vecs = check_symmetry(atoms)
        result = []
        rotations = check_proper_rotational_axes(atoms, vecs)

        improper_rotations = check_improper_rotations(atoms, rotations)
        mirrors = check_mirrors(atoms, rotations, vecs)
        inversion = check_inversion_center(atoms)
        if inversion != None:
            result.append(inversion)
        result.extend(rotations)
        result.extend(improper_rotations)
        result.extend(mirrors)
        result.append(get_identity_operator(atoms))
        print_symmetry_operations(result)
        raw_input()
        return result

def print_symmetry_operations(symmetry_operations):
    for symmetry_operation in symmetry_operations:
        print symmetry_operation

def get_identity_operator(atoms):
    equals = numpy.arange(0, len(atoms), 1)
    return SymmetryOperation("Identity", equals, None)

def check_proper_rotational_axes(atoms, vectors_of_inertia):
    result = []
    
    result.extend(remove_equals(check_rotations_along_vectors(atoms, vectors_of_inertia)))
    result.extend(remove_equals(check_com_atom_rotational_axes(atoms)))
    result = remove_equals(result)
    result.extend(remove_equals(check_com_atom_mid_rotational_axes(atoms)))
    result = remove_equals(result)
    result.extend(remove_equals(check_rotations_along_vectors(atoms, vectors_of_inertia)))
    result = remove_equals(result)
    result.extend(remove_equals(check_rotations_along_symmetry_operations(atoms, result, only_magnitude=2)))
    result = remove_equals(result)
    result.extend(remove_equals(check_rotations_along_cross_products_of_symmetry_operations(atoms, result, only_magnitude=2)))
    result = remove_equals(result)
    print "%i proper rotations found" % len(result)
    return result

    

def check_com_atom_rotational_axes(atoms):
    result = []
    com = atoms.get_center_of_mass()
    for atom in atoms:
        result.extend(check_rotation(atoms, atom.get_position()-com))
    return result

def check_com_atom_mid_rotational_axes(atoms):
    result = []
    com = atoms.get_center_of_mass()
    for atom_1 in atoms:
        for atom_2 in atoms:
            if atom_1.get_atomic_number() == atom_2.get_atomic_number():
                p = (atom_1.get_position() + atom_2.get_position())/2
                result.extend(check_rotation(atoms, p-com))
    return result

def check_rotations_along_symmetry_operations(atoms, operations, only_magnitude=-1):
    result = []
    for operation in operations:
        if only_magnitude == -1 or operation.magnitude == only_magnitude:
            result.extend(check_rotation(atoms, operation.vector))
    return result

def check_rotations_along_cross_products_of_symmetry_operations(atoms, operations, only_magnitude=-1):
    result = []
    for i, operation in enumerate(operations):
        for l, operation2 in enumerate(operations):
            if i != l and (only_magnitude == -1 or (operation.magnitude == only_magnitude and operation2.magnitude == only_magnitude)):
                vector = numpy.cross(operation.vector, operation2.vector)
                result.extend(check_rotation(atoms, vector))
    return result

def check_rotations_along_vectors(atoms, vectors):
    result = []
    for vector in vectors:
        result.extend(check_rotation(atoms, vector))
    return result

def check_mirror(atoms, normal_vector, debug=False):
    com = atoms.get_center_of_mass()
    # normalize vector
    L = numpy.linalg.norm(normal_vector)
    normal_vector /= L
    mirrored = mirror_through_plane(atoms, normal_vector, debug = debug)
    if debug:
        print normal_vector
        print atoms.get_positions()
        print mirrored
        from ice_package.help_methods import get_oxygens
        import ase
        ase.visualize.view(get_oxygens(mirrored))
        ase.visualize.view(atoms)
        print normal_vector
        raw_input("Continue")
    result = get_equals(atoms, mirrored, debug=debug)
    if result == None:
        return None
    return SymmetryOperation('sigma', result, None, vector = normal_vector, magnitude = 1, order_no = 0, type='sigma')

def check_mirrors(atoms, rotation_operations, inertia_axis):
    result = []
    result.extend(check_mirrors_from_previously_found(atoms, rotation_operations, inertia_axis))
    result.extend(check_com_atom_mirrors(atoms))
    result.extend(check_com_atom_mid_mirrors(atoms))
    result = remove_equals(result)
    print "%i mirrors found" % len(result)
    return result

def check_mirrors_from_previously_found(atoms, rotation_operations, inertia_axis):
    result = []
    for rotation_operation in rotation_operations:
        mirror = check_mirror(atoms, rotation_operation.vector)
        if mirror != None:
            result.append(mirror)
        for axis in inertia_axis:
            vector = numpy.cross(rotation_operation.vector, axis)
            if any(vector != numpy.zeros_like(vector)):
                mirror = check_mirror(atoms, vector)
                if mirror != None:
                    result.append(mirror)
    return result

def check_com_atom_mirrors(atoms):
    result = []
    com = atoms.get_center_of_mass()
    for atom in atoms:
        if any(atom.get_position() != com): 
            mirror = check_mirror(atoms, atom.get_position()-com)
            if mirror != None:
                result.append(mirror)
    return result

def check_com_atom_mid_mirrors(atoms, debug=False):
    result = []
    com = atoms.get_center_of_mass()
    for atom_1 in atoms:
        for atom_2 in atoms:
            if atom_1.get_atomic_number() == atom_2.get_atomic_number():
                p = (atom_1.get_position() + atom_2.get_position())/2
                if any(p != com): 
                    if debug:
                        print com[2]
                        print p[2]
                        print atom_1.get_position()[2]
                    mirror = check_mirror(atoms, p-com, debug=debug)
                    if mirror != None:
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
        
    positions = positions_related_to_center_of_mass(atoms, atoms.get_positions())
    masses = atoms.get_masses()
    #initialize elements of the inertial tensor
    I11 = I22 = I33 = I12 = I13 = I23 = 0.0
    for i in range(len(atoms)):
        x, y, z = positions[i]
        m = masses[i]
        I11 += m * (y**2 + z**2)
        I22 += m * (x**2 + z**2)
        I33 += m * (x**2 + y**2)
        I12 += -m * x * y
        I13 += -m * x * z
        I23 += -m * y * z

    I = numpy.array([[I11, I12, I13],
                  [I12, I22, I23],
                  [I13, I23, I33]])

    evals, evecs = numpy.linalg.eig(I)
    return evals, evecs

## PYSPGLIB

def get_periodic_symmetry_group(atoms, error_tolerance = 0.01, debug = False):
    from pyspglib import spglib
    print atoms
    dataset = spglib.get_symmetry_dataset(atoms, symprec=error_tolerance)
    print dataset['equivalent_atoms']
    print dataset['international']
    print dataset['hall']
    print dataset['wyckoffs']
    print dataset['transformation_matrix']
    print "Number of symmetry operations %i"  % len(dataset['rotations'])
    return dataset['international']

def get_periodic_symmetry_operations(atoms, error_tolerance=0.01, debug=False):
    from ase.utils import irotate
    from ase.visualize import view
    from pyspglib import spglib
    #symmetry = spglib.get_symmetry(atoms, symprec=1e-5)
    symmetry = spglib.get_symmetry(atoms, symprec=1e-2)
    dataset = spglib.get_symmetry_dataset(atoms, symprec=1e-2)
    
    result = []
    if debug:
        cell, scaled_positions, numbers = spglib.find_primitive(atoms, symprec=1e-5)
        a = Atoms( symbols='O4',
                 cell=cell,
                 scaled_positions=scaled_positions,
                 pbc=True)
        #symmetry = spglib.get_symmetry(a, symprec=1e-2)
        #dataset = spglib.get_symmetry_dataset(a, symprec=1e-2)
        print dataset['equivalent_atoms']
        print dataset['international']
        print dataset['hall']
        print dataset['wyckoffs']
        print dataset['transformation_matrix']
        print "Number of symmetry operations %i"  % len(dataset['rotations'])
    for i in range(dataset['rotations'].shape[0]):
        
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
       
        
        for l, pos in enumerate(new_atoms.get_scaled_positions()):
            #print new_pos[l]
            new_pos[l] = (numpy.dot(rot, pos))
            new_pos[l] += trans
            #print new_pos[l]
        new_atoms.set_scaled_positions(new_pos)
        equals = get_equals_periodic(atoms, new_atoms, error_tolerance=error_tolerance, debug=debug)
        if equals != None:
            so = SymmetryOperation(str(i), equals, None, vector = None, magnitude = 1, rotation_matrix=rot, translation_vector=trans)
            #if debug:
            #    print so
            result.append(so)
        else:
            print "Equivalent not found"
            #view(test)
            #view(new_atoms)
            #raw_input()

    return result
        
        


### Space group determination

def get_lattice_system(lattice_vectors):
    assert len(lattice_vectors) == 3
    a = numpy.linalg.norm(lattice_vectors[0])
    b = numpy.linalg.norm(lattice_vectors[1])
    c = numpy.linalg.norm(lattice_vectors[2])
    alpha = numpy.arccos(numpy.dot(lattice_vectors[0], lattice_vectors[1])/ (a*b))
    beta = numpy.arccos(numpy.dot(lattice_vectors[0], lattice_vectors[2]) / (a*c))
    gamma = numpy.arccos(numpy.dot(lattice_vectors[1], lattice_vectors[2]) / (b*c))
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
    elif (alpha == beta and alpha == math.pi/2.0 and gamma == 2*math.pi/3.0) or (alpha == gamma and alpha == math.pi/2.0 and beta == 2*math.pi/3.0) or (beta == gamma and beta == math.pi/2.0 and alpha == 2*math.pi/3.0):
        assert (a == b or a == c or b == c) and (a != b or b != c or a != c)
        return 5 # HEXAGONAL 

    elif (alpha == beta and alpha == math.pi/2.0 and gamma != 2*math.pi/2.0) or (alpha == gamma and alpha == math.pi/2.0 and beta != 2*math.pi/2.0) or (beta == gamma and beta == math.pi/2.0 and alpha != 2*math.pi/2.0):
        assert (a != b and a != c and b != c)
        return 1 # MONOCLINIC
    else:
        assert a != b and b != c and a != c and alpha != beta and beta != gamma and alpha != gamma
        return 0 #TRICLINIC

def get_lattice_centering(origo, atoms, lattice_system, lattice_vectors, cell):
    """
    """
    if lattice_system == 0 or lattice_system == 2 or lattice_system == 5:
        return 0 # PRIMITIVE (P)
    else:
        p1 = (origo + lattice_vectors[0]/2.0 + lattice_vectors[1]/2.0)
        p2 = (origo + lattice_vectors[0]/2.0 + lattice_vectors[2]/2.0)
        p3 = (origo + lattice_vectors[0]/2.0 + lattice_vectors[2]/2.0)
        p4 = (origo + lattice_vectors[0]/2.0 + lattice_vectors[1]/2.0 + lattice_vectors[2]/2.0)
        if atom_at(p4, cell) != -1:
            return 1 # BODY CENTERED (I)
        elif atom_at(p1, cell) != -1 and atom_at(p2, cell) != -1 and atom_at(p3, cell) != -1:
            return 2 # FACE CENTERED (F) 
        elif atom_at(p1, cell) != -1:
            return 3 # A
        elif atom_at(p2, cell) != -1:
            return 4 # B
        elif atom_at(p3, cell) != -1:
            return 5 # C
        else:
            return 0 # PRIMITIVE (P)
        

def atom_at(atoms, point, max_error=0.01, symbol=None):
    """
        Checks if there is an atom at point
        point is a relative position
    """
    real_point = get_real_point(point)
    #print point
    #print real_point
    for i, position in enumerate(atoms.get_scaled_positions()):
        R = numpy.linalg.norm(position-real_point)
        #print R
        if abs(R) < max_error and (symbol == None or symbol == atoms.get_chemical_symbols()[i]):
            return i
    return -1

def get_real_positions(relative_positions):
    result = numpy.zeros_like(relative_positions)
    for i, position in enumerate(relative_positions):
        result[i] = get_real_point(position)
    return result

def get_rotation_matrix(axis, angle):
    axis = axis/numpy.sqrt(numpy.dot(axis,axis))
    a = numpy.cos(angle/2)
    b,c,d = -axis*numpy.sin(angle/2)
    return numpy.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

def get_angle_between_vectors(vector1, vector2):
    l1 = numpy.linalg.norm(vector1)
    l2 = numpy.linalg.norm(vector2)
    return numpy.arccos(numpy.dot(vector1, vector2) / (l1*l2))    


    

        

def get_real_point(point):
    assert len(point) == 3
    result = point.copy()
    rec = numpy.zeros(3, dtype=int)
    for i, l in enumerate(result):
        
        while result[i] < 0:
            rec[i] += 1
            result[i] += 1
        while result[i] >= 1:
            rec[i] -= 1
            result[i] -= 1
    #print "Relocated point %s" % rec
    return result

    
    
        
        
    
