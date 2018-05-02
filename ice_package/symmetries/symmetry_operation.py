import numpy as np

def remove_equals(symmetry_operations, debug=False):
    result = []
    if debug:
        print "Symmetry operations before: %i" % len(symmetry_operations)
    for sym_o in symmetry_operations:
        found_eq = False
        for sym_b in result:
            if sym_b == sym_o:
                if debug:
                    print "%s and %s are equal" % (sym_b, sym_o) 
                found_eq = True
        if not found_eq:
            result.append(sym_o)
    return result

class SymmetryOperation:
    def __init__(self,  name, molecule_change_matrix,  bond_change_matrix,  orientation_change_matrix = None, vector = None, magnitude = 1, type='C', order_no=0, rotation_matrix=None, translation_vector=None):
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
        self.bond_change_matrix = bond_change_matrix
        self.orientation_change_matrix = orientation_change_matrix
        self.vector = vector
        self.magnitude = magnitude
        self.type = type
        self.order_no = order_no
        self.nn_change_matrix = None
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector
        self.has_translate = False
        if self.translation_vector != None and any(self.translation_vector != 0):
            self.has_translate = True
            

    
            

    def calculate_nn_change_matrix(self, nearest_neighbors_nos, water_algorithm, debug = False):
        result = np.zeros((len(nearest_neighbors_nos[0]), len(nearest_neighbors_nos[0][0])), dtype=int)        
        result.fill(-1)
        # Result size should be 20 x 4 in Ice 20:s case (20 molecules and 4 neighbors)
        #  The result consist of a matrix that has the order number of the bond of new_
        if debug:
            print "------------------------------------"
            print self.rotation_matrix
            print self.translation_vector
            print self.molecule_change_matrix
        for i, new_no in enumerate(self.molecule_change_matrix):
            # go through all the nearest neighbors of currently selected molecule
            for j, nn_s in enumerate(nearest_neighbors_nos[0][i]):
                ij_periodic = nearest_neighbors_nos[1][i][j]
                new_axis = apply_symmetry_operation_to_periodicity_axis(i, nn_s, new_no, self.molecule_change_matrix[nn_s], self.rotation_matrix, self.translation_vector, nearest_neighbors_nos[2][i][j], water_algorithm, debug=debug)

                print_str = "Setting (%i - %i)_%i" % (i, nn_s, nearest_neighbors_nos[2][i][j])
                
                #print "Length of nn %i" % len(nearest_neighbors_nos[0][new_no])
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

        self.name += " %s" % result[0]
        return result
            
        
    def __str__(self):
        print self.vector
        if self.vector != None:
            return   "%s%i ***** (%i), %f, %f, %f, %s, %s, %s" % (self.type, self.magnitude, self.order_no, self.vector[0], self.vector[1], self.vector[2], self.molecule_change_matrix, self.nn_change_matrix, self.has_translate)
        else:
            return "%s" % self.molecule_change_matrix 
    
    def __eq__(self, other):
        rotations_agree = (self.rotation_matrix == None and other.rotation_matrix == None) or (self.rotation_matrix == other.rotation_matrix).all() 
        translations_agree = (self.translation_vector == None and other.translation_vector == None) or all(self.translation_vector == other.translation_vector)
        return all(self.molecule_change_matrix == other.molecule_change_matrix) and  rotations_agree and translations_agree
        if self.vector == None and other.vector == None:
            return self.name  == other.name
        else:
            return all(self.molecule_change_matrix == other.molecule_change_matrix)
            diff = np.sum(np.abs(self.vector-other.vector))
            diff2 = np.sum(np.abs(self.vector+other.vector))
            return (diff < 10**-2 or diff2 < 10**-2) and self.order_no == other.order_no and self.type == other.type and (self.magnitude == other.magnitude or (self.magnitude < other.magnitude and float(self.order_no) / float(self.magnitude) == float(other.order_no) / float(other.magnitude)))
    
    def apply(self,  water_orientations,  water_algorithm, nearest_neighbors_nos = None):
        result = np.zeros(len(water_orientations))
        if self.orientation_change_matrix == None:
            if self.bond_change_matrix == None:
                return self.apply_using_nn_change_matrix(water_algorithm, water_orientations, nearest_neighbors_nos)
            else:
                return self.apply_the_hard_way(water_algorithm,  water_orientations)
        for i, water_orientation in enumerate(water_orientations):
            new_no = np.abs(self.molecule_change_matrix[i])
            if new_no > len(water_orientations)-1: #cannot be symmetric
                return None
            s = water_algorithm.get_water_molecule_class(new_no)
            current_orientation = water_orientations[new_no]
            result[i] = self.orientation_change_matrix[s][current_orientation]
        return result

    
    def apply_using_nn_change_matrix(self, water_algorithm, water_orientations, nearest_neighbors_nos):
        # Lazy load for nn change matrix
        if self.nn_change_matrix == None:
            self.nn_change_matrix = self.calculate_nn_change_matrix(nearest_neighbors_nos, water_algorithm)

        result = np.zeros(len(water_orientations))
        for i, water_orientation in enumerate(water_orientations):
            new_no = np.abs(self.molecule_change_matrix[i])
            #print "New no %i vs len(water_orientations): %i " % (new_no,  len(water_orientations)-1)
            if new_no > len(water_orientations)-1: #cannot be symmetric
                return None
            current_orientation = water_orientations[new_no]
            if current_orientation == -1:
                result[i] = -1
                continue
            current_bond_variables = get_bond_variable_values_from_water_orientation(current_orientation)
            new_bond_variables = np.zeros_like(current_bond_variables)
            for j, new_bond in enumerate(self.nn_change_matrix[i]):
                new_bond_variables[j] = current_bond_variables[new_bond]
            #print "new_bond_variables"
            #print new_bond_variables
            #print get_water_orientation_from_bond_variable_values(new_bond_variables)
            result[i] = get_water_orientation_from_bond_variable_values(new_bond_variables)

        return result
        
    def apply_the_hard_way(self, water_algorithm, water_orientations):
        result = np.zeros(len(water_orientations))
        for i, water_orientation in enumerate(water_orientations):
            new_no = np.abs(self.molecule_change_matrix[i])
            #print "New no %i vs len(water_orientations): %i " % (new_no,  len(water_orientations)-1)
            if new_no > len(water_orientations)-1: #cannot be symmetric
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
            
    def apply_to_bond_variables(self, water_algorithm, molecule_no,  current_bond_variables):
        s = water_algorithm.get_water_molecule_class(molecule_no)
        result = np.zeros((len(current_bond_variables)))
        for i in range(len(current_bond_variables)):
            result[i] = current_bond_variables[self.bond_change_matrix[s][i]]
        return result

    
        
        
    def get_symbolic_bond_variable_matrix(self,  nearest_neighbors_nos, water_algorithm):
        if self.bond_change_matrix != None:
            self.symbolic_bond_variable_matrix = get_symbolic_bond_variable_matrix(self.molecule_change_matrix,  self.bond_change_matrix, nearest_neighbors_nos)
        else:
            if self.nn_change_matrix == None:
                self.nn_change_matrix = self.calculate_nn_change_matrix(nearest_neighbors_nos, water_algorithm)
            self.symbolic_bond_variable_matrix = get_symbolic_bond_variable_matrix_from_nn(self.molecule_change_matrix, self.nn_change_matrix, nearest_neighbors_nos)
        return self.symbolic_bond_variable_matrix
        

bvv = np.array([[1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1], [-1, 1, 1, -1], [-1, 1, -1, 1], [-1, -1, 1, 1], [1, -1, -1, -1], [-1, 1, -1, -1], [-1, -1, 1, -1], [-1, -1, -1, 1], [-1, 1, 1, 1], [1, -1, 1, 1], [1, 1, -1, 1], [1, 1, 1, -1]], dtype=np.int8)

def get_bond_variable_values_from_water_orientation(water_orientation):
    if water_orientation == -1:
        return np.array([0, 0, 0, 0], dtype=np.int8)
    else:
        return bvv[water_orientation]

def get_bond_variable_values_from_water_orientations(water_orientations):
    result = []
    for wo in water_orientations:
        result.append(get_bond_variable_values_from_water_orientation(wo))
    return result

def get_water_orientation_from_bond_variable_values(bvvv):
    for i,  bvvr in enumerate(bvv):
        #print bvvr
        #print bvv
        if np.all(bvvr == bvvv):
            return i

def get_symbolic_bond_variable_matrix_from_nn(molecule_change_matrix, nn_change_matrix, nearest_neighbors_nos):
    """
        Result is a dict that contains the changes in bonds

        Each matrix element contains:
            0: The number of the first molecule that is switched to its place
            1: The number of the other molecule switched to its place
            2: Periodicity axis (13 if not periodic)
             

        N is the number of molecules in system        
    """
    molecule_count = len(molecule_change_matrix)
    result = {}
    for molecule_no,  new_molecule_no in enumerate(molecule_change_matrix):
        if molecule_no not in result:
            result[molecule_no] = {}
        for bond_no, new_bond_no in enumerate(nn_change_matrix[molecule_no]):
            neighbor_no = nearest_neighbors_nos[0][molecule_no][bond_no] 
            periodicity_axis = nearest_neighbors_nos[2][molecule_no][bond_no]
            if neighbor_no not in result[molecule_no]:
                result[molecule_no][neighbor_no] = {}
            assert periodicity_axis not in result[molecule_no][neighbor_no]
            nmno = np.abs(new_molecule_no)
            result[molecule_no][neighbor_no][periodicity_axis] = np.array([nmno, nearest_neighbors_nos[0][nmno][new_bond_no], nearest_neighbors_nos[2][nmno][new_bond_no]])


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

def apply_symmetry_operation_to_periodicity_axis(atom_no, neighbor_no, new_atom_no, new_neighbor_no, rotation, translation, periodicity_axis, water_algorithm, debug=False):
    if rotation == None and translation == None:
        assert periodicity_axis == 13
        return 13
    scaled_positions = water_algorithm.atoms.get_scaled_positions()
    periodicity_vector = get_vector_from_periodicity_axis_number(periodicity_axis)
    real_position = scaled_positions[neighbor_no] + periodicity_vector
    pos = (np.dot(rotation, real_position)) + translation
    nppos = (np.dot(rotation, scaled_positions[neighbor_no])) + translation
    napos = (np.dot(rotation, scaled_positions[atom_no])) + translation
    new_periodicity_vector = pos - scaled_positions[new_neighbor_no] - (napos - scaled_positions[new_atom_no])  
    new_number = get_periodicity_axis_number_from_vector(new_periodicity_vector)
    if debug : 
        print "********************"
        print napos - scaled_positions[new_atom_no]
        print pos - scaled_positions[new_neighbor_no]
        print periodicity_vector
        print new_periodicity_vector
        print new_number
        print "********************"
    
    return new_number
    

def get_vector_from_periodicity_axis_number(number):
    a = int(number/9)
    b = int(np.mod(number, 9) /3)
    c = np.mod(number, 3)
    return np.array([a-1, b-1, c-1])

def get_opposite_periodicity_axis_number(number):
    vector = get_vector_from_periodicity_axis_number(number)
    vector *= -1
    return get_periodicity_axis_number_from_vector(vector)

def get_periodicity_axis_number_from_vector(vector):
    a = round(vector[0]) + 1
    b = round(vector[1]) + 1
    c = round(vector[2]) + 1
    return int(round(a*9 + b*3 + c))
