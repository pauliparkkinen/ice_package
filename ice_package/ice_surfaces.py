import numpy as np, math
from collections import OrderedDict

ice_types = OrderedDict()
ice_types["ice_ih"] = "Ice Ih (hexagonal ice)"
ice_types["ice_ih_orth"] =  "Ice Ih (hexagonal ice): Orthorhombic cell"
ice_types["ice_ic"] =  "Ice Ic (cubic ice)"
#ice_types["ice_ii"] =  "Ice II"
#ice_types["ice_iii"] = "Ice III"
#ice_types["ice_iv"] =  "Ice IV"
#ice_types["ice_v"] =   "Ice V"
#ice_types["ice_vi"] =  "Ice VI"
#ice_types["ice_vii"] = "Ice VII"
#ice_types["ice_viii"] = "Ice VIII"
#ice_types["ice_ix"] =  "Ice IX"
#ice_types["ice_x"] = "Ice X"
#ice_types["ice_xii"] = "Ice XII"
#ice_types["ice_xiii"] = "Ice XIII"
#ice_types["ice_xiv"] = "Ice XIV"

cell_lengths = {"ice_ih": [4.4, 4.4975, 7.3224], 
                "ice_ic": [6.358, 6.358, 6.358],
                "ice_ii": [7.78, 7.78, 7.78],
                "ice_iii":[6.73, 6.73, 6.73],
                "ice_iv": [7.6, 7.6, 7.6],
                "ice_v":  [9.22, 7.54, 10.35],
                "ice_vi": [6.27, 6.27, 5.29],
                "ice_vii":[3.3, 3.3, 3.3],
                "ice_viii":[4.449, 4.449, 6.413],
                "ice_ix": [6.73, 6.73, 6.73],
                "ice_x":  [2.78, 2.78, 2.78],
                "ice_xii":[8.304, 8.304, 4.024],
                "ice_xiii":[9.2417, 7.4724, 10.297],
                "ice_xiv":[8.3499, 8.1391, 4.0825],
                "ice_xv" :[6.2323, 6.2438, 5.7903]}

cell_angles =  {"ice_ih": [90., 90., 120.], 
                "ice_ic": [90., 90., 90.],
                "ice_ii": [113.1, 113.1, 113.1],
                "ice_iii":[90., 90., 90.],
                "ice_iv": [70.1, 70.1, 70.1],
                "ice_v":  [90., 109.2, 90.],
                "ice_vi": [90., 90., 90.],
                "ice_vii":[90., 90., 90.],
                "ice_viii":[90., 90., 90.],
                "ice_ix": [90., 90., 90.],
                "ice_x":  [90., 90., 90.],
                "ice_xii":[90., 90., 90.],
                "ice_xiii":[90., 109.6873, 90.],
                "ice_xiv":[90., 90., 90.],
                "ice_xv" :[90.06, 89.99, 89.92]}

atom_positions = {"ice_ic": np.array([[0., 0., 0.], 
                                      [0., 2., 2.],
                                      [2., 0., 2.],
                                      [2., 2., 0.],
                                      [3., 3., 3.],
                                      [3., 1., 1.],
                                      [1., 3., 1.],
                                      [1., 1., 3.]]) / 4.0 + 0.01
                 }

def get_cell(ice_type, n_x = 1, n_y = 1, n_z = 1, orthogonal_cell = False, slab = False):
    if ice_type == "ice_ih":
        return get_hexagonal_cell(n_x, n_y, n_z, O_O_distance = 2.7, orthogonal_cell = False, slab = slab)
    if ice_type == "ice_ih_orth":
        return get_hexagonal_cell(n_x, n_y, n_z, O_O_distance = 2.7, orthogonal_cell = True, slab = slab)
    else:
        min_cell_lengths = cell_lengths[ice_type]
        min_cell_angles = cell_angles[ice_type]
        cell_vectors = np.zeros((3, 3), dtype=np.float)
        # initialize the radian values for lattice angles
        alpha = min_cell_angles[0] * math.pi / 180.
        beta = min_cell_angles[1] * math.pi / 180.
        gamma = min_cell_angles[2] * math.pi / 180.

        # cell vector a (= 0) is always in the x direction
        a = cell_vectors[0]
        b = cell_vectors[1]
        c = cell_vectors[2]
        # 
        a[0] = 1.0

        # cell vector b is always in the x-y plane, because of the a choice (gamma is the angle between a and b) 
        # i.e., z component of b is 0
        b[0] = math.cos( gamma )
        b[1] = math.sqrt(1 - b[0]*b[0])

        # determine c using the previous info as starting point
        c[0] = math.cos( beta )
        c[1] = (math.cos( alpha ) - math.cos( beta ) * math.cos( gamma )) / b[1]
        c[2] = math.sqrt(1 - c[0]*c[0] - c[1]*c[1])   
        
        
        cell_vectors[0] *= min_cell_lengths[0] * n_x
        cell_vectors[1] *= min_cell_lengths[1] * n_y
        cell_vectors[2] *= min_cell_lengths[2] * n_z
        return cell_vectors

def get_oxygen_coordinates(ice_type, n_x = 1, n_y = 1, n_z = 1, O_O_distance = 2.7, orthogonal_cell = False, slab=False, misc = False):
    if ice_type == "ice_ih":
        return get_hexagonal_ice_coordinates(n_x, n_y, n_z, O_O_distance, orthogonal_cell = False, slab = slab, misc = False)
    if ice_type == "ice_ih_orth":
        return get_hexagonal_ice_coordinates(n_x, n_y, n_z, O_O_distance, orthogonal_cell = True, slab = slab, misc = False)
    else:
        # get the minimum_cell
        minimum_cell = get_cell(ice_type, n_x = 1, n_y = 1, n_z = 1, orthogonal_cell = orthogonal_cell, slab = slab)
        # get the relative minimum cell coordinates
        minimum_cell_atom_positions = np.dot(atom_positions[ice_type], minimum_cell)
        minimum_cell_atom_count = minimum_cell_atom_positions.shape[0]

        # initialize the result array
        result = np.zeros((n_x * n_y * n_z * minimum_cell_atom_count, 3))
        for z in range(n_z):
            for y in range(n_y):
                for x in range(n_x):
                    # determine the order number of the currently handled cell
                    cell_no = (z * n_x * n_y) + (y * n_x) + x  
                    result[cell_no * minimum_cell_atom_count : (cell_no + 1) * minimum_cell_atom_count] = minimum_cell_atom_positions + minimum_cell[0] * x + minimum_cell[1] * y + minimum_cell[2] * z

        return result

def get_hexagonal_ice_coordinates(n_x,  n_y,  n_z, O_O_distance, orthogonal_cell = True, slab=False, misc = False):
    if orthogonal_cell:
        result =  get_hexagonal_ice_coordinates_orth(n_x,  n_y,  n_z, O_O_distance)
    elif misc:
        result = get_hexagonal_ice_coordinates_in_misc_cell(n_x, n_y, n_z, O_O_distance)
    else:
        result = get_hexagonal_ice_coordinates_in_minimum_cell(n_x, n_y, n_z, O_O_distance)
    if slab:
        result[:, 2] += 5
    return result

def get_hexagonal_cell(n_x,  n_y,  n_z, O_O_distance, orthogonal_cell = True, slab=False, misc = False):
    if orthogonal_cell:
        cell = get_hexagonal_ice_orthogonal_cell(n_x,  n_y,  n_z, O_O_distance)
    elif misc:
        cell = get_hexagonal_ice_misc_cell(n_x, n_y, n_z, O_O_distance)
    else:
        cell =  get_hexagonal_ice_minimum_cell(n_x, n_y, n_z, O_O_distance)
    if slab:
        cell[2][2] += 10 
    return cell

def get_hexagonal_ice_orthogonal_cell(n_x,  n_y,  n_z, O_O_distance):
    return np.array([n_x * np.sqrt(8./3.) * O_O_distance, n_y * np.sqrt(8)  * O_O_distance, n_z * 8 * O_O_distance / 3]) * np.identity(3)


def get_hexagonal_slab_order_orth(n_x, n_y, n_z):
    N = 8
    length = N * n_x * n_y * n_z
    result = []
    a = 0
    if n_z % 2 == 1:
        z_order = [n_z/2, n_z/2]
        z_top = [0, 1]
        z_a = n_z/2 + 1 
        z_b = n_z/2 - 1
        while z_b >= 0:
          
            z_order.append(z_b)
            z_top.append(1)
            z_order.append(z_a)
            z_top.append(0)
            z_order.append(z_b)
            z_top.append(0)
            z_order.append(z_a)
            z_top.append(1)
            z_b -= 1
            z_a += 1
            print z_order, z_b
    else:
        z_order = []
        z_top = []
        z_a = n_z/2 
        z_b = n_z/2 -1
        while z_b >= 0:
            z_order.append(z_b)
            z_top.append(1)
            z_order.append(z_a)
            z_top.append(0)
            z_order.append(z_b)
            z_top.append(0)
            z_order.append(z_a)
            z_top.append(1)
            z_b -= 1
            z_a += 1
    for i, z in enumerate(z_order):
        result.extend(_get_layer_numbers(z_top[i], n_x, n_y, z))
        
    return np.array(result, dtype=np.uint8)      

def _get_layer_numbers(top, n_x, n_y, z):
    result = []
    layer_zero = z * n_x * n_y * 8
    unit_zero = 0
    for y in range(n_y):    
        for x in range(n_x):
            c = layer_zero + unit_zero
            if top:
                result.extend([2 + c, 3 + c, 6 + c, 7 + c])
            else:
                result.extend([0 + c, 1 + c, 4 + c, 5 + c])
            unit_zero += 8
    return result
            
                    

def get_hexagonal_ice_coordinates_orth(n_x,  n_y,  n_z, O_O_distance):
        """
            n_x : number of cells in x direction
            n_y : number of cells in y direction
            n_z : number of cells in z direction
            Minimum cell contains 8 molecules
        """

        x_0l = np.sqrt(8./3.) * O_O_distance
        y_0l = np.sqrt(8)  * O_O_distance
        z_0l =  8 * O_O_distance / 3
        multiplier_numbers = np.array([
                [1, 0.01, 5],
                [1, 2.01, 3],
                [1, 0.01, 11],
                [1, 2.01, 13],
                [3, 3.01, 5],
                [3, 5.01, 3],
                [3, 3.01, 11],
                [3, 5.01, 13]
               ])
        
        
        length = len(multiplier_numbers) * n_x * n_y * n_z
        result = np.zeros((length,3))
        n_x_ = n_x
        n_y_ = n_y
        n_z_ = n_z
        N = len(multiplier_numbers)
        no = 0
        
        for z in range(n_z):
            for y in range(n_y):
                for x in range(n_x):
                    for i in range(N):
                        result[no][0] = x * x_0l + multiplier_numbers[i][0] * x_0l / 4
                        result[no][1] = y * y_0l + multiplier_numbers[i][1] * y_0l / 6
                        result[no][2] = z * z_0l + multiplier_numbers[i][2] * z_0l / 16
                        no += 1
        return result

def get_hexagonal_ice_minimum_cell(n_x, n_y, n_z, O_O_distance):
    cell = np.array(
    [[ 1.63299316,  0.,          0.        ],
     [ 0.81649658,  1.41421356,  0.        ],
     [ 0.,          0.,          2.66666667]])
    return np.array([n_x * cell[0], n_y * cell[1], n_z * cell[2]]) * O_O_distance

def get_hexagonal_ice_misc_cell(n_x, n_y, n_z, O_O_distance):
    cell = np.array(
    [[ 2.44948,     1.414213562,  0.        ],
     [ 0,           2.828427125,  0.        ],
     [ 0.,          0.,           2.66666667]])
    return np.array([n_x * cell[0], n_y * cell[1], n_z * cell[2]]) * O_O_distance

def get_hexagonal_ice_coordinates_in_misc_cell(n_x, n_y, n_z, O_O_distance):
    cell = np.array(
    [[ 2.44948,     1.414213562,  0.        ],
     [ 0,           2.828427125,  0.        ],
     [ 0.,          0.,           2.66666667]])
    multiplier_numbers=np.array(
      [[ 1.30681121,  3.54558446,  2.25000004],
       [ 1.30681121,  3.54558446,  4.95000009],
       [ 1.30681121,  6.09116891,  1.35000002],
       [ 1.30681121,  6.09116891,  5.8500001 ],
       [ 3.51135194,  2.27279223,  1.35000002],
       [ 3.51135194,  2.27279223,  5.8500001 ],
       [ 3.51135194,  7.36396114,  2.25000004],
       [ 3.51135194,  7.36396114,  4.95000009],
       [ 5.71589275,  3.54558446,  2.25000004],
       [ 5.71589275,  3.54558446,  4.95000009],
       [ 5.71589275,  6.09116891,  1.35000002],
       [ 5.71589275,  6.09116891,  5.8500001 ]]
       ) / 2.7
    N = len(multiplier_numbers)
    result = np.zeros((N*(n_x*n_y*n_z), 3))
    no = 0
    for z in range(n_z):
        for y in range(n_y):
            for x in range(n_x):
                for i in range(N):
                    result[no] = (x * cell[0] + y * cell[1] + z * cell[2] + (multiplier_numbers[i])) * O_O_distance
                    no += 1
    return result

def get_hexagonal_ice_coordinates_in_minimum_cell(n_x, n_y, n_z, O_O_distance):
    cell = np.array(
    [[ 1.63299316,  0.,          0.        ],
     [ 0.81649658,  1.41421356,  0.        ],
     [ 0.,          0.,          2.66666667]])
    multiplier_numbers = np.array(
    [[ 0.08333333,  0.33333333,  0.1875    ],
     [ 0.41666667, -0.33333333,  0.3125    ],
     [ 0.08333333,  0.33333333, -0.1875    ],
     [ 0.41666667, -0.33333333, -0.3125    ]])
    multiplier_numbers += np.array([0, 0.43333333, 0.4125])
    N = len(multiplier_numbers)
    result = np.zeros((N*(n_x*n_y*n_z), 3))
    no = 0
    for x in range(n_x):
        for y in range(n_y):
            for z in range(n_z):
                for i in range(N):
                    result[no] = (x * cell[0] + y * cell[1] + z * cell[2] + (np.dot(multiplier_numbers[i], cell))) * O_O_distance
                    no += 1
    return result

if __name__ == "__main__":
    print "HERE"
    print get_cell("ice_ic", 2, 1, 1)
    print get_oxygen_coordinates("ice_ic", 2, 1, 1)
