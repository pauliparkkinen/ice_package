import numpy as np
import scipy
import math
import matplotlib.pyplot
import os
from handle_results import get_parser, HandleResults 
from classification import Classification
from ase.io import read
from ice_ih import add_options_ih, IceIh
from ice_surfaces import get_hexagonal_cell, get_hexagonal_slab_order_orth
from water_algorithm_cython import write_results_to_file,  all_nearest_neighbors_no, get_distance,  WaterAlgorithm,   get_oxygens, get_periodic_distance
from symmetries.symmetry_operation import get_bond_variable_values_from_water_orientation



class IceIhLimited(IceIh):
    
    def __init__(self, O_O_distance=2.726, O_H_distance=1.0, intermediate_saves=[], folder="ice_ih", group_saves=[], n_x = 1, n_y = 1, n_z = 1, slab=False,  orth=True, misc = False, limit = 2):
        WaterAlgorithm()
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        self.orth = orth
        cell = get_hexagonal_cell(n_x, n_y, n_z, O_O_distance, orth, slab, misc)
        
        folder = folder + "_" + str(n_x) + "_" + str(n_y) + "_" + str(n_z) 
        if orth:
            folder += "_orth"
        elif misc:
            folder += "_misc"
    
        self.limit = limit
        if slab and orth:
            order = get_hexagonal_slab_order_orth(n_x, n_y, n_z)
        else:
            order = None
        if slab:
            periodic = [True, True, False]
        else:
            periodic = True
        self.misc = misc
        if slab:            
            folder += "_slab"
        
        folder += "_%i" % limit
        # (1, 1, 1), (1, 1, 2)
        preset_bond_values = [[5, 5, -1], [1, 1, -1]]
        #(2, 1, 1), (1, 2, 1)
        #preset_bond_values = [[1, 1, -1], [5, 5, -1], [9, 9, -1], [13, 13, -1]]
        #(2, 1, 2)
        #preset_bond_values = [[1, 1, -1], [5, 5, -1], [9, 9, -1], [13, 13, -1], [0, 2, 1], [4, 6, 1], [8, 10, 1], [12, 14, 1], [3, 17, 1], [7, 21, 1], [11, 25, 1], [15, 29, 1], [16, 18, 1], [20, 22, 1], [24, 26, 1], [28, 30, 1], [19, 19, 1], [23, 23, 1], [27, 27, 1], [31, 31, 1]]
        #(2, 1, 3)
        #preset_bond_values = [[1, 1, -1], [5, 5, -1], [9, 9, -1], [13, 13, -1], [0, 2, 1], [4, 6, 1], [8, 10, 1], [12, 14, 1], [3, 17, 1], [7, 21, 1], [11, 25, 1], [15, 29, 1], [16, 18, 1], [20, 22, 1], [24, 26, 1], [28, 30, 1], [19, 33, 1], [23, 37, 1], [27, 41, 1], [31, 45, 1], [32, 34, 1], [36, 38, 1], [40, 42, 1], [44, 46, 1], [35, 35, 1], [39, 39, 1], [43, 43, 1], [47, 47, 1]]
        #preset_bond_values = [[1, 1, -1], [5, 5, -1], [9, 9, -1], [13, 13, -1]]
        #(3, 1, 1)
        #preset_bond_values = [[1, 1, -1], [5, 5, -1], [9, 9, -1], [13, 13, -1], [17, 17, -1],  [0, 2, 1], [4, 6, 1], [8, 10, 1], [12, 14, 1], [16, 18, 1], [20, 22, 1], [3, 3, 1], [7, 7, 1], [11, 11, 1], [15, 15, 1], [19, 19, 1], [23, 23, 1] ]
        #(3,2, 1)
        preset_bond_values = [[1, 1, -1], [5, 5, -1], [9, 9, -1], [13, 13, -1], [17, 17, -1], [21, 21, -1], [45, 45, -1], [41, 41, -1], [37, 37, -1], [33, 33, -1], [29, 29, -1], [25, 25, -1],  [47, 47, 1], [43, 43, 1], [39, 39, 1], [35, 35, 1], [31, 31, 1], [27, 27, 1], [44, 46, 1], [40, 42, 1], [36, 38, 1], [32, 34, 1], [28, 30, 1], [24, 26, 1], [0, 2, 1], [4, 6, 1], [8, 10, 1], [12, 14, 1], [16, 18, 1], [20, 22, 1]]
        #preset_bond_values = [[1, 1, -1], [5, 5, -1], [9, 9, -1], [13, 13, -1]]
        #preset_bond_values = [[29,29,1], [25, 25, 1], [21, 21, 1], [17, 17, 1], [1, 1, 1], [5, 5, 1], [9, 9, 1], [13, 13, 1], [0, 2, -1], [4, 6, -1], [8, 10, -1], [12, 14, -1], [3, 33, -1], [7, 37, -1], [11, 41, -1], [15, 45, -1], [32, 34, -1], [36, 38, -1], [40, 42, -1], [44, 46, -1], [28, 30, -1], [24, 26, -1], [20, 22, -1], [16, 18, -1], [31, 61, -1], [27, 57, -1], [23, 53, -1], [19, 49, -1], [60, 62, -1], [56, 58, -1], [52, 54, -1], [48, 50, -1], [63, 63, -1], [59, 59, -1], [55, 55, -1], [51, 51, -1], [35, 35, -1], [39, 39, -1], [43, 43, -1], [47, 47, -1]]
        #preset_bond_values = [[1, 1, 1], [5, 5, 1], [9, 9, 1], [13, 13, 1], [17, 17, 1], [21, 21, 1], [25, 25, 1],  [45, 45, 1], [41, 41, 1], [37, 37, 1], [33, 33, 1], [29, 29, 1], [0, 2, -1], [4, 6, -1], [8, 10, -1], [12, 14, -1], [16, 18, -1], [20, 22, -1], [3, 49, -1], [7, 53, -1], [11, 57, -1], [15, 61, -1], [19, 65, -1], [23, 69, -1], [48, 50, -1], [52, 54, -1], [56, 58, -1], [60, 62, -1], [64, 66, -1], [68, 70, -1], [51, 51, -1], [55, 55, -1], [59, 59, -1], [63, 63, -1], [67, 67, -1], [71, 71, -1], [44, 46, -1], [40, 42, -1], [36, 38, -1], [32, 34, -1], [28, 30, -1], [24, 26, -1], [47, 93, -1], [43, 89, -1], [39, 85, -1], [35, 81, -1], [31, 77, -1], [27, 73, -1], [92, 94, -1], [88, 90, -1], [84, 86, -1], [80, 82, -1], [76, 78, -1], [72, 74, -1]]
        self.slab = slab
        self.initialize(O_H_distance=O_H_distance, O_O_distance=O_O_distance, intermediate_saves=intermediate_saves, group_saves=group_saves, folder=folder, periodic=periodic, cell=cell, charge=0, order = order, preset_bond_values = preset_bond_values)   

        

    """def additional_requirements_met(self, water_orientation, water_orient, molecule_no):
       
        #if water_orientation[molecule_no] != -1:
        #    return False

        # Check the number of AAD-AAD and ADD-ADD bonds
        if molecule_no == 13:
            wo = water_orient.copy()
            wo[molecule_no] = water_orientation
            res, counts = self.classification.get_bond_types(wo)
            if (counts[9][0] == self.limit or counts[9][1] == self.limit):
                return True
            else:
                return False
        return True"""

    def get_single_molecule_hydrogen_coordinates(self, site, water_orientation, i, oxygen_positions,  nearest_neighbors_nos, nn_periodicity, periodicity_axis, cell):
        bvv = get_bond_variable_values_from_water_orientation(water_orientation)
        result = np.zeros((0,3))
        index = 0
        for n,  x in enumerate(nearest_neighbors_nos):
            if bvv[n] == 1:
                if i != x or nn_periodicity[n]:
                    if nn_periodicity[n]:
                        distance,  real_position = get_periodic_distance(oxygen_positions[i], oxygen_positions[x], cell, periodicity_axis[n])
                        
                    else:
                        distance = get_distance(oxygen_positions[i], oxygen_positions[x], False, None)
                        real_position = oxygen_positions[x]
                    result = np.vstack((result, oxygen_positions[i] - (( self.O_H_distance * (oxygen_positions[i]-real_position)) / distance))) 
                    index += 1
                elif i == x and not nn_periodicity[n]:
                    vector = np.array([0, 0, 1])
                    if i % 8 in [0, 1, 4, 5]:
                        result = np.vstack((result, oxygen_positions[i] - (self.O_H_distance * vector)))
                    else:
                        result = np.vstack((result, oxygen_positions[i] + (self.O_H_distance * vector)))
                else:
                    site = np.mod(i, 8)
                    if site == 2 or site == 6:
                        add = np.array(oxygen_positions[i])
                        add[2] += self.O_H_distance
                        result = np.vstack((result, add))
                    elif site == 0 or site == 4:
                        add = np.array(oxygen_positions[i])
                        add[2] -= self.O_H_distance
                        result = np.vstack((result, add))
                    index += 1
        return result
    
    
   
        
def run():
    wa = IceIhLimited()
    wa.run()

def add_options_limit(parser):
    parser.add_option("--limit", dest='limit',  action='store', type = "int", default = 2,
                      help="The allowed total number of up/down facing dangling bonds.")
                
def handle_options():
    from water_algorithm import add_options
    parser = get_parser()
    add_options(parser)
    add_options_ih(parser)
    add_options_limit(parser)
    
    (options, args) = parser.parse_args()
    wa =  IceIhLimited(n_x = options.n_x, n_y = options.n_y, n_z = options.n_z, slab = options.slab, orth = options.orth, misc = options.misc, limit = options.limit)
    wa.classification = Classification(wa) 
    hr = HandleResults(wa)
    wa.invariant_count = options.invariant_count
    wa.execute_commands(options, args)
    wa.execute_local_commands(options, args)
    hr.execute_commands(options, args)


if __name__ == '__main__':
    handle_options()
