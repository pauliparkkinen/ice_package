import numpy as np
import scipy
import math
import matplotlib.pyplot
import os
from handle_results import get_parser, HandleResults
from help_methods import * 
from classification import Classification
from symmetries.symmetry_operation import SymmetryOperation,  get_bond_variable_values_from_water_orientation
from ase.io import read
from water_algorithm_cython import WaterAlgorithm



class Pepa(WaterAlgorithm):
    
    def __init__(self, O_O_distance=2.7, O_H_distance=1.0, intermediate_saves=[], folder="pepa_up", group_saves=[]):
        WaterAlgorithm()
        cell = np.array([[+20.5739789056,  +0.0000000000,  +0.0000000000], 
                         [+0.0000000000 ,  +7.2740000000,  +0.0000000000], 
                         [+0.0000000000 ,  +0.0000000000, +45.0000000000]])
        periodic = [True, True, False]
        self.initialize(O_H_distance=O_H_distance, O_O_distance=O_O_distance, intermediate_saves=intermediate_saves,  group_saves=group_saves, folder=folder, do_symmetry_check=False, periodic=periodic, cell = cell)
        self.classification = Classification(self) 

    def get_single_molecule_hydrogen_coordinates(self, site, water_orientation, i, oxygen_positions,  nearest_neighbors_nos, nn_periodicity, nn_periodicity_axis, cell):
        bvv = get_bond_variable_values_from_water_orientation(water_orientation)
        
        if water_orientation > 9:
            result = np.zeros((3,3))
        else:
            result = np.zeros((2,3))
        index = 0
        #print nearest_neighbors_nos
        counter = 0
        for n,  x in enumerate(nearest_neighbors_nos):
            
            if bvv[n] == 1:
                # i == x means that this is a dangling bond
                if i == x:
                    counter += 1
                    if i <= 3 or counter == 2:
                        vector = np.array([0, 0, +1])
                    else:
        
                        com = get_nn_com(i, nearest_neighbors_nos, oxygen_positions, nn_periodicity_axis,  cell)
                        vector = oxygen_positions[i] - com
                        # normalize the vector  
                        vector_length = scipy.linalg.norm(vector)
                        vector /= vector_length
                    # the dangling hydrogen is along this vector
                    result[index] = np.array(oxygen_positions[i] + self.O_H_distance * vector)
                else:
                    if nn_periodicity[n]:
                        distance,  real_position = get_periodic_distance(oxygen_positions[i], oxygen_positions[x], cell, nn_periodicity_axis[n])
                    else:
                        distance = get_distance(oxygen_positions[i], oxygen_positions[x], False, None)
                        real_position = oxygen_positions[x]
                    result[index] = oxygen_positions[i] - (( self.O_H_distance * (oxygen_positions[i]-real_position)) / distance)
                index += 1
        #print result
        return result
        
        
    
        
    
    def get_all_oxygen_coordinates(self):
        return read('oxygens.xyz').get_positions()
    
def get_nn_com(molecule_no, nearest_neighbors_nos, oxygen_positions, nn_periodicity_axis, cell):
    com_coordinates = []
    for n,  x in enumerate(nearest_neighbors_nos):
        if molecule_no != x:
            if nn_periodicity_axis[n] == 13:
                com_coordinates.append(oxygen_positions[x])
            else:
                position = oxygen_positions[x] + np.dot(cell, get_vector_from_periodicity_axis_number(nn_periodicity_axis[n]))
                print molecule_no, n
                print position
                print oxygen_positions[x]
                com_coordinates.append(position)
            
    return get_oxygens(np.array(com_coordinates)).get_center_of_mass()  

def get_vector_from_periodicity_axis_number(number):
    result = np.zeros((3), dtype=float)
    a = number/9 -1 
    b = np.mod(number, 9) / 3 -1 
    c = np.mod(number, 3) -1
    result[0] = a 
    result[1] = b 
    result[2] = c 
    return result 
        
def run():
    wa = Pepa()
    wa.run()
    
def handle_options():
    from water_algorithm import add_options
    parser = get_parser()
    add_options(parser)

    (options, args) = parser.parse_args()
    wa =  Pepa()
    hr = HandleResults(wa)
    wa.execute_commands(options, args)
    hr.execute_commands(options, args)
    



if __name__ == '__main__':
    handle_options()
