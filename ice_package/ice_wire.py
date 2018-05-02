import numpy as np
import scipy
import math
import os
from handle_results import get_parser, HandleResults
from ase.io import read
from symmetries.symmetry_operation import SymmetryOperation,  get_bond_variable_values_from_water_orientation
from water_algorithm_cython import write_results_to_file,  all_nearest_neighbors_no, get_distance,  WaterAlgorithm,   get_oxygens



class INT(WaterAlgorithm):
    
    def __init__(self, n=2, O_O_distance=2.71, O_H_distance=1.0, intermediate_saves=[], group_saves=[]):
        WaterAlgorithm()
        self.n = n # NUMBER OF MOLECULES IN A LAYER
        self.m = m # LAYER COUNT
        folder = "wire_6_%i" % (n)
        self.initialize(O_H_distance=O_H_distance, O_O_distance=O_O_distance, intermediate_saves=intermediate_saves, group_saves=group_saves, folder=folder)

    
    def get_all_oxygen_coordinates(self):
        result = np.zeros((self.n*self.m, 3), dtype=float)
        interior_angle = (self.n-2)*math.pi/self.n 
        exterior_angle = 2*math.pi/self.n
        r = 0.5 / np.sin(exterior_angle/2)
        counter = 0
        for mi in range(self.m): 
            z = (self.m + 1) / 2.0 
            if mi % 2 == 1:
                counter += 1
                z -= counter 
            else:
                z += counter
            for ni in range(self.n):
                result[mi*self.n + ni] =  np.array([r * math.cos(2 * math.pi * ni / self.n), r * math.sin(2 * math.pi * ni / self.n), z])
                
        result *= self.O_O_distance
        return result
    
    
    def get_single_molecule_hydrogen_coordinates(self, site, water_orientation, i, oxygen_positions,  nearest_neighbors_nos, nn_periodicity, periodicity_axis, cell):
        bvv = get_bond_variable_values_from_water_orientation(water_orientation)
        result = np.zeros((2,3))
        index = 0
        #print nearest_neighbors_nos
        for n,  x in enumerate(nearest_neighbors_nos):
            
            if bvv[n] == 1:
                if i ==x:
                    com = get_oxygens(oxygen_positions).get_center_of_mass()
                    vector = oxygen_positions[i] - com
                    # normalize the vector  
                    vector_length = scipy.linalg.norm(vector)
                    vector /= vector_length
                    # the dangling hydrogen is along this vector
                    result[index] = np.array(oxygen_positions[i] + self.O_H_distance * vector)
                else:
                    result[index] = np.array(oxygen_positions[i] - (( self.O_H_distance * (oxygen_positions[i]-oxygen_positions[x])) / get_distance(oxygen_positions[i], oxygen_positions[x], False, None))) 
                index += 1
        return result
        
    
   
                

def handle_options():
    from water_algorithm import add_options
    parser = get_parser()
    add_options(parser)
    parser.add_option("--number_of_nodes", dest='n', action="store", type="int", help="Number of 4 molecule nodes in wire.", default=1)

    
    (options, args) = parser.parse_args()
    wa =  Wire(n=options.n)
    wa.execute_commands(options, args)
    hr = HandleResults(wa)
    hr.execute_commands(options, args)

    
    

if __name__ == '__main__':
    handle_options()
