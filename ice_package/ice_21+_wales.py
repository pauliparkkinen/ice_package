import numpy as np
import scipy
import math
import matplotlib.pyplot
import os
from handle_results import get_parser, HandleResults 
from classification import Classification
from ase.io import read
from water_algorithm_cython import write_results_to_file,  all_nearest_neighbors_no, get_distance,  WaterAlgorithm,   get_oxygens
from symmetries.symmetry_operation import get_bond_variable_values_from_water_orientation



class Wales(WaterAlgorithm):
    
    def __init__(self, O_O_distance=2.8, O_H_distance=1.0, intermediate_saves=[], folder="wales_21+", group_saves=[]):
        WaterAlgorithm()
        self.initialize(O_H_distance=O_H_distance, O_O_distance=O_O_distance, intermediate_saves=intermediate_saves, group_saves=group_saves, folder=folder, charge=1, do_symmetry_check=False, order=[3,14,9,2,10,19,16,15,7,18,4,0,17,1,12,6,5,8,20,11,13])
        self.N = 21
        self.classification = Classification(self) 

    def get_single_molecule_hydrogen_coordinates(self, site, water_orientation, i, oxygen_positions,  nearest_neighbors_nos, nn_periodicity, nn_periodicity_axis, cell):
        bvv = get_bond_variable_values_from_water_orientation(water_orientation)
        
        if water_orientation > 9:
            result = np.zeros((3,3))
        else:
            result = np.zeros((2,3))
        index = 0
        #print nearest_neighbors_nos
        for n,  x in enumerate(nearest_neighbors_nos):
            
            if bvv[n] == 1:
                # i == x means that this is a dangling bond
                if i == x:
                    com = oxygen_positions[3]
                    vector = oxygen_positions[i] - com
                    # normalize the vector  
                    vector_length = scipy.linalg.norm(vector)
                    vector /= vector_length
                    # the dangling hydrogen is along this vector
                    result[index] = np.array(oxygen_positions[i] + self.O_H_distance * vector)
                else:
                    result[index] = np.array(oxygen_positions[i] - (( self.O_H_distance * (oxygen_positions[i]-oxygen_positions[x])) / get_distance(oxygen_positions[i], oxygen_positions[x], False, None))) 
                index += 1
        #print result
        return result
        
        
    
    
    def get_all_oxygen_coordinates(self):
        """result = np.array(
       [[  0.000, 0.000,  0.000 ],
        [  0.427, -0.000,  0.565 ], 
        [  0.188, 0.577,  0.795 ], 
        [ -0.491, 0.357,  0.795 ],
        [ -0.491,-0.357,  0.795 ],
        [  0.188,-0.577,  0.795 ],
        [  0.982, 0.000,  0.188 ],
        [  0.304, 0.934,  0.188 ],
        [ -0.795, 0.577,  0.188 ],
        [ -0.795,-0.577,  0.188 ],
        [  0.304,-0.934,  0.188 ],
        [  0.645, 0.447, -0.118 ],
        [ -0.304, 0.934, -0.188 ],
        [ -0.702, 0.000, -0.158 ],
        [ -0.304,-0.934, -0.188 ],
        [  0.795,-0.577, -0.188 ],
        [  0.491, 0.357, -0.795 ],
        [ -0.188, 0.577, -0.795 ],
        [ -0.607, 0.000, -0.795 ],
        [ -0.188,-0.577, -0.795 ],
        [  0.491,-0.357, -0.795 ]]) * (self.O_O_distance / 0.713644)
        return result"""
        return read('optimal_wales.xyz').get_positions()

    def additional_requirements_met(self, water_orientation, water_orient, molecule_no):
        wo = water_orient.copy()
        if wo[molecule_no] != -1:
            return False
        wo[molecule_no] = water_orientation
        res, counts = self.classification.get_bond_types(wo)
        if counts[10][1] > 0:
            print water_orient
            print counts
            raw_input()
        # Check the number of AAD-AAD and ADD-ADD bonds
        if counts[4][0]+counts[8][0]> 2 or counts[3][0] > 1 or counts[10][1] > 0 or counts[11][1] > 0 or counts[12][1] > 0 or counts[13][1] > 0:
            #print "----------------"
            #print counts[4][0]+counts[8][0]
            #self.view_result(wo)
            #raw_input()
            #print 
            return False
        else:
            return True
    
    
   
        
def run():
    wa = Wales()
    wa.run()
    
def handle_options():
    from water_algorithm import add_options
    parser = get_parser()
    add_options(parser)

    (options, args) = parser.parse_args()
    wa =  Wales()
    hr = HandleResults(wa)
    wa.execute_commands(options, args)
    hr.execute_commands(options, args)
    



if __name__ == '__main__':
    handle_options()
