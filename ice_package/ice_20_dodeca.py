import numpy as np
import scipy.linalg
import math
#import matplotlib.pyplot
import os
import ase
import sys
from classification import Classification
from ase.io import read
from symmetries.symmetry_operation import SymmetryOperation,  get_bond_variable_values_from_water_orientation, remove_equals 
from water_algorithm_cython import write_results_to_file,  all_nearest_neighbors_no, get_distance,  WaterAlgorithm,  get_oxygens
from handle_results import HandleResults, get_parser
import symmetries.interface_cython as sym



class Dodecahedron(WaterAlgorithm):
    
    def __init__(self, O_O_distance=2.71, O_H_distance=1.0, intermediate_saves=[], folder=None, group_saves=[]):
        WaterAlgorithm()
        self.initialize(O_H_distance=O_H_distance, O_O_distance=O_O_distance, intermediate_saves=intermediate_saves, group_saves=group_saves, folder=folder)
        self.dangling_bond_indeces = None
        
    def get_water_molecule_class(self,  molecule_no):
        return 0
    
    def get_all_oxygen_coordinates(self):
        result = np.array(
       [[  0.607, 0.000,  0.795 ], 
        [  0.188, 0.577,  0.795 ], 
        [ -0.491, 0.357,  0.795 ],
        [ -0.491,-0.357,  0.795 ],
        [  0.188,-0.577,  0.795 ],
        [  0.982, 0.000,  0.188 ],
        [  0.304, 0.934,  0.188 ],
        [ -0.795, 0.577,  0.188 ],
        [ -0.795,-0.577,  0.188 ],
        [  0.304,-0.934,  0.188 ],
        [  0.795, 0.577, -0.188 ],
        [ -0.304, 0.934, -0.188 ],
        [ -0.982, 0.000, -0.188 ],
        [ -0.304,-0.934, -0.188 ],
        [  0.795,-0.577, -0.188 ],
        [  0.491, 0.357, -0.795 ],
        [ -0.188, 0.577, -0.795 ],
        [ -0.607, 0.000, -0.795 ],
        [ -0.188,-0.577, -0.795 ],
        [  0.491,-0.357, -0.795 ]]) * (self.O_O_distance / 0.713644)
        return result
    
    def get_single_molecule_hydrogen_coordinates(self, site, water_orientation, i, oxygen_positions,  nearest_neighbors_nos, nn_periodicity, nn_periodicity_axis, cell):
        bvv = get_bond_variable_values_from_water_orientation(water_orientation)
        result = np.zeros((2,3))
        index = 0
        #print nearest_neighbors_nos
        for n,  x in enumerate(nearest_neighbors_nos):
            
            if bvv[n] == 1:
                # i == x means that this is a dangling bond
                if i == x:
                    com = get_oxygens(self.oxygen_coordinates).get_center_of_mass()
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
        
    
        
    def add_dangling_bonds_to_nearest_neighbors(self,  molecule_no,  nearest_neighbors_nos, periodicity, periodicity_axis):
        """
            For dodecahedron, every molecule has a dangling bonds
        """
        nearest_neighbors_nos = np.append(nearest_neighbors_nos, molecule_no)
        periodicity = np.append(periodicity, False)
        periodicity_axis = np.append(periodicity_axis, 13)
        return nearest_neighbors_nos, periodicity, periodicity_axis



                
def handle_options():
    from water_algorithm_cython import add_options
    parser = get_parser()
    add_options(parser)

    
    (options, args) = parser.parse_args()
    wa =  Dodecahedron(intermediate_saves=[8, 11, 12, 13, 14, 18], folder="dodecahedron", group_saves=[19])
    hr = HandleResults(wa)
    wa.execute_commands(options, args)
    hr.execute_commands(options, args)


def write():
    wa = Dodecahedron(folder="dodecahedron")
    wo = wa.load_results()
    wa.save_results(19, wo, np.arange(0, 20, 1))


def calculate_small_dm():
    wa = Dodecahedron(folder="dodecahedron")
    hr = HandleResults(0, 30026, wa)
    numbers = hr.get_small_dipole_moment_estimates(limit=0.7)
    from run_energy_2 import EnergyCalculations
    EnergyCalculations(folder="dodecahedron", number_of_results=30026,  calculated_numbers=numbers).run()
      
        
def run():
    wa = Dodecahedron(intermediate_saves=[8, 11, 12, 13, 14, 18], folder="dodecahedron", group_saves=[19])
    do_symmetry_check = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1], dtype=int)
    wa.run(do_symmetry_check=do_symmetry_check, i=8)

def try_class():    
    wa = Dodecahedron(folder="dodecahedron")               
    classi = Classification(wa).group_by_aadd_and_aad(nozeros=False) 
    hr = HandleResults(0, 30026, wa)
    smdip = hr.get_small_dipole_moment_estimates()
    #print smdip
    print classi[6].keys()
    count = 0
    for i in classi[6][6]:
        if i in smdip:
            energy = hr.read_energy(i)
            count += 1
            if energy != None:
                print "Geometry %i has 0 add and a small dipole moment (Energy %f eV)" % (i, energy)
            else:
                print "Geometry %i has 0 add and a small dipole moment" % i
    print "Count: %i vs %i" % (count, len(classi[6][6]))


if __name__ == '__main__':
    handle_options()

