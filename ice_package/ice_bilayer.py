import numpy

import numpy as np
import scipy
import math

from water_algorithm_cython import write_results_to_file,  all_nearest_neighbors_no, get_distance,  WaterAlgorithm,   get_oxygens
from handle_results import HandleResults, get_parser


class IceIh(WaterAlgorithm):
    
    def __init__(self, O_O_distance=2.7, O_H_distance=1.0, intermediate_saves=[], folder="ice_bilayer", group_saves=[], n_x = 1, n_y = 1):
        WaterAlgorithm()
        self.n_x = n_x
        self.n_y = n_y
        cell = get_orthogonal_ice_cell(n_x,  n_y,  O_O_distance)
        
        folder = folder + "_" + str(n_x) + "_" + str(n_y) 
        
        periodic = [True, True, False]
        self.initialize(O_H_distance=O_H_distance, O_O_distance=O_O_distance, intermediate_saves=intermediate_saves, group_saves=group_saves, folder=folder, periodic=periodic, cell=cell, charge=0)   
  
        
    def get_all_oxygen_coordinates(self):
        from ice_surfaces import get_hexagonal_ice_coordinates
        return get_orthogonal_ice_coordinates(self.n_x, self.n_y, self.O_O_distance)

        
            


def fit_bilayer_to_cell(cell):
    cell_axis_multipliers = [0.0, 0.0, 0.0]
    ns = [0.0, 0.0, 0.0]
    O_O_distances = [0.0, 0.0, 0.0]
    for i, cell_axis in cell:
        cell_axis_multiplier[i] = cell_axis / 2.7
        n[i] = round(cell_axis_multiplier[i])
        O_O_distances = cell[i] / n[i]
    O_O_distance = sum(O_O_distances) / 3
    return get_orthogonal_ice_coordinates_orth(n[0], n[1], n[2])

def get_orthogonal_ice_cell(n_x,  n_y,  O_O_distance):
    return numpy.array([n_x * numpy.sqrt(8./3.) * O_O_distance, n_y * numpy.sqrt(8)  * O_O_distance, 10.00]) * numpy.identity(3)


def get_orthogonal_ice_coordinates(n_x,  n_y, O_O_distance):
        """
            n_x : number of cells in x direction
            n_y : number of cells in y direction
            n_z : number of cells in z direction
            Minimum cell contains 8 molecules
        """

        x_0l = numpy.sqrt(8./3.) * O_O_distance
        y_0l = numpy.sqrt(8)  * O_O_distance
        z_0l =  8 * O_O_distance / 3
        multiplier_numbers = numpy.array([
                [1, 0.01, 5],
                [1, 2.01, 3],
                [3, 3.01, 5],
                [3, 5.01, 3],
               ])
        
        
        length = len(multiplier_numbers) * n_x * n_y
        result = numpy.zeros((length,3))
        n_x_ = n_x
        n_y_ = n_y
        N = len(multiplier_numbers)
        no = 0
        
        for y in range(n_y):
            for x in range(n_x):
                for i in range(N):
                    result[no][0] = x * x_0l + multiplier_numbers[i][0] * x_0l / 4
                    result[no][1] = y * y_0l + multiplier_numbers[i][1] * y_0l / 6
                    result[no][2] =  multiplier_numbers[i][2] * z_0l / 16
                    no += 1
        return result


def add_options_ih(parser):
    parser.add_option("--n_x", dest='n_x', action="store", type="int", help="Number of primitive cell multiples in x direction", default=1)
    parser.add_option("--n_y", dest='n_y', action="store", type="int", help="Number of primitive cell multiples in y direction", default=1)
    
                

def handle_options():
    from water_algorithm import add_options
    parser = get_parser()
    add_options(parser)
    add_options_ih(parser)
    
    (options, args) = parser.parse_args()
    wa =  IceIh(n_x = options.n_x, n_y = options.n_y)
    hr = HandleResults(wa)
    wa.execute_commands(options, args)
    hr.execute_commands(options, args)
    
if __name__ == '__main__':
    handle_options() 

