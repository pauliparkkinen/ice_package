import numpy as np
import scipy
import math

from ice_package.run import Run
from ice_package.handle_results import HandleResults, get_parser


class INT(Run):   

    def __init__(self, n=2, m=4, O_O_distance=2.71, O_H_distance=1.0, intermediate_saves=[], group_saves=[], options = None, args = None):
        self.n = n # NUMBER OF MOLECULES IN A LAYER
        self.m = m # LAYER COUNT
        folder = "INT_%i_%i" % (n, m)
        options.folder =  folder + "/"   
        Run.__init__(self, options = options, args = args, O_H_distance = O_H_distance, O_O_distance = O_O_distance)
        
        oxygen_coordinates = self.get_all_oxygen_coordinates()
        self.initialize(charge=0, options = options, oxygen_coordinates = oxygen_coordinates,  invariant_count = options.invariant_count)
        
  
    def execute_local_commands(self, options, args = None):
        pass

    
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
    
    """
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
    """
        
    
   
                

def handle_options():
    from water_algorithm import add_options
    parser = get_parser()
    add_options(parser)
    parser.add_option("--number_of_molecules_in_layer", dest='n', action="store", type="int", help="Number of molecules in a layer", default=2)
    parser.add_option("--number_of_layers", dest='m', action="store", type="int", help="Number layers in a nanotube", default=4)

    
    (options, args) = parser.parse_args()
    wa =  INT(m = options.m, n=options.n, options = options, args = args)
    wa.execute_commands()

    
    

if __name__ == '__main__':
    handle_options()
