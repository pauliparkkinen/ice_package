import numpy as np
import scipy
import math
from ice_surfaces import get_hexagonal_cell, get_hexagonal_slab_order_orth

#from symmetries.symmetry_operation_cython import SymmetryOperation,  get_bond_variable_values_from_water_orientation
#from vesi_algoritmi import write_results_to_file,  all_nearest_neighbors_no,  WaterAlgorithm
from water_algorithm_cython import write_results_to_file,  all_nearest_neighbors_no, get_distance,  WaterAlgorithm,   get_oxygens
from handle_results import HandleResults, get_parser


class IceIh(WaterAlgorithm):
    
    def __init__(self, O_O_distance=2.7, O_H_distance=1.0, intermediate_saves=[], folder="ice_ih", group_saves=[], n_x = 1, n_y = 1, n_z = 1, slab=False,  orth=True, misc = False):
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
        if slab and orth:
            order = get_hexagonal_slab_order_orth(n_x, n_y, n_z)
            print order
        else:
            order = None
        if slab:
            periodic = [True, True, False]
        else:
            periodic = True
        self.misc = misc
        if slab:            
            folder += "_slab"
        self.slab = slab
        self.initialize(O_H_distance=O_H_distance, O_O_distance=O_O_distance, intermediate_saves=intermediate_saves, group_saves=group_saves, folder=folder, periodic=periodic, cell=cell, charge=0, order = order)   
  
        
    def get_all_oxygen_coordinates(self):
        from ice_surfaces import get_hexagonal_ice_coordinates
        return get_hexagonal_ice_coordinates(self.n_x, self.n_y, self.n_z, self.O_O_distance, orthogonal_cell = self.orth, slab=self.slab, misc = self.misc)

         

    def view_proton_ordered(self):
        if self.orth:
            result = np.zeros(self.n_x*self.n_y*self.n_z*8, dtype=int)
            pos = [1, 3, 2, 5, 1, 4, 5, 1]
            for i in range(self.n_x*self.n_y*self.n_z*8):
                p = i % 8
                result[i] = pos[p]
            self.view_result(result)

    def execute_local_commands(self, options, args = None):
        if options.ih_method != None:
            method = getattr(self, str(options.ih_method))
            if args != None:
                method(*args)
            else:
                method()
        
            

def add_options_ih(parser):
    parser.add_option("--view_proton_ordered", dest='ih_method', const="view_proton_ordered", action='store_const',
                      help="View proton ordered hydrogen bond topology")
    parser.add_option("--n_x", dest='n_x', action="store", type="int", help="Number of primitive cell multiples in x direction", default=1)
    parser.add_option("--n_y", dest='n_y', action="store", type="int", help="Number of primitive cell multiples in y direction", default=1)
    parser.add_option("--n_z", dest='n_z', action="store", type="int", help="Number of primitive cell multiples in z direction", default=1)
    parser.add_option("--slab", dest='slab', action="store_true", help="If the structure is a slab")
    parser.add_option("--misc", dest='misc', action="store_true", help="If the structure is in Hex() -cell [see Kuo03]")
    parser.add_option("--orthogonal", dest='orth', action="store_true", help="If the structure is orthogonal")
    
                

def handle_options():
    from water_algorithm import add_options
    parser = get_parser()
    add_options(parser)
    add_options_ih(parser)
    
    (options, args) = parser.parse_args()
    wa =  IceIh(n_x = options.n_x, n_y = options.n_y, n_z = options.n_z, slab = options.slab, orth = options.orth, misc = options.misc)
    hr = HandleResults(wa)
    wa.execute_commands(options, args)
    wa.execute_local_commands(options, args)
    hr.execute_commands(options, args)
    
if __name__ == '__main__':
    handle_options() 


