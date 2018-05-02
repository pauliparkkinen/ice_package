import numpy as np
import scipy
import math
from ice_package.ice_surfaces import get_hexagonal_cell, get_hexagonal_slab_order_orth, get_hexagonal_ice_coordinates

from ice_package.run import Run
from ice_package.handle_results import HandleResults, get_parser
from ice_package.help_methods import get_distance, get_oxygens


class IceIh(Run):
    
    def __init__(self, O_O_distance=None, O_H_distance=None, intermediate_saves=[], folder="ice_ih", group_saves=[], n_x = 1, n_y = 1, n_z = 1, slab=False,  orth=True, misc = False, options = None, args = None):
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        self.orth = orth
        
        
        folder = folder + "_" + str(n_x) + "_" + str(n_y) + "_" + str(n_z) 
        if orth:
            folder += "_orth"
        elif misc:
            folder += "_misc"
        #if slab and orth:
        #    order = get_hexagonal_slab_order_orth(n_x, n_y, n_z)
        #else:
        order = None
        if slab:
            periodic = [True, True, False]
        else:
            periodic = True
        self.misc = misc
        if slab:            
            folder += "_slab"
        if options is not None and options.charge is not None and options.charge != 0:
            if options.charge > 0:
                folder += "_cat"
            else:
                folder += "_ani"

        self.slab = slab
        options.folder =  folder + "/"   
        Run.__init__(self, options = options, args = args, O_H_distance = O_H_distance, O_O_distance = O_O_distance)
        cell = get_hexagonal_cell(n_x, n_y, n_z, self.O_O_distance, orth, slab, misc)
        self.cell = cell
        oxygen_coordinates = get_hexagonal_ice_coordinates(self.n_x, self.n_y, self.n_z, self.O_O_distance, orthogonal_cell = self.orth, slab=self.slab, misc = self.misc)
        self.initialize( order = order, options = options, oxygen_coordinates = oxygen_coordinates, cell = cell, periodic = periodic, invariant_count = options.invariant_count)
        
  
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
    print options
    hr =  IceIh(n_x = options.n_x, n_y = options.n_y, n_z = options.n_z, slab = options.slab, orth = options.orth, misc = options.misc, options = options, args = args)
    hr.execute_commands()
    
if __name__ == '__main__':
    handle_options() 


