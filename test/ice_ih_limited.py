import numpy as np
import scipy
import math
from ice_package.ice_surfaces import get_hexagonal_cell, get_hexagonal_slab_order_orth, get_hexagonal_ice_coordinates

from ice_package.run import Run
from ice_package.handle_results import HandleResults, get_parser
from ice_package.help_methods import get_distance, get_oxygens
from ice_ih import IceIh, add_options_ih


class IceIhLimited(IceIh):
    
    def __init__(self, O_O_distance=None, O_H_distance=None, intermediate_saves=[], folder="ice_ih_limited", group_saves=[], n_x = 1, n_y = 1, n_z = 1, slab=False,  orth=True, misc = False, options = None, args = None):
        IceIh.__init__(self, O_O_distance = O_O_distance, O_H_distance = O_H_distance, intermediate_saves = intermediate_saves, folder=folder, n_x = n_x, n_y = n_y, n_z = n_z, slab = slab, orth = orth, misc = misc, options = options, args = args)
        preset_bond_values = {}
        #values = [-1, -1, -1, 1, 1 , -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1]
        """for i in range(8):
            preset_bond_values[4*i] = {}
            preset_bond_values[4*i][2+4*i] = {}
            preset_bond_values[4*i][2+4*i][13] = values[i]
        """
        for i in range(4):
            preset_bond_values[4*i] = {}
            preset_bond_values[4*i][2+4*i] = {}
            preset_bond_values[4*i][2+4*i][13] = 1
            
            
            preset_bond_values[32+4*i] = {}
            preset_bond_values[32+4*i][34+4*i] = {}
            preset_bond_values[32+4*i][34+4*i][13] = 1

            preset_bond_values[16+4*i] = {}
            preset_bond_values[16+4*i][18+4*i] = {}
            preset_bond_values[16+4*i][18+4*i][13] = -1

            preset_bond_values[48+4*i] = {}
            preset_bond_values[48+4*i][50+4*i] = {}
            preset_bond_values[48+4*i][50+4*i][13] = -1
    
            # dangling bonds
            """
            preset_bond_values[1+4*i] = {}
            preset_bond_values[1+4*i][1+4*i] = {}
            preset_bond_values[1+4*i][1+4*i][13] = 1
            preset_bond_values[3+4*i] = {}
            preset_bond_values[3+4*i][3+4*i] = {}
            preset_bond_values[3+4*i][3+4*i][13] = -1

            preset_bond_values[17+4*i] = {}
            preset_bond_values[17+4*i][17+4*i] = {}
            preset_bond_values[17+4*i][17+4*i][13] = 1
            preset_bond_values[19+4*i] = {}
            preset_bond_values[19+4*i][19+4*i] = {}
            preset_bond_values[19+4*i][19+4*i][13] = -1

            preset_bond_values[33+4*i] = {}
            preset_bond_values[33+4*i][33+4*i] = {}
            preset_bond_values[33+4*i][33+4*i][13] = -1
            preset_bond_values[35+4*i] = {}
            preset_bond_values[35+4*i][35+4*i] = {}
            preset_bond_values[35+4*i][35+4*i][13] = 1
    
            preset_bond_values[49+4*i] = {}
            preset_bond_values[49+4*i][49+4*i] = {}
            preset_bond_values[49+4*i][49+4*i][13] = -1
            preset_bond_values[51+4*i] = {}
            preset_bond_values[51+4*i][51+4*i] = {}
            preset_bond_values[51+4*i][51+4*i][13] = 1"""
        for i in preset_bond_values:
            for j in preset_bond_values[i]:
                if i == j:
                    print "%i, %i: %i" % (i, j, preset_bond_values[i][j][13]) 
        self.wa.set_preset_bond_values(preset_bond_values)
        
                
                

def handle_options():
    from water_algorithm import add_options
    parser = get_parser()
    add_options(parser)
    add_options_ih(parser)
    
    (options, args) = parser.parse_args()
    hr =  IceIhLimited(n_x = options.n_x, n_y = options.n_y, n_z = options.n_z, slab = options.slab, orth = options.orth, misc = options.misc, options = options, args = args)
    hr.execute_commands()
    
if __name__ == '__main__':
    handle_options() 
