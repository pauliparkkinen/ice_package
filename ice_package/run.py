from water_algorithm_cython import WaterAlgorithm
from structure_commons import StructureCommons
from handle_results import HandleResults

class Run(HandleResults):
    
    def __init__(self, options = None, args = None, O_O_distance = None, O_H_distance = None):
        HandleResults.__init__(self, options = options, O_O_distance = O_O_distance, O_H_distance = O_H_distance)
        self.wa = WaterAlgorithm()
        self.options = options
        self.args = args
        
        
    def initialize(self, cell = None, periodic = None, oxygen_coordinates = None, options = None, order = None, do_symmetry_check = None, charge = None, invariant_count = 20, preset_bond_values = None):
        super(Run, self).initialize(oxygen_coordinates = oxygen_coordinates, cell = cell, periodic = periodic)
        self.wa.initialize(self.get_folder(), nearest_neighbors_nos = self.nearest_neighbors_nos, atoms = self.atoms, order = order, invariant_count = invariant_count, do_symmetry_check = do_symmetry_check, charge = self.charge, preset_bond_values = preset_bond_values)
        
    def execute_commands(self):
        self.wa.execute_commands(self.options, self.args)
        super(Run, self).execute_commands(self.options, self.args)
        
        

    
def handle_options():
    from handle_results import get_parser
    
    from water_algorithm import add_options
    parser = get_parser()
    add_options(parser)

    (options, args) = parser.parse_args()
    hr = Run(options, args)
    hr.initialize()
    hr.execute_commands()
    

if __name__ == '__main__':
    handle_options()
