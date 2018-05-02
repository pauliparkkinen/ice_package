import numpy as np
from water_algorithm_cython import WaterAlgorithm
from structure_commons import StructureCommons
from handle_results import HandleResults, get_parser
from water_algorithm import add_options
from help_methods import get_oxygens, get_opposite_periodicity_axis_number, get_periodic_distance, get_vector_from_periodicity_axis_number, get_distance, get_water_orientations_from_bond_variable_values#, get_water_orientations_from_bond_variable_values, get_bond_variable_values_from_water_orientations
from classification import Classification, get_bond_type_strings, get_water_molecule_type_strings, get_coordination_numbers
from collections import OrderedDict
has_matplotlib = True
try:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.size'] = 16
except ImportError:
    has_matplotlib = False

available_tags = {'O': {'identifier': 'O', 'name': 'Oxygen atoms', 'default_enabled': True, 'default_color': '1.0, 0.0, 0.0, 0.7', 'default_radius': 0.6, 'importance': 0}, 
                  'H': {'identifier':'H', 'name': 'Hydrogen atoms', 'default_enabled': True, 'default_color': '1.0, 1.0, 1.0, 0.7', 'default_radius': 0.25, 'importance':  0},
                 'B': {'identifier': 'B', 'name': 'Boron atoms', 'default_enabled': True, 'default_color': '0.0, 1.0, 0.0, 0.7', 'default_radius': 0.6, 'importance': 0}, 
                 'N': {'identifier': 'N', 'name': 'Nitrogen atoms', 'default_enabled': True, 'default_color': '0.0, 0.0, 1.0, 0.7', 'default_radius': 0.6, 'importance': 0}, 
                 'Na': {'identifier': 'Na', 'name': 'Sodium atoms', 'default_enabled': True, 'default_color': '0.7, 0.0, 1.0, 0.7', 'default_radius': 1.6, 'importance': 0}, 
                 'Pt': {'identifier': 'Pt', 'name': 'Platinum atoms', 'default_enabled': True, 'default_color': '0.7, 0.7, 0.2, 0.7', 'default_radius': 1.6, 'importance': 0}, 
                  'other': {'identifier':'other', 'name': 'Other atoms', 'default_enabled': True, 'default_color': '1.0, 0.7, 0.0, 0.7', 'default_radius': 0.6, 'importance': 0},
                  'four_fold': {'identifier':'four_fold', 'name': 'Four fold-coordinated molecules', 'default_enabled': False, 'default_color': '1.0, 1.0, 1.0, 0.7', 'default_radius': 1.0, 'importance':  1},
                  'three_fold': {'identifier':'three_fold', 'name': 'Three-fold coordinated molecules', 'default_enabled': False, 'default_color': '0.7, 0.7, 0.0, 0.7', 'default_radius': 1.0, 'importance': 1},
                  'two_fold': {'identifier':'two_fold', 'name': 'Two-fold coordinated molecules', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.0, 0.7', 'default_radius': 1.0, 'importance':  1},
                  'one_fold': {'identifier':'one_fold', 'name': 'One-fold coordinated molecules', 'default_enabled': False, 'default_color': '0.0, 0.0, 1.0, 0.7', 'default_radius': 1.0, 'importance':  1},
                  'zero_fold': {'identifier':'zero_fold', 'name': 'Uncoordinated molecules', 'default_enabled': False, 'default_color': '0.0, 0.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 1},
                  'aad': {'identifier':'aad', 'name': 'Double acceptor, donor molecules (AAD)', 'default_enabled': False, 'default_color': '0.0, 0.0, 1.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'add': {'identifier':'add', 'name': 'Double donor, acceptor molecules (ADD)', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'ad': {'identifier':'ad', 'name': 'Single donor, single acceptor molecules (AD)', 'default_enabled': False, 'default_color': '1.0, 0.0, 1.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'aa': {'identifier':'aa', 'name': 'Double acceptor molecules (AA)', 'default_enabled': False, 'default_color': '1.0, 1.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'dd': {'identifier':'dd', 'name': 'Double donor molecules (DD)', 'default_enabled': False, 'default_color': '0.0, 1.0, 1.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'a': {'identifier':'a', 'name': 'Single acceptor molecules (A)', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.5, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'd': {'identifier':'d', 'name': 'Single donor molecules (D)', 'default_enabled': False, 'default_color': '0.5, 0.0, 1.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'ddd': {'identifier':'ddd', 'name': 'Triple donor molecules (DDD)', 'default_enabled': False, 'default_color': '0.5, 0.0, 1.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'aaa': {'identifier':'aaa', 'name': 'Triple acceptor molecules (AAA)', 'default_enabled': False, 'default_color': '0.5, 0.0, 1.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'hydrogen_bond': {'identifier':'hydrogen_bond', 'name': 'Hydrogen bond', 'default_enabled': True, 'default_color': '0.7, 0.0, 0.7, 0.7', 'default_radius': 0.15, 'importance': 0},
                  'aadd_aadd': {'identifier':'aadd_aadd', 'name': 'AADD-AADD bond', 'default_enabled': False, 'default_color': '0.0, 0.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'aad_aadd': {'identifier':'aad_aadd', 'name': 'AAD-AADD bond', 'default_enabled': False, 'default_color': '1.0, 0.7, 0.0, 0.7', 'default_radius': 1.25, 'importance': 2},
                  'aadd_aad': {'identifier':'aadd_aad', 'name': 'AADD-AAD bond', 'default_enabled': False, 'default_color': '1.0, 0.7, 0.0, 0.7', 'default_radius': 1.25, 'importance': 2},
                  'aad_aad': {'identifier':'aad_aad', 'name': 'AAD-AAD bond', 'default_enabled': False, 'default_color': '1.0, 0.0, 0.0, 0.7', 'default_radius': 1.5, 'importance': 2},
                  'aad_add': {'identifier':'aad_add', 'name': 'AAD-ADD bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.0, 0.7', 'default_radius': 1.5, 'importance': 2},
                  'add_aad': {'identifier':'add_aad', 'name': 'ADD-AAD bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'add_add': {'identifier':'add_add', 'name': 'ADD-ADD bond', 'default_enabled': False, 'default_color': '1.0, 0.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'add_aadd': {'identifier':'add_aadd', 'name': 'ADD-AADD bond', 'default_enabled': False, 'default_color': '0.0, 0.7, 0.7, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'aadd_add': {'identifier':'aadd_add', 'name': 'AADD-ADD bond', 'default_enabled': False, 'default_color': '1.0, 0.7, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'dd_aadd': {'identifier':'dd_aadd', 'name': 'DD-AADD bond', 'default_enabled': False, 'default_color': '1.0, 0.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'ad_aadd': {'identifier':'ad_aadd', 'name': 'AD-AADD bond', 'default_enabled': False, 'default_color': '1.0, 0.7, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'aadd_ad': {'identifier':'aadd_ad', 'name': 'AADD-AD bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'aadd_aa': {'identifier':'aadd_aa', 'name': 'AADD-AA bond', 'default_enabled': False, 'default_color': '1.0, 0.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'aad_aa': {'identifier':'aad_aa', 'name': 'AAD-AA bond', 'default_enabled': False, 'default_color': '1.0, 0.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'aad_ad': {'identifier':'aad_ad', 'name': 'AAD-AD bond', 'default_enabled': False, 'default_color': '1.0, 0.7, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'add_aa': {'identifier':'add_aa', 'name': 'ADD-AA bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'add_ad': {'identifier':'add_ad', 'name': 'ADD-AD bond', 'default_enabled': False, 'default_color': '0.0, 0.7, 0.7, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'dd_add': {'identifier':'dd_add', 'name': 'DD-ADD bond', 'default_enabled': False, 'default_color': '1.0, 0.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'ad_add': {'identifier':'ad_add', 'name': 'AD-ADD bond', 'default_enabled': False, 'default_color': '0.0, 0.7, 0.7, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'dd_aad': {'identifier':'dd_aad', 'name': 'DD-AAD bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'ad_aad': {'identifier':'ad_aad', 'name': 'AD-AAD bond', 'default_enabled': False, 'default_color': '0.0, 0.7, 0.7, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'dd_aa': {'identifier':'dd_aa', 'name': 'DD-AA bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'dd_ad': {'identifier':'dd_ad', 'name': 'DD-AD bond', 'default_enabled': False, 'default_color': '1.0, 0.7, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'ad_aa': {'identifier':'ad_aa', 'name': 'AD-AA bond', 'default_enabled': False, 'default_color': '0.0, 0.7, 0.7, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'ad_ad': {'identifier':'ad_ad', 'name': 'AD-AD bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'dd_a': {'identifier':'dd_a', 'name': 'DD-A bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'ad_a': {'identifier':'ad_a', 'name': 'AD-A bond', 'default_enabled': False, 'default_color': '0.7, 0.7, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'd_aa': {'identifier':'d_aa', 'name': 'D-AA bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'd_ad': {'identifier':'d_ad', 'name': 'D-AD bond', 'default_enabled': False, 'default_color': '0.7, 0.7, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'd_aad': {'identifier':'d_aad', 'name': 'D-AAD bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'd_add': {'identifier':'d_add', 'name': 'D-ADD bond', 'default_enabled': False, 'default_color': '1.0, 0.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'd_aadd': {'identifier':'d_aadd', 'name': 'D-AADD bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.7, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'aad_a': {'identifier':'aad_a', 'name': 'AAD-A bond', 'default_enabled': False, 'default_color': '1.0, 0.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'add_a': {'identifier':'add_a', 'name': 'ADD-A bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'aadd_a': {'identifier':'aadd_a', 'name': 'AADD-A bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.7, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'd_a': {'identifier':'d_a', 'name': 'D-A bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.7, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'ddd_aadd': {'identifier':'ddd_aadd', 'name': 'DDD-AADD bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.7, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'ddd_aad': {'identifier':'ddd_aad', 'name': 'DDD-AAD bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.7, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'ddd_add': {'identifier':'ddd_add', 'name': 'DDD-ADD bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.7, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'ddd_aaa': {'identifier':'ddd_aaa', 'name': 'DDD-AAA bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.7, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'ddd_ad': {'identifier':'ddd_ad', 'name': 'DDD-AD bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.7, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'ddd_aa': {'identifier':'ddd_aa', 'name': 'DDD-AA bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.7, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'ddd_a': {'identifier':'ddd_a', 'name': 'DDD-A bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.7, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'aadd_aaa': {'identifier':'aadd_aaa', 'name': 'AADD-AAA bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.7, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'aad_aaa': {'identifier':'aad_aaa', 'name': 'AAD-AAA bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.7, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'add_aaa': {'identifier':'add_aaa', 'name': 'ADD-AAA bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.7, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'dd_aaa': {'identifier':'dd_aaa', 'name': 'DD-AAA bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.7, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'ad_aaa': {'identifier':'ad_aaa', 'name': 'AD-AAA bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.7, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'd_aaa': {'identifier':'d_aaa', 'name': 'D-AAA bond', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.7, 0.7', 'default_radius': 1.0, 'importance': 2},
                  'four_fold_four_fold': {'identifier':'four_fold_four_fold', 'name': 'Bonds between four-fold coordinated molecules', 'default_enabled': False, 'default_color': '0.0, 0.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 1},
                  'three_fold_four_fold': {'identifier':'three_fold_four_fold', 'name': 'Bonds between three-fold and four-fold coordinated molecules', 'default_enabled': False, 'default_color': '1.0, 0.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 1},
                  'four_fold_three_fold': {'identifier':'four_fold_three_fold', 'name': 'Bonds between four-fold and three-fold coordinated molecules', 'default_enabled': False, 'default_color': '1.0, 0.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 1}, 
                  'three_fold_three_fold': {'identifier':'three_fold_three_fold', 'name': 'Bonds between two three-fold coordinated molecules', 'default_enabled': False, 'default_color': '0.0, 0.0, 1.0, 0.7', 'default_radius': 1.0, 'importance': 1}, 
                  'two_fold_three_fold': {'identifier':'two_fold_three_fold', 'name': 'Bonds between two-fold and three-fold coordinated molecules', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 1},
                  'three_fold_two_fold': {'identifier':'three_fold_two_fold', 'name': 'Bonds between three-fold and two-fold coordinated molecules', 'default_enabled': False, 'default_color': '0.0, 1.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 1},
                  'two_fold_two_fold': {'identifier':'two_fold_two_fold', 'name': 'Bonds between two two-fold coordinated molecules', 'default_enabled': False, 'default_color': '0.5, 0.0, 1.0, 0.7', 'default_radius': 1.0, 'importance': 1},
                  'two_fold_four_fold': {'identifier':'two_fold_four_fold', 'name': 'Bonds between  two-fold and four-fold coordinated molecules', 'default_enabled': False, 'default_color': '0.0, 1.0, 1.0, 0.7', 'default_radius': 1.0, 'importance': 1},
                  'four_fold_two_fold': {'identifier':'four_fold_two_fold', 'name': 'Bonds between four-fold and two-fold coordinated molecules', 'default_enabled': False, 'default_color': '0.0, 1.0, 1.0, 0.7', 'default_radius': 1.0, 'importance': 1},
                  'two_fold_one_fold': {'identifier':'two_fold_one_fold', 'name': 'Bonds between two-fold and one-fold coordinated molecules', 'default_enabled': False, 'default_color': '1.0, 1.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 1},
                  'one_fold_two_fold': {'identifier':'one_fold_two_fold', 'name': 'Bonds between one-fold and two-fold coordinated molecules', 'default_enabled': False, 'default_color': '1.0, 1.0, 0.0, 0.7', 'default_radius': 1.0, 'importance': 1}, 
                  'one_fold_one_fold': {'identifier':'one_fold_one_fold', 'name': 'Bonds between two one-fold coordinated molecules', 'default_enabled': False, 'default_color': '1.0, 1.0, 1.0, 0.7', 'default_radius': 1.0, 'importance': 1}, 
                  'one_fold_three_fold': {'identifier':'one_fold_three_fold', 'name': 'Bonds between one-fold and three-fold coordinated molecules', 'default_enabled': False, 'default_color': '1.0, 0.0, 1.0, 0.7', 'default_radius': 1.0, 'importance': 1},
                  'three_fold_one_fold': {'identifier':'three_fold_one_fold', 'name': 'Bonds between three-fold and one-fold coordinated molecules', 'default_enabled': False, 'default_color': '1.0, 0.0, 1.0, 0.7', 'default_radius': 1.0, 'importance': 1}, 
                  'one_fold_four_fold': {'identifier':'one_fold_four_fold', 'name': 'Bonds between one-fold and four-fold coordinated molecules', 'default_enabled': False, 'default_color': '0.5, 0.8, 0.7, 0.7', 'default_radius': 1.0, 'importance': 1}, 
                  'four_fold_one_fold': {'identifier':'four_fold_one_fold', 'name': 'Bonds between four-fold and one-fold coordinated molecules', 'default_enabled': False, 'default_color': '0.5, 0.8, 0.7, 0.7', 'default_radius': 1.0, 'importance': 1}}
                

class WebInterface(HandleResults):
    
    def __init__(self, folder, charge = 0, dissociation_count = 0, preset_bond_values = None, overwrite = True):
        parser = get_parser()
        add_options(parser)

        (options, args) = parser.parse_args()
        options.folder = folder
        options.charge = charge
        options.dissosiation_count = dissociation_count
        self.dissociation_count = dissociation_count
        self.overwrite = overwrite
        self.wa = WaterAlgorithm()
        HandleResults.__init__(self, water_algorithm = self.wa, options = options)
        self.preset_bond_values = preset_bond_values
        self.classification = Classification(options = options)
        self.options = options
        self.args = args
        self.initialized = False
        self.allowed_degrees_of_freedom = 150;
        
        
    def initialize(self):
        super(WebInterface, self).initialize()
        self.wa.initialize(self.get_folder(), nearest_neighbors_nos = self.nearest_neighbors_nos, atoms = self.atoms, charge = self.charge, dissosiation_count = self.dissociation_count, preset_bond_values = self.preset_bond_values, overwrite = self.overwrite)
        self.initialized = True
        
    def execute_commands(self):
        self.wa.execute_commands(self.options, self.args)
        super(Run, self).execute_commands(self.options, self.args)

    def view_oxygen_raft(self):
        self.all_tags = []
        oxygens = get_oxygens(self.oxygen_coordinates, cell = self.cell, periodic = self.periodic)
        coordination_numbers = self.classification.get_coordination_numbers()
        result =  self.atoms_to_javascript(oxygens, self.nearest_neighbors_nos, coordination_numbers = coordination_numbers) 
        result += cell_to_javascript(self.cell)
        result += self.bonds_to_javascript_for_oxygen_raft(self.nearest_neighbors_nos, coordination_numbers, oxygens)
        
        return result
        
    def view_limited_generation_atoms(self, preset_bond_values = None):
        self.all_tags = []
        oxygens = get_oxygens(self.oxygen_coordinates, cell = self.cell, periodic = self.periodic)
        coordination_numbers = self.classification.get_coordination_numbers()
        result =  self.limited_generation_atoms_to_javascript(oxygens, self.nearest_neighbors_nos, coordination_numbers, preset_bond_values)
        result += cell_to_javascript(self.cell)
        return result

    def get_all_tags(self):
        return self.all_tags

    def get_javascript_tag_listeners(self):
        return get_javascript_tag_listeners(self.get_all_tags())

    def select(self, column_names = ['number', 'AAD_AAD', 'ADD_ADD'], where_columns = None, group_by_columns = None, order_by_columns = None):
        return self.select_data(column_names = column_names, where = where_columns, group_by = group_by_columns, order_by = order_by_columns)

    def show_plot(self):
        import cStringIO
        stream = cStringIO.StringIO()
        plt.savefig(stream, format="png")
        plt.clf()
        return stream

    def generate_symmetry_independent(self):
        if not self.initialized:
            self.initialize()
        self.wa.run_command("run", do_symmetry_check = True)

    def limited_generate_symmetry_independent(self):
        self.generate_symmetry_independent()
        
    def generate_all(self):
        if not self.initialized:
            self.initialize()
        self.wa.run_command("run", do_symmetry_check = False)

    def limited_generate_all(self):
        self.generate_all()

    def generate_one(self):
        if not self.initialized:
            self.initialize()
        self.wa.generate_one()

    def view_result_number(self, number):
        self.all_tags = []
        water_orientations = self.load_single_result(number)
        coordination_numbers = self.classification.get_coordination_numbers()
        molecule_types = self.classification.get_water_molecule_type_strings(water_orientations)
        bond_types = self.classification.get_bond_type_strings(water_orientations)
        atoms = self.read(number)
        result =  self.atoms_to_javascript(atoms, self.nearest_neighbors_nos, water_orientations = water_orientations, coordination_numbers = coordination_numbers, molecule_types = molecule_types)
        result += cell_to_javascript(self.cell)
        result += self.bonds_to_javascript_for_proton_configuration(self.nearest_neighbors_nos, water_orientations, coordination_numbers, bond_types, atoms = atoms)
        
        return result
        
    def get_proton_configuration_dict(self, number):
        self.all_tags = []
        water_orientations = self.load_single_result(number)
        coordination_numbers = self.classification.get_coordination_numbers()
        molecule_types = self.classification.get_water_molecule_type_strings(water_orientations)
        bond_types = self.classification.get_bond_type_strings(water_orientations)
        atoms = self.read(number)
    
        result = {}
        
        result['atoms'] = self.atoms_to_dict(atoms, self.nearest_neighbors_nos, water_orientations = water_orientations, coordination_numbers = coordination_numbers, molecule_types = molecule_types)
        result['bonds'] = self.bonds_to_dict_for_proton_configuration(self.nearest_neighbors_nos, water_orientations, coordination_numbers, bond_types, atoms)
        result['cell'] = self.cell.tolist()
        return result
    
    
    def bonds_to_dict_oxygen_raft(self, nearest_neighbors_nos, coordination_numbers, atoms):
        bond_rows, tags, bonds = bonds_to_javascript_for_proton_configuration(nearest_neighbors_nos, coordination_numbers, atoms = atoms)
        self.all_tags.extend(tags)
        return bonds

    
    def bonds_to_javascript_for_oxygen_raft(self, nearest_neighbors_nos, coordination_numbers, atoms):
        """
            Converts bonds to javascript for oxygen raft
        """
        bond_rows, tags, bonds = bonds_to_javascript_for_proton_configuration(nearest_neighbors_nos, coordination_numbers, atoms = atoms)
        self.all_tags.extend(tags)
        return bond_rows
            
    
    def bonds_to_dict_for_proton_configuration(self, nearest_neighbors_nos, water_orientations, coordination_numbers, bond_types, atoms = None):
        bond_rows, tags, bonds = bonds_to_javascript_for_proton_configuration(nearest_neighbors_nos, coordination_numbers, water_orientations = water_orientations, bond_types = bond_types, atoms = atoms)
        self.all_tags.extend(tags)
        return bonds  

    def bonds_to_javascript_for_proton_configuration(self, nearest_neighbors_nos, water_orientations, coordination_numbers, bond_types, atoms = None):
        """
            Converts bonds to javascript for oxygen raft
        """
        bond_rows, tags, bonds = bonds_to_javascript_for_proton_configuration(nearest_neighbors_nos, coordination_numbers, bond_types, water_orientations = water_orientations, atoms = atoms)
        self.all_tags.extend(tags)
        return bond_rows
        
    
    def limited_generation_atoms_to_javascript(self, oxygen_raft, nearest_neighbors_nos, coordination_numbers, preset_bond_values):
        """
            Converts atoms to javascript function suitable for limited generation webgl interface
                -oxygen_raft should be an atoms object containing the oxygen positions
                -preset_bond_values should be a dict containing dicts containing dicts, meaning
                 that the first key is the donor molecule number, second is acceptor molecule number
                 and third is the bond axis. For example if a non-periodic bond between molecule 1 and molecule 4
                 is set to be so that the molecule 1 is the donor, then the preset_bond_values should contain
                 {1: {4: {13: 1}}}.
        """
        atom_rows = "/* Rows defining atoms */\n"
        positions = oxygen_raft.get_positions()
        N = nearest_neighbors_nos.shape[1]
        hydrogen_no = N
        oxygen_tag = 'O'
        hydrogen_tag = 'H' 
        
        oxygen_tags = "\"%s\"" % oxygen_tag
        # Print the oxygen rows 
        for i in range(N):
            atom_rows += "        atom%i = new Atom(8, [%.6f, %.6f, %.6f], [%s], %i);\n" % (i, positions[i, 0], positions[i, 1], positions[i, 2], oxygen_tags, i)
            atom_rows += "        atoms_etc.push(atom%i);\n" % i
           
        
        hydrogen_tags = "\"%s\"" %  hydrogen_tag
        # Handle the hydrogens
        hydrogen_coordinates = self.get_selector_hydrogen_coordinates(coordination_numbers, preset_bond_values)
        hydrogen_no = N
        for i in range(N):
            for j in range(nearest_neighbors_nos.shape[2]):
                neighbor_no = nearest_neighbors_nos[0, i, j]
                if hydrogen_coordinates[i, j, 0] != np.nan:
                    atom_rows += "        atom%i = new Atom(1, [%.6f, %.6f, %.6f], [%s], [%i, %i, %i]);\n" % (hydrogen_no, hydrogen_coordinates[i, j, 0], hydrogen_coordinates[i, j, 1], hydrogen_coordinates[i, j, 2], hydrogen_tags, i, neighbor_no, nearest_neighbors_nos[2, i, j])
                    atom_rows += "        atoms_etc.push(atom%i);\n" % hydrogen_no
                    hydrogen_no += 1
             
        # add the centering parameters
        center = oxygen_raft.get_center_of_mass()
        atom_rows += "        moleculeCenter = [%.6f, %.6f, %.6f]\n" % (center[0], center[1], center[2])
        if any(oxygen_raft.get_pbc()):
            atom_rows += "        drawCell = true;\n"
        atom_rows += "\n"
        return atom_rows


    def atoms_to_javascript(self, atoms, nearest_neighbors_nos, water_orientations = None, coordination_numbers = None, molecule_types = None):
        """
            Converts atoms to javascript function suitable for webgl interface
        """
        content, tags, atom_list = atoms_to_javascript(atoms, nearest_neighbors_nos, water_orientations = water_orientations, coordination_numbers = coordination_numbers, molecule_types = molecule_types)
        self.all_tags.extend(tags)
        return content

    def atoms_to_dict(self, atoms, nearest_neighbors_nos, water_orientations = None, coordination_numbers = None, molecule_types = None):
        """
            Converts atoms to javascript function suitable for webgl interface
        """
        content, tags, atom_list = atoms_to_javascript(atoms, nearest_neighbors_nos, water_orientations = water_orientations, coordination_numbers = coordination_numbers, molecule_types = molecule_types)
        self.all_tags.extend(tags)
        return atom_list

    def get_allowed_field_names(self):
        coordination_numbers = self.classification.get_coordination_numbers()
        result = LastUpdatedOrderedDict()
        result['impossible_angle_count'] = "N<sub>impossible</sub>"
        result['estimated_dipole_moment'] = "Dipole Moment<sub>est.</sub>"
        result['original_core_electrostatic_energy'] = "CoreCoulomb<sub>est.</sub>"
        

        donor_types = []
        acceptor_types = []
        double_acceptor_types = []
        double_donor_types = []
        if coordination_numbers is not None:
            # add the couple of special parameters
            result['AAD_AAD+ADD_ADD'] = "b<sub>homog.</sub>"
            result['AAD_AAD_AAD+ADD_ADD_ADD'] = "a<sub>Th</sub>"

            if 4 in coordination_numbers:
                donor_types.append("AADD")
                acceptor_types.append("AADD")
                double_acceptor_types.append("AADD")
                double_donor_types.append("AADD")
            if 3 in coordination_numbers:
                donor_types.append("AAD")
                donor_types.append("ADD")
                double_acceptor_types.append("AAD")
                acceptor_types.append("AAD")
                acceptor_types.append("ADD")
                double_donor_types.append("ADD")
            if 2 in coordination_numbers:
                donor_types.append("AD")
                donor_types.append("DD")
                double_donor_types.append("DD")
                acceptor_types.append("AA")
                double_acceptor_types.append("AA")
                acceptor_types.append("AD")
                
            if 1 in coordination_numbers:
                donor_types.append("D")
                acceptor_types.append("A")

            for donor_type in donor_types:
                for acceptor_type in acceptor_types:
                    result[donor_type+'_'+acceptor_type] ="b<sub>%s&#8594;%s</sub>" % (donor_type, acceptor_type)

            for donor_type in donor_types:
                for acceptor_type in acceptor_types:
                    if acceptor_type in donor_types:
                        for acceptor_type2 in acceptor_types:
                            result[donor_type+'_'+acceptor_type+'_'+acceptor_type2+'_0'] ="a<sub>%s&#8594;%s&#8594;%s</sub>" % (donor_type, acceptor_type, acceptor_type2)
                    if acceptor_type in double_acceptor_types:
                        for donor_type2 in donor_types:
                            result[donor_type+'_'+acceptor_type+'_'+donor_type2+'_1'] ="a<sub>%s&#8594;%s&#8592;%s</sub>" % (donor_type, acceptor_type, donor_type2)
            for acceptor_type in acceptor_types:        
                for donor_type in donor_types:
                    if donor_type2 in double_donor_types:
                        for acceptor_type2 in acceptor_types:
                            result[acceptor_type+'_'+donor_type+'_'+acceptor_type2+'_2'] ="a<sub>%s&#8592;%s&#8594;%s</sub>" % (acceptor_type, donor_type, acceptor_type2)
            
                    
            
        return result 

bvvs = np.array([[1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1], [-1, 1, 1, -1], [-1, 1, -1, 1], [-1, -1, 1, 1], [1, -1, -1, -1], [-1, 1, -1, -1], [-1, -1, 1, -1], [-1, -1, -1, 1], [-1, 1, 1, 1], [1, -1, 1, 1], [1, 1, -1, 1], [1, 1, 1, -1]], dtype=np.int8)

def cell_to_javascript(cell):
    """
        Converts cell to javascript function suitable for webgl interface
    """
    cell_rows =  "        cell = new Cell([[%.6f, %.6f, %.6f], [%.6f, %.6f, %.6f], [%.6f, %.6f, %.6f]]);\n" % (cell[0, 0], cell[0, 1], cell[0, 2], cell[1, 0], cell[1, 1], cell[1, 2], cell[2, 0], cell[2, 1], cell[2, 2])
    #cell_rows += "        atoms_etc.push(cell);\n\n"
    #center = cell[0] / 2.0 + cell[1] / 2.0 + cell[2] / 2.0
        
    return cell_rows

def get_symbol_tag(atomic_number):
    if atomic_number == 8:
        return "oxygen"
    elif atomic_number == 1:
        return "hydrogen"
    elif atomic_number == 7:
        return "nitrogen"
    elif atomic_number == 5:
        return "boron"
    else:
        return "other"

def get_javascript_tag_listeners(all_tags):
    result = "\n        var color_input;\n"
    result +=   "        var radius_input;\n"
    result +=   "        var check_input;\n\n"
        
    for tag in all_tags:
        result += "        color_input = document.getElementById(\"%s-color\");\n" % tag
        result += "        radius_input = document.getElementById(\"%s-radius\");\n" % tag
        result += "        check_input = document.getElementById(\"%s-enabled\");\n" % tag
            
        result += "        if (color_input != null) {\n"
        result += "            color_input.addEventListener(\"change\", reloadAttributes, false);\n"
        result += "        }\n"
        result += "        if (radius_input != null) {\n"
        result += "            radius_input.addEventListener(\"change\", reloadAttributes, false);\n"
        result += "        }\n"
        result += "        if (check_input != null) {\n"
        result += "            check_input.addEventListener(\"change\", reloadAttributes, false);\n\n"
        result += "        }\n"
    return result


def atoms_to_javascript(atoms, nearest_neighbors_nos, water_orientations = None, bond_variables = None, coordination_numbers = None, molecule_types = None, oxygen_indeces = None, nearest_neighbors_hydrogens = None):
    """
        Converts atoms to javascript function suitable for webgl interface
    """
    atom_rows = "/* Rows defining atoms */\n"
    all_tags = []
    
    positions = atoms.get_positions()
    atom_list = [None]*len(atoms)
    if nearest_neighbors_nos is not None:
        N = nearest_neighbors_nos.shape[1]
    else:
        N = 0
    hydrogen_no = N
    oxygen_tag = "O"
    hydrogen_tag = "H"
    # add oxygen tag to all tags
    if N != 0:
        all_tags.append(oxygen_tag)  
    # add hydrogen tag to all tags 
    if N != 0 and (water_orientations is not None or bond_variables is not None):
        all_tags.append(hydrogen_tag)

        # initialize molecule_types
        if molecule_types is None:
            if water_orientations is None:
                water_orientations = get_water_orientations_from_bond_variable_values(bond_variables)
            molecule_types = get_water_molecule_type_strings(water_orientations, nearest_neighbors_nos)

    # follow which atom indices have been added
    added_atoms = []

    for i in range(N):
        atomic_number = 8
        oxygen_tags = "\"%s\"" % oxygen_tag
        hydrogen_tags = "\"%s\"" %  hydrogen_tag
        tags = ""
        js_oxygen_tags = [oxygen_tag]
        js_hydrogen_tags = [hydrogen_tag]

        # if oxygen_indeces are given, get the index from there
        if oxygen_indeces is not None: 
            oxygen_index = oxygen_indeces[i]
        # otherwise, assume that the index is the same as the order number
        else:
            oxygen_index = i

        # get oxygen position
        oxygen_position = atoms.get_positions()[oxygen_index]
        
        # add coordination number tags
        if coordination_numbers is not None and len(coordination_numbers) > i:
            coordination_number_tag = get_coordination_number_tag(coordination_numbers[i])
            tags += ", \"%s\"" % coordination_number_tag
            
            js_oxygen_tags.append(coordination_number_tag)
            js_hydrogen_tags.append(coordination_number_tag)
            
            if coordination_number_tag not in all_tags:
                all_tags.append(coordination_number_tag)

        # add molecule type tags
        if molecule_types is not None and len(molecule_types) > i:
            mtype_tag = get_molecule_type_tag(molecule_types[i])
            
            if mtype_tag is not None:
                js_oxygen_tags.append(mtype_tag)
                js_hydrogen_tags.append(mtype_tag)
                tags += ", \"%s\"" % mtype_tag
                if mtype_tag not in all_tags:
                    all_tags.append(mtype_tag)

        oxygen_tags += tags
        hydrogen_tags += tags

        # Finally print the oxygen rows
        atom_list[oxygen_index] = [8, 'O', oxygen_position.tolist(), js_oxygen_tags, oxygen_index]
        atom_rows += "        atom%i = new Atom(8, [%.6f, %.6f, %.6f], [%s], %i);\n" % (oxygen_index, oxygen_position[0], oxygen_position[1], oxygen_position[2], oxygen_tags, oxygen_index)
        atom_rows += "        atoms_etc.push(atom%i);\n" % oxygen_index
        added_atoms.append(oxygen_index)  

        # Lets add the hydrogens 

        # If the hydrogens are in 'atoms' and the hydrogen indices have been specified,
        # let's use the indices. Otherwise assume that the hydrogens are after oxygens
        # in the atoms object (use counter)
        if nearest_neighbors_hydrogens is not None:
            bvv = _get_single_molecule_bvvs(i, water_orientations, bond_variables)
            
            for j in range(4):
                if bvv[j] == 1: 
                    # get the specified index
                    if nearest_neighbors_hydrogens is not None:
                        hydrogen_no = nearest_neighbors_hydrogens[i, j]

                    # add the row
                    atom_list[hydrogen_no] = [1, 'H', positions[hydrogen_no].tolist(), js_hydrogen_tags, [i, nearest_neighbors_nos[0, i, j], nearest_neighbors_nos[2, i, j]]]
                    atom_rows += "        atom%i = new Atom(1, [%.6f, %.6f, %.6f], [%s], [%i, %i, %i]);\n" % (hydrogen_no, positions[hydrogen_no, 0], positions[hydrogen_no, 1], positions[hydrogen_no, 2], hydrogen_tags, i, nearest_neighbors_nos[0, i, j], nearest_neighbors_nos[2, i, j]) 
                    atom_rows += "        atoms_etc.push(atom%i);\n" % hydrogen_no
                    added_atoms.append(hydrogen_no)

                    # add default index
                    if nearest_neighbors_hydrogens is None:
                        hydrogen_no += 1
               
        # otherwise, there are no hydrogens
        
            
    # add the non-hydrogen, non-oxygen atoms
    for i in range(positions.shape[0]):
        if i not in added_atoms:
            atomic_number = atoms.get_atomic_numbers()[i]
            symbol_tag = atoms.get_chemical_symbols()[i]
            if symbol_tag not in all_tags:
                all_tags.append(symbol_tag)
            tags = "\"%s\"" % symbol_tag
            atom_list[i] = [atomic_number, symbol_tag, positions[i].tolist(),  [symbol_tag],  i]
            atom_rows += "        atom%i = new Atom(%i, [%.6f, %.6f, %.6f], [%s], %i);\n" % (i, atomic_number, positions[i, 0], positions[i, 1], positions[i, 2], tags, i)
            atom_rows += "        atoms_etc.push(atom%i);\n" % i
        
    # add the centering parameters
    center = atoms.get_center_of_mass()
    atom_rows += "        moleculeCenter = [%.6f, %.6f, %.6f]\n" % (center[0], center[1], center[2])
    if any(atoms.get_pbc()):
        atom_rows += "        drawCell = true;\n"
    atom_rows += "\n"
    return atom_rows, all_tags, atom_list

def simple_atoms_to_javascript(atoms):
    # initialize results etc
    atom_rows = "/* Rows defining atoms */\n"
    positions = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    center = atoms.get_center_of_mass()
    all_tags = []
    
    for i, position in enumerate(positions):
        atomic_number = atomic_numbers[i]
        # add symbol tag
        symbol_tag = get_symbol_tag(atomic_number)
        if symbol_tag not in all_tags:
            all_tags.append(symbol_tag)
        tags = "\"%s\"" % symbol_tag

        # Finally print the atom rows
        atom_rows += "        atom%i = new Atom(%i, [%.6f, %.6f, %.6f], [%s], %i);\n" % (i, atomic_number, positions[i, 0], positions[i, 1], positions[i, 2], tags, i)
        atom_rows += "        atoms_etc.push(atom%i);\n" % i
        
    # add the centering parameters
    center = atoms.get_center_of_mass()
    atom_rows += "        moleculeCenter = [%.6f, %.6f, %.6f]\n" % (center[0], center[1], center[2])
    if any(atoms.get_pbc()):
        atom_rows += "        drawCell = true;\n"
    atom_rows += "\n"
    return atom_rows, all_tags
    
def _get_single_molecule_bvvs(molecule_no, water_orientations, bond_variables):
    """
        Returns the bond variables for a single molecule.
        
        Primary source: bond_variables, Secondary source: water_orientations
        
        raises Exception if both are None
    """
    if bond_variables is not None:
        result = bond_variables[molecule_no]
    elif water_orientations is not None:
        water_orientation = water_orientations[molecule_no]
        if water_orientation == -1:
            result = [0, 0, 0, 0]
        else:
            result = bvvs[water_orientation]
    else:
        result = [0, 0, 0, 0]
        #raise Exception("Invalid input! You need to specify 'bond_variables' or 'water_orientations'.")
    return result


def bonds_to_javascript_for_proton_configuration(nearest_neighbors_nos, coordination_numbers = None, bond_types = None, water_orientations = None, bond_variables = None, nearest_neighbors_hydrogens = None, oxygen_indeces = None, atoms = None):
    """
        Converts bonds to javascript for oxygen raft
           nearest_neighbors_nos : int array of size (3, N, 4)
           nearest_neighbor_hydrogens : int array of size (N, 4)
           bond_variables : int array of size (N, 4), values (1, -1)
           water_orientations: int array of  size (N), values (-1 - 13)
    """
    #assert bond_variables is not None or water_orientations is not None
    N = nearest_neighbors_nos.shape[1]
    hydrogen_no = N
    bond_count = 0
    all_tags = []
    all_tags.append("hydrogen_bond")
    cell = atoms.get_cell()
    bond_rows = ""
    bonds = []
    # initialize coordination numbers
    if coordination_numbers is None and nearest_neighbors_nos is not None:
        coordination_numbers = get_coordination_numbers(nearest_neighbors_nos)
        
    # initialize bond types
    if bond_types is None and nearest_neighbors_nos is not None and water_orientations is not None:
        coordination_numbers = get_bond_type_strings(water_orientations, nearest_neighbors_nos)
    
    for i in range(N):
        bvv = _get_single_molecule_bvvs(i, water_orientations, bond_variables)
        is_set = 1 in bvv or -1 in bvv
        # if oxygen_indeces are given, get the index from there
        if oxygen_indeces is not None: 
            oxygen_index = oxygen_indeces[i]
        # otherwise, assume that the index is the same as the order number
        else:
            oxygen_index = i
        # get oxygen position
        oxygen_position = atoms.get_positions()[oxygen_index]
        
        

        for j in range(4):
            # get periodicity axis
            axis = nearest_neighbors_nos[2, i, j]
        
            # if the indeces of the hydrogens are supplied, use them
            # otherwise use the expected numbering (hydrogens are after oxygens)
            if nearest_neighbors_hydrogens is not None:
                hydrogen_no = nearest_neighbors_hydrogens[i, j]
            
            neighbor_no = nearest_neighbors_nos[0, i, j]
            
            # if oxygen_indeces are given, get the neighbor index from there
            if oxygen_indeces is not None: 
                neighbor_oxygen_index = oxygen_indeces[neighbor_no]
            # otherwise, assume that the index is the same as the order number
            else:
                neighbor_oxygen_index = neighbor_no
            
            # get neighbor bond variable values and determine if the neighbor is set
            neighbor_bvvs = _get_single_molecule_bvvs(neighbor_no, water_orientations, bond_variables)
            neighbor_is_set = 1 in neighbor_bvvs or -1 in neighbor_bvvs
            
            # if handled molecule has no set direction
            if not is_set:
                
                # is_dangling_bond
                is_dangling_bond = neighbor_no == i and nearest_neighbors_nos[0, i, j] == 13

                # Draw oxygen-oxygen "hydrogen bond" if the neighbor has no set value, 
                # it is not a dangling bond, or the neighbor is a acceptor to the
                # corresponding bond
                if is_dangling_bond:
                    draw_bond = False
                elif not neighbor_is_set:
                    if i < neighbor_no:
                        draw_bond = True
                    else:
                        draw_bond = False
                else:
                    # is acceptor
                    neighbor_bvvs = bvvs[water_orientations[neighbor_no]]
                    opposite_axis = get_opposite_periodicity_axis_number(nearest_neighbors_nos[2, i, j])
                    for k in range(4):
                        is_opposite_bond = nearest_neighbors_nos[0, neighbor_no, k] == i and nearest_neighbors_nos[2, neighbor_no, k] == opposite_axis
                        if is_opposite_bond:
                            if neighbor_bvvs[k] == 1:
                                draw_bond = False
                            else:
                                draw_bond = True

                # draw oxygen-oxygen hydrogen bond
                if draw_bond:
                    # get periodic offset to be able to draw periodic bonds correctly
                    if axis == 13:
                        periodic_offset = "undefined"
                        offset = None
                    else:
                        offset = np.dot(get_vector_from_periodicity_axis_number(axis), cell).tolist()
                        periodic_offset = "[%f, %f, %f]" % (offset[0], offset[1], offset[2])
                    
                    # get coordination number tag
                    if coordination_numbers is not None:
                        coordination_number_tag = get_coordination_number_bond_tag(coordination_numbers[i], coordination_numbers[neighbor_no])
                        if coordination_number_tag not in all_tags:
                            all_tags.append(coordination_number_tag)
                        tags = "[\"hydrogen_bond\", \"%s\"]" % coordination_number_tag
                        js_tags = ["hydrongen_bond", coordination_number_tag]
                    else:
                        tags = "[\"hydrogen_bond\"]"
                        js_tags = ["hydrogen_bond"]
                    
                    # write the rows
                    bonds.append([oxygen_index, neighbor_oxygen_index, "HydrogenBond", js_tags, offset])
                    bond_rows += "        bond%i = new Bond(atom%i, atom%i, \"HydrogenBond\", %s, %s, cell);\n" % (bond_count, oxygen_index, neighbor_oxygen_index, tags, periodic_offset) 
                    bond_rows += "        atoms_etc.push(bond%i);\n" % bond_count
                    bond_count += 1
                        
            # Draw bonds between oxygen and its hydrogens
            elif bvv[j] == 1:
                tags = "[]"
                if axis == 13 or cell is None:
                    acceptor_periodic_offset = "undefined"
                    hydrogen_periodic_offset = "undefined"
                    acceptor_offset = None
                    hydrogen_offset = None
                else:
                    # determine if the hydrogen is on the other side of the cell
                    hydrogen_position = atoms.get_positions()[hydrogen_no]
                    periodic_distance, real_position = get_periodic_distance(oxygen_position, hydrogen_position, cell, axis)
                    nonperiodic_distance = get_distance(oxygen_position, hydrogen_position)
                    acceptor_offset = np.dot(get_vector_from_periodicity_axis_number(axis), cell).tolist()
                    acceptor_periodic_offset =  "[%f, %f, %f]" % (acceptor_offset[0], acceptor_offset[1], acceptor_offset[2])
                    # if the periodic distance is smaller than the nonperiodic distance, the hydrogen of the bond is on the other side of the cell
                    if periodic_distance < nonperiodic_distance:
                        
                        hydrogen_offset = np.dot(get_vector_from_periodicity_axis_number(axis), cell).tolist()
                        hydrogen_periodic_offset =  "[%f, %f, %f]" % (hydrogen_offset[0], hydrogen_offset[1], hydrogen_offset[2])
                        acceptor_periodic_offset = "undefined"
                        acceptor_offset = None
                    else:
                        hydrogen_offset = None
                        hydrogen_periodic_offset = "undefined"
                bonds.append([oxygen_index, hydrogen_no, "Normal", None, hydrogen_offset])
                bond_rows += "        bond%i = new Bond(atom%i, atom%i, \"Normal\", %s, %s, cell);\n" % (bond_count, oxygen_index, hydrogen_no, tags, hydrogen_periodic_offset) 
                bond_rows += "        atoms_etc.push(bond%i);\n" % bond_count
                bond_count += 1


                # Draw hydrogen bonds from hydrogen to the accepting part of the hydrogen bond
                if neighbor_no != i:
                    # Bjerrum defect check: draw bond if there is no D-bjerrum defect
                    if neighbor_is_set:
                        opposite_axis = get_opposite_periodicity_axis_number(nearest_neighbors_nos[2, i, j])
                        for k in range(4):
                            if nearest_neighbors_nos[0, neighbor_no, k] == i and nearest_neighbors_nos[2, neighbor_no, k] == opposite_axis:
                                if neighbor_bvvs[k] == 1:
                                    draw_bond = False
                                else:
                                    draw_bond = True
                    else:
                        draw_bond = True
                    if draw_bond:
                        # determine bond type tag
                        if bond_types is not None:
                            bond_type_tag = get_bond_type_tag(bond_types[i][j])
                        else:
                            bond_type_tag = None
                            
                        # determine coordination number tag
                        if coordination_numbers is not None:
                            coordination_number_tag = get_coordination_number_bond_tag(coordination_numbers[i], coordination_numbers[neighbor_no])
                        else:
                            coordination_number_tag = None

                        if bond_type_tag is not None and bond_type_tag not in all_tags:
                            all_tags.append(bond_type_tag)
                        if coordination_number_tag is not None and coordination_number_tag not in all_tags:
                            all_tags.append(coordination_number_tag)
    
                        # the actual rows
                        tags = "[\"hydrogen_bond\", \"%s\", \"%s\"]" % (coordination_number_tag, bond_type_tag) 
                        js_tags = ["hydrogen_bond", coordination_number_tag, bond_type_tag]
                        bonds.append([hydrogen_no, neighbor_oxygen_index, "HydrogenBond", js_tags, acceptor_offset])
                        bond_rows += "        bond%i = new Bond(atom%i, atom%i, \"HydrogenBond\", %s, %s, cell);\n" % (bond_count, hydrogen_no, neighbor_oxygen_index, tags, acceptor_periodic_offset)
                        bond_rows += "        atoms_etc.push(bond%i);\n" % bond_count
                        bond_count += 1
                if nearest_neighbors_hydrogens is None:
                    hydrogen_no += 1
            
    bond_rows += "\n"
    return bond_rows, all_tags, bonds

def get_molecule_type_tag(molecule_type):
    return molecule_type.lower()

def get_coordination_number_bond_tag(coordination_number1, coordination_number2):
    return get_coordination_number_tag(coordination_number1) + "_" + get_coordination_number_tag(coordination_number2)

def get_coordination_number_tag(coordination_number):
    if coordination_number == 4:
        return "four_fold"
    elif coordination_number == 3:
        return "three_fold"
    elif coordination_number == 2:
        return "two_fold"
    elif coordination_number == 1:
        return "one_fold"
    else:
        return "zero_fold"

def get_bond_type_tag(bond_type):
    return bond_type.lower()



class LastUpdatedOrderedDict(OrderedDict):
    """
       Store items in the order the keys were last added
    """

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        OrderedDict.__setitem__(self, key, value)

