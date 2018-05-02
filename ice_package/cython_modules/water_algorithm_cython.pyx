#cython: wraparound=False
#cython: nonecheck=False
#cython: infer_types=True
#cython: boundscheck=False
#cython: c_line_in_traceback=False
import numpy as np  
cimport numpy as np
np.import_array()
import scipy
import math
cimport cython

import copy
import ase
import os
import sys
from help_methods import get_oxygens, print_, split_list, merge_water_orientations, merge, get_periodic_distance, remove_hydrogens, are_equal, get_coordinates_in_basis, remove_equals
from time import time
from .result_group_cython cimport ResultGroup, merge_groups
from .graph_invariants_cython cimport get_invariants, Invariant, InvariantTerm, print_invariants, initialize_new_indexing, get_dangling_bond_invariants

from .symmetries.symmetry_operation_cython cimport SymmetryOperation, do_symmetry_operation_filtering, remove_earlier_found, mark_earlier_found, mark_all_not_found, all_found, get_sub_symmetry_level_improved, filter_symmetry_operations_by_dangling_bond_profile
from .symmetries.self_symmetry_group_cython cimport SelfSymmetryGroup
from .symmetries.interface_cython cimport find_symmetry_operations, print_symmetry_operations
from symmetries.interface_cython import get_moments_of_inertia, get_closest_positions
from cpython cimport bool
from structure_commons import load_results
from random import randint


# Check if we have MPI and give the correct version output
IF USE_MPI:
    from mpi4py cimport MPI
    from mpi4py import MPI
    from mpi4py cimport mpi_c 
    cdef public MPI.Comm comm = MPI.COMM_WORLD
    cdef public unsigned char size = comm.Get_size()
    cdef public unsigned char rank = comm.Get_rank()
    version = "PARALLEL"
ELSE:
    cdef public comm = None
    cdef public unsigned char size = 1
    cdef public unsigned char rank = 0
    version = "SERIAL"

#python side variables
cpdef tuple get_mpi_variables():
    return comm, size, rank




cdef class WaterAlgorithm: 
    """ 
        The main class that handles the proton configuration generation and does some additional tasks. 
        (TODO: move the additional tasks if possible)
    """
    
    def initialize(self, folder, periodic=None, bool slab=False, cell=None, store_bond_variables=False,  intermediate_saves=None,  group_saves=None, signed char charge=0, signed char dissosiation_count=0, do_symmetry_check = None, order = None, preset_bond_values = None, nearest_neighbors_nos = None, atoms = None, invariant_count = 20, overwrite = True):
        """
            periodic: is the cell periodic
            cell : the x y z lengths of the cell i an array
            NOTE: needed only if the periodic is True
            preset_bond_values: 3-dimension dict containing the values for given bonds so that key1 is the number of first atom,
                                key2 the number of second atom, and key3 the periodicity axis, if key3 is not given the periodicity axis is 13 (no periodicity).
                                The value is either 1 or -1.
            dissociation_count : int, how many molecules are dissosiated in the structure
        """
        
        self.invariant_count = invariant_count  
        self.folder = folder
        
        # at which steps additional saves are made
        if intermediate_saves is not None:
            self.intermediate_saves = np.array(intermediate_saves, dtype=np.uint8)
        else:
            self.intermediate_saves = None
        # at which steps additional groups are saved individually
        if group_saves is not None:
            self.group_saves = np.array(group_saves, dtype=np.uint8)
        else:
            self.group_saves = None

        self.charge = charge
        self.dissosiation_count = dissosiation_count
        self.nearest_neighbors_nos = nearest_neighbors_nos
        self.atoms = atoms
        self.overwrite = overwrite
        # if nearest_neighbors_nos is not given, initializing the other generation parameters is void  
        if self.nearest_neighbors_nos is not None:
            self.symmetry_operations = []
            self.N = self.nearest_neighbors_nos.shape[1]
            self.store_bond_variables = store_bond_variables   
            self.set_preset_bond_values(preset_bond_values)
            self.logfile = None 
            self.write_geometries = False
            self.time_start = 0
            self.preset_water_orientations = None
            
            # initilize at which points the symmetry is checked, default is on every molecule
            self.initialize_symmetry_check(do_symmetry_check)
            
            
            # the order in which molecules are iterated, default is from 0 to N-1
            if order == None:
                self.order = np.arange(0, self.N, 1, dtype=np.uint16) 
            else:
                self.order = np.array(order, dtype=np.uint16)

    def set_preset_bond_values(self, preset_bond_values):
        """
            Initializes the preset bond values from input, possible input types: list and dict
        """
        if preset_bond_values is not None and type(preset_bond_values) == list:
            self.preset_bond_values = self.preset_bonds_list_to_dict(preset_bond_values)
        elif preset_bond_values is None or type(preset_bond_values) == dict:
            self.preset_bond_values = preset_bond_values
        else:
            raise Exception("Invalid preset bond value type")


    def initialize_symmetry_check(self, do_symmetry_check):
        """
            Initializes an array from the input that tells at which steps the symmetry check is performed 
        """
        if do_symmetry_check == None or do_symmetry_check:
            self.do_symmetry_check = np.ones(self.N, dtype=np.uint8)
        elif do_symmetry_check == False:
            self.do_symmetry_check = np.zeros(self.N, dtype=np.uint8)
        else:
            self.do_symmetry_check = do_symmetry_check
        
        
    def get_folder(self):
        return self.folder
    

    cdef void load_invariants(self):  
        """
            Loads graph invariants to the graph_invariants parameter of WaterAlgorithm object (self).
            The actual method are located at graph_invariants_cython -module.
        """
        cdef list symmetry_operations
        cdef int i, N
        cdef SymmetryOperation symmetry_operation
        cdef dict conversion_table 
        cdef list gi

        if self.symmetry_operations == None or self.N <= 1:
            self.graph_invariants == None
            self.do_symmetry_check = np.zeros(self.N, dtype=np.uint8)
        else:
            symmetry_operations = self.symmetry_operations
            N = len(symmetry_operations)
        
        if self.invariant_count > 0 and rank == 0 and self.symmetry_operations is not None and len(self.symmetry_operations) > 1:
            #for i from 0 <= i < N: 
            #    symmetry_operation = symmetry_operations[i]
            #    symmetry_operation.get_symbolic_bond_variable_matrix(self.nearest_neighbors_nos, self) 
            #print self.get_folder()
            gi = get_invariants(self.symmetry_operations,  self.nearest_neighbors_nos, True, False, self.get_folder(), False, self, self.invariant_count, False)
            #print_invariants(gi)
            if len(gi) > self.invariant_count:
                gi = gi[:self.invariant_count]
            
            conversion_table = get_index_conversion_table(self.nearest_neighbors_nos)
            initialize_new_indexing(conversion_table, gi)
        else:
            gi = None
        
        
        if self.invariant_count > 0 and size > 1:
            self.graph_invariants = comm.bcast(gi,  root=0)
            
        else:
            self.graph_invariants = gi
            wos = load_results(self.folder)


        #if self.invariant_count > 0 and self.graph_invariants is not None and wos is not None:
        #    for i in range(wos.shape[0]):
        #        length = len(self.graph_invariants)
        #        self.verify_invariants(wos[i])
        #        print "Removed %i invariants. Finally %i invariants." % (length - len(self.graph_invariants), len(self.graph_invariants))
        #    print_invariants(self.graph_invariants)
        

    def random_water_orientations(self):
        """
            Generate a random water orientation without even checking the ice rules
        """
        wo = np.empty((self.N), dtype=np.int8) 
        for i in range(self.N):
            wo[i] = randint(0, 5)
        return wo

    def run_profile_args(self, *args):
        """
            Generate proton configurations for dangling bond profile given in args
            See 'parse_profile_to_preset_bond_values' to see how the profile input is handled
        """
        cdef np.int8_t[::1] profile = self.parse_profile_to_preset_bond_values(*args)
        self.run_profile(profile)

    def run_profile_number(self, profile_number, molecule_no = None, invariant_level = 0, group = -1):
        """
            Generates proton configurations for dangling bond profile with order number 'profile_number'
        """
        # initialize symmetry operations if needed
        if self.symmetry_operations is None:
            self.initialize_symmetry_operations()
            self.load_invariants()

        cdef np.int8_t[:, ::1] profiles = self.get_allowed_dangling_bond_profiles()
        cdef np.int8_t[::1] profile = profiles[int(profile_number)]
        
        original_preset_bond_values = copy.deepcopy(self.preset_bond_values)

        # modify preset bond values
        self.preset_bond_values = self.add_dangling_bond_profile_to_preset_bond_values(profile, original_preset_bond_values)

        self.run_profile(profile, molecule_no = molecule_no, invariant_level = invariant_level, group = group)
        

    def run_profile(self, np.int8_t[::1] profile, molecule_no = None, invariant_level = 0, group = -1):
        """
            Generates proton configurations for input 'profile'
        """
        # Start timing
        s = time()
        # initialize result array   
        wator = np.zeros((0, self.N), dtype=np.int8)

        # if molecule_no is None, start from the first molecule
        if molecule_no is None:
            molecule_no = self.order[0]
        
        # initialize symmetry operations if needed
        if len(self.symmetry_operations) == 0 and 1 in self.do_symmetry_check:
            self.initialize_symmetry_operations()
            self.load_invariants()

        # filter symmetry operation according to dangling bond profile
        self.symmetry_operations = filter_symmetry_operations_by_dangling_bond_profile(self.symmetry_operations, profile)

        cdef np.ndarray[np.int8_t, ndim=2] wos
        if self.preset_bond_values is not None:
            # This means that the original preset bond values and the profile preset bond values are not compatible 
            # -> the execution of the selected profile can be skipped
            wos = self.perform_2(wator, molecule_no, self.order, self.do_symmetry_check, group, invariant_level, [])
        print_parallel("Profile execution was successful", self.logfile)
        print_parallel("Total time used in execution %f s" % (time() - s), self.logfile)

    

    def run(self, molecule_no=None, invariant_level=0,  group=-1):
        """
            Runs the water algorithm by initializing symmetry operations, invariants and finally
            calling the main method 'perform_2' 
        """
        s = time()

        cdef np.int8_t[:, ::1] profiles = None 
        cdef np.int8_t[::1] profile
        cdef np.ndarray[np.int8_t, ndim=2] wos
        cdef profile_count
        cdef dict original_preset_bond_values
        cdef list three_coordinated = []
        cdef int N = self.N

        cdef int i, j, total = 0

        # if molecule_no is None, start from the first molecule
        if molecule_no is None:
            molecule_no = self.order[0]

        # get number of three coordinated molecules
        cdef int three_coordinated_count
        for i in xrange(N):
            if self.get_coordination_number(i) == 3:
                three_coordinated.append(i)
        three_coordinated_count = len(three_coordinated)

        # initialize symmetry operations and invariants, if needed
        if 1 in self.do_symmetry_check:
            self.initialize_symmetry_operations()
            self.load_invariants()

            # Use profiles if there is advantage of it
            if self.charge == 0 and three_coordinated_count != 0 and float(three_coordinated_count) / float(N) < 0.7:
                profiles = self.get_allowed_dangling_bond_profiles()
            else: 
                profiles = None
         
        # initialize result array   
        wator = np.zeros((0, self.N), dtype=np.int8)
        
        # save original preset bond values so that adding dangling bond profiles to these do no mix things up
        self.profile_no = 0
        if self.preset_bond_values is not None:
            original_preset_bond_values = copy.deepcopy(self.preset_bond_values)
        else:
            original_preset_bond_values = None

        
        # Profiles are useless when charge is not 0, 
        #  - the handling of charges works very similarly and is almost as fast
        # Profiles are obviously useless when there are no three-fold coordinated molecules, 
        # They are useless also when there are too many three fold-coordinated molecules (like dodecahedron where there are only three fold coordinated molecules)
        if profiles is None:
            self.perform_2(wator, molecule_no, self.order, self.do_symmetry_check, group, invariant_level, [])
        else:
            # Run profiles one by one
            profile_count = profiles.shape[0]
            all_symmetry_operations = self.symmetry_operations
            for j in xrange(profile_count):
                print_parallel("--------------------------------", self.logfile)
                print_parallel("Handling profile number %i/%i" % (j, profile_count), self.logfile)
                print_parallel("--------------------------------", self.logfile)
                # Reset result array and take the profile                
                wator = np.zeros((0, self.N), dtype=np.int8)
                profile = profiles.base[j]

                # Reset symmetry operations
                self.symmetry_operations = all_symmetry_operations
                mark_all_not_found(self.symmetry_operations)

                # Filter symmetry operations in a way that only symmetry operations that are self symmertic with profile are present 
                self.symmetry_operations = filter_symmetry_operations_by_dangling_bond_profile(self.symmetry_operations, profile)
                self.preset_bond_values = self.add_dangling_bond_profile_to_preset_bond_values(profile, original_preset_bond_values)

                if self.preset_bond_values is None:
                    # This means that the original preset bond values and the profile preset bond values are not compatible 
                    # -> the execution of the selected profile can be skipped
                    print "Not compatible"
                    print profile.base
                    print original_preset_bond_values
                    continue
                else:
                    wos = self.perform_2(wator, molecule_no, self.order, self.do_symmetry_check, group, invariant_level, [])
                    #print wos
                    print "Total before %i" % total
                    total += wos.shape[0]
                    print total
                self.profile_no += 1
                print "Finally total of %i geometries" % total
            
        print_parallel("Execution was successful", self.logfile)
        print_parallel("Total time used in execution %f s" % (time() - s), self.logfile)

    def generate_profile_representatives(self, profile_number = -1):
        """
            Generates representative proton configurations for all dangling bond
            profiles of only for the 'profile_number' if the input is different than -1
        """
        if self.symmetry_operations is None or len(self.symmetry_operations) == 0:
            self.initialize_symmetry_operations()
            self.load_invariants()
        
        cdef int i, j, N = self.N, profile_count
        profiles = self.get_allowed_dangling_bond_profiles()
        profile_count = profiles.shape[0] 
        cdef np.int8_t[:, ::1] result
        cdef dict original_preset_bond_values
        self.profile_no = 0
        if self.preset_bond_values is not None:
            original_preset_bond_values = copy.deepcopy(self.preset_bond_values)
        else:
            original_preset_bond_values = None
       
        if profile_number == -1 or profile_number is None:
            result = np.zeros((profile_count, self.N), dtype = np.int8)
            for j in range(profile_count):
                self.preset_bond_values = self.add_dangling_bond_profile_to_preset_bond_values(profiles[j], original_preset_bond_values)
                result[j] = self.generate_first_result()
        else:
            profile_number = int(profile_number[0])
            if profile_number > profile_count:
                raise Exception("Profile number is too large. Give a number between 0 and %i." % profile_count)
            else:
                result = np.zeros((1, self.N), dtype = np.int8)
                self.preset_bond_values = self.add_dangling_bond_profile_to_preset_bond_values(profiles[j], original_preset_bond_values)
                result[0]  = self.generate_first_result()
        self.save_results(N - 1, result, np.arange(0, N, 1, dtype=np.uint8))

    cdef np.int8_t[::1] generate_first_result(self):
        cdef int i, N = self.N
        cdef np.uint8_t[:, :, :, ::1] possible_combinations = np.zeros((self.N, 4, 6, 6), dtype=np.uint8)
        cdef np.uint8_t[:, :, ::1] pc
        for i in range(N):
            pc = self.calculate_possible_combinations(i)
            possible_combinations[i] = pc
        return self.get_first_possible_combination(possible_combinations)

    def generate_one(self):
        """
            Generates just one structure according to the options given for the
            WaterAlgorithm. Used for example by the web_interface
        """
        cdef np.int8_t[:, ::1] result = np.zeros((1, self.N), dtype = np.int8)
        result[0] = self.generate_first_result()
        self.save_results(self.N - 1, result, np.arange(0, self.N, 1, dtype=np.uint8))

    def parse_profile_to_preset_bond_values(self, *args):
        """
            Parse dangling bond profile and adds the limitations to preset bond values
                args: list of string or int values: 
                H/1: dangling hydrogen
                O/1: dangling oxygen
                H3O/H3O+: DDD hydronium
                OH/OH-: AAA hydroxide
        """
        print args
        cdef dict original_preset_bond_values
        cdef np.int8_t[::1] profile
        cdef int i, argument_no, N, argument
 
        # get the initial presend bond values
        if self.preset_bond_values is not None:
            original_preset_bond_values = copy.deepcopy(self.preset_bond_values)
        else:
            original_preset_bond_values = None

        # N is the number of molecules in the structure
        N = self.N
        total = 0
        profile = np.zeros((N), dtype = np.int8)
        
        # Calculate preset bond values
        if True:
            argument_no = 0
            total = 0
            for i in range(N):
                if self.get_coordination_number(i) == 3:
                    if len(args) > argument_no:
                        if args[argument_no] == 'H':
                            argument = 1
                        elif args[argument_no] == 'OH' or args[argument_no] == 'OH-':
                            argument = 1
                            original_preset_bond_values = self.add_oh_to_preset_bond_values(i, original_preset_bond_values)
                        elif args[argument_no] == 'O':
                            argument = -1
                        elif args[argument_no] == 'H3O' or args[argument_no] == 'H3O+':
                            argument = -1
                            original_preset_bond_values = self.add_h3o_to_preset_bond_values(i, original_preset_bond_values)
                        else:
                            argument = int(args[argument_no])
                        profile[i] = argument  
                        total += argument                   
                    argument_no += 1

            if argument_no > len(args):
                print_parallel("Error: %i arguments needed, only %i was given." % (argument_no, len(args)), self.logfile)
                raise Exception("Error parsing arguments")
            elif len(args) > argument_no:
                print_parallel("Warning: %i arguments needed, %i was given. Truncating the extra arguments." % (argument_no, len(args)), self.logfile)
        if total/2 != self.charge:
            print_parallel("Error: The profile given does not fulfill ice rules.", self.logfile)
            raise Exception("Error parsing arguments")
        self.preset_bond_values = self.add_dangling_bond_profile_to_preset_bond_values(profile, original_preset_bond_values)
        return profile


    def generate_profile_representative(self, *args):
        """
            the argument name is args to allow two different kind of inputs
            (1) the number of profile
            (2) the values for dangling bonds
        """
        cdef np.int8_t[::1]  water_orientations
        cdef np.int8_t[:, ::1] result
        cdef np.uint8_t[:, :, :, ::1] possible_combinations
        cdef np.uint8_t[:, :, ::1] pc
        cdef int i, argument_no, N = self.N, argument, wo_max = 6
        if len(args) == 0:
            return
        elif len(args) == 1:
            #generate_profile_representatives(args)
            pass
        else:
            try:
                self.parse_profile_to_preset_bond_values(*args)
            except:
                return
            
            # Recreate possible combinations
            if self.charge != 0 or self.dissosiation_count != 0:
                wo_max = 14
            possible_combinations = np.zeros((N, 4, wo_max, wo_max), dtype=np.uint8)
            for i in range(N):
                pc = self.calculate_possible_combinations(i)
                possible_combinations[i] = pc

            # Generate first possible combination
            water_orientations  = self.get_first_possible_combination(possible_combinations)

            # Save result to file
            result = np.zeros((1, N), dtype = np.int8)
            if water_orientations is not None:
                result[0] = water_orientations
                self.save_results(N - 1, result, np.arange(0, N, 1, dtype=np.uint8))
 
    cdef dict add_dangling_bond_profile_to_preset_bond_values(self, np.int8_t[::1] profile, dict original_preset_bond_values):
        cdef int i, N = profile.shape[0]
        # init the result dict
        if original_preset_bond_values is None:
            preset_bond_values = {}
        else:
            preset_bond_values = copy.deepcopy(original_preset_bond_values)
        for i in xrange(N):
            if profile[i] != 0:
                if i not in preset_bond_values:
                    preset_bond_values[i] = {}
                if i not in preset_bond_values[i]:
                    #preset_bond_values[i][i] = {}
                    preset_bond_values[i][i] = {}
                #elif 13 not in preset_bond_values[i][i]:
                #    preset_bond_values[i][i] = profile[i]
                elif 13 in preset_bond_values[i][i] and preset_bond_values[i][i][13] != profile[i]:
                    print i, preset_bond_values[i][i][13], profile[i]
                    return None
                preset_bond_values[i][i][13] = profile[i]
        return preset_bond_values

    cdef dict add_h3o_to_preset_bond_values(self, int molecule_no, dict preset_bond_values):
        """ 
            Adds a triple donor H3O+ to preset bond values
        """
        cdef np.int_t[:, :, ::1] nearest_neighbors_nos = self.nearest_neighbors_nos
        cdef int neighbor_no, axis, i
        cdef int value
        cdef dict preset_water_orientations
        if self.preset_water_orientations is None:
            self.preset_water_orientations = {}
        preset_water_orientations = self.preset_water_orientations
        if preset_bond_values is None:
            preset_bond_values = {}
        for i in xrange(4):
            neighbor_no = nearest_neighbors_nos[0, molecule_no, i]
            axis = nearest_neighbors_nos[2, molecule_no, i]
            
            if neighbor_no != molecule_no or axis != 13:
                value = 1
            else:
                value = -1
                preset_water_orientations[molecule_no] = 10 + i

            if molecule_no not in preset_bond_values:
                preset_bond_values[molecule_no] = {}
            if neighbor_no not in preset_bond_values[molecule_no]:
                preset_bond_values[molecule_no][neighbor_no] = {}
            if axis in preset_bond_values[molecule_no][neighbor_no] and preset_bond_values[molecule_no][neighbor_no][axis] != value:
                return None
            preset_bond_values[molecule_no][neighbor_no][axis] = value
                
                
        return preset_bond_values

    cdef dict add_oh_to_preset_bond_values(self, int molecule_no, dict preset_bond_values):
        """ 
            Adds a triple acceptor OH- to preset bond values
        """
        cdef np.int_t[:, :, ::1] nearest_neighbors_nos = self.nearest_neighbors_nos
        cdef int neighbor_no, axis , i
        cdef int value, water_orientation
        cdef dict preset_water_orientations
        if self.preset_water_orientations is None:
            self.preset_water_orientations = {}
        preset_water_orientations = self.preset_water_orientations
        if preset_bond_values is None:
            preset_bond_values = {}
        for i in xrange(4):
            neighbor_no = nearest_neighbors_nos[0, molecule_no, i]
            axis = nearest_neighbors_nos[2, molecule_no, i]
            
            if neighbor_no != molecule_no or axis != 13:
                value = -1
            else:
                value = 1
                preset_water_orientations[molecule_no] = 6 + i

            if molecule_no not in preset_bond_values:
                preset_bond_values[molecule_no] = {}
            if neighbor_no not in preset_bond_values[molecule_no]:
                preset_bond_values[molecule_no][neighbor_no] = {}
            if axis in preset_bond_values[molecule_no][neighbor_no] and preset_bond_values[molecule_no][neighbor_no][axis] != value:
                return None
            preset_bond_values[molecule_no][neighbor_no][axis] = value
            
                
                
        return preset_bond_values

    cdef void initialize_symmetry_operations(self) except *:
        """
            Calls the initialization of symmetry operations made by interface_cython module
            and sets the symmetry operations to the symmetry_operations attribute of WaterAlgorithm object (self)
        """
        cdef list so, primitive_symmetry_operations, translation_operations
        cdef float error_tolerance = 0.3
        cdef bool debug = False
        cdef SymmetryOperation sop
        if rank == 0:
            try:
                oxygens = self.atoms
                s = time()
                so, primitive_symmetry_operations, translation_operations = find_symmetry_operations(oxygens, error_tolerance)   
                print "Time used in symmetry operation loading %f" % (time() - s)
                s = time()
                for sop in so:
                    sop.initialize(self.nearest_neighbors_nos, self.atoms)
                print "Time used in symmetry operation initialization %f" % (time() - s)
                so = remove_equals(so, debug)
                print "Found %i symmetry operations" % len(so)
            except Exception:
                print "Exception occurred while generating symmetry operations."
                so = None
            
            #so = remove_equals(so, debug)
        else:
            so = None
            primitive_symmetry_operations = None
            translation_operations = None
        if size > 1:
            self.symmetry_operations = comm.bcast(so,  root=0)
            self.primitive_symmetry_operations = comm.bcast(primitive_symmetry_operations,  root=0)
            self.translation_operations = comm.bcast(translation_operations,  root=0)
        else:
            self.symmetry_operations = so
            self.primitive_symmetry_operations = primitive_symmetry_operations
            self.translation_operations = translation_operations
            
    cdef np.ndarray[np.int8_t, ndim=2] get_symmetries(self, np.ndarray[np.int8_t, ndim=1] result, np.ndarray[np.int_t, ndim=3] nearest_neighbors_nos, list symmetry_operations):
        """
            DEPRECATED!
            Loads the symmetries of given water orientations (result)
            Parameters:
                result               : the water orientations of the given result
                nearest_neighbors_nos: the nearest neighbor nos of all water molecules
                symmetry_operations  : the SymmetryOperation objects in a python list
            Returns the symmetries in a two dimensional numpy array
                
        """
        cdef np.ndarray[np.int8_t, ndim=2] symmetries = np.zeros((len(symmetry_operations), result.shape[0]), dtype=np.int8)
        cdef np.ndarray[np.int8_t, ndim=1] symmetry
        cdef SymmetryOperation symmetry_operation
        cdef int i, N = len(symmetry_operations)
        for i from 0 <= i < N:
            symmetry_operation = symmetry_operations[i]
            symmetry = symmetry_operation.apply(result)
            symmetries[i] = symmetry
            #if symmetry is not None:g
            #    symmetries = np.vstack((symmetries,  symmetry))
            #    #symmetrynames.append(symmetry_operation.name)
                
        #print "Symmetries count %i" % len(symmetries) 
        return symmetries


    def print_time_stats(self):
        time_now = time() 
        if self.time_start > 0.0:
            if size > 1:
                self.symmetry_total_time = gather_and_max(self.symmetry_total_time)
                self.conversion_time = gather_and_max(self.conversion_time)
                self.result_group_tryout_time = gather_and_max(self.result_group_tryout_time)
                self.symmetry_load_time = gather_and_max(self.symmetry_load_time)
                self.symmetry_check_time = gather_and_max(self.symmetry_check_time)
                self.iteration_time = gather_and_max(self.iteration_time)
                self.self_symmetry_time = gather_and_max(self.self_symmetry_time)
            #Follow the time used
            total = time_now-self.time_start
            print_parallel("-------------------------------------------------------------------------", self.logfile)
            print_parallel("Total time elapsed %f s" % (total), self.logfile)
            print_parallel("  Total time elapsed in symmetry check %f s" % self.symmetry_total_time, self.logfile)
            print_parallel("    Time used in bond variable conversion %f s" % self.conversion_time, self.logfile)
            print_parallel("    Time used trying the result groups %f s" % self.result_group_tryout_time, self.logfile)
            print_parallel("    Time used in symmetry loading %f s" % self.symmetry_load_time, self.logfile)
            print_parallel("    Time used in symmetry checking %f s" % self.symmetry_check_time, self.logfile)
            print_parallel("    Time used in handling of self symmetry groups %f s" % self.self_symmetry_time, self.logfile)
            print_parallel("  Time used in the iteration of water orientations %f s" % self.iteration_time, self.logfile)
            print_parallel(" ------------------------------------------------------------------------", self.logfile)
        self.time_start = time()
        self.symmetry_load_time = 0
        self.symmetry_check_time = 0
        self.iteration_time = 0
        self.symmetry_total_time = 0
        self.self_symmetry_time = 0
        self.result_group_tryout_time = 0
        self.conversion_time = 0

    
    

    
        
        

    def save_results(self, np.uint16_t molecule_no, water_orientations, order, group_save=False, group_number=-1, all_symmetries_found = False):
        """
            If i is in intermediate saves save only a text file containing water orientations
                -if not, then do nothing
            If i is the last molecule to be handled, then save the water_orientations also
        """
        is_last = molecule_no == order[len(order)-1]
        file_name = None
        if group_save:
            file_name = self.get_folder()+"group_%i_%i.txt" % (molecule_no, group_number)
        else:
            file_name = self.get_folder()+"allresults.txt"

        # handle overwrite input
        overwrite = self.overwrite or not os.path.exists(file_name)

        if group_save or is_last:
            for j in range(size):
                if rank == j  and is_last:
                    if j == 0 and self.profile_no == 0 and overwrite:
                        mode = "wb"
                    else:
                        mode = "ab+"
                    continue_sequence = True
                    print "Processor number %i writes %i results" % (j, len(water_orientations))
                    ar_file = open(file_name, mode)
                    self.write_water_orientations(ar_file, water_orientations)
                    ar_file.close()
                else:
                    continue_sequence = True
                if size > 1:
                    cs = comm.allgather(continue_sequence)
        else:
            if rank == 0:
                folder = self.get_folder()
                if self.intermediate_saves is not None and molecule_no in self.intermediate_saves:
                    np.savetxt(folder+"intermediate_%i.txt" % (molecule_no), water_orientations)

    def write_water_orientations(self, target_file, water_orientations):
        for wo in water_orientations:
            wo_str = ""
            for water_orientation in wo:
                wo_str += "%i " % water_orientation  
            wo_str += "\n"
            target_file.write(wo_str)
            
    def print_result_profiles(self):
        wos = load_results(self.folder)
        self.print_profiles(wos)
        
    def print_profiles(self, wos):
        profiles = self.get_result_profiles(wos)
        for i, profile in enumerate(profiles):
            print i, ": ", profile
            
    def get_result_profiles(self, wos):
        cdef np.ndarray[np.int8_t, ndim=2] result = np.zeros((len(wos), self.N), dtype=np.int8)
        for i, wo in enumerate(wos):
            result[i] = self.get_result_profile(wo)
        return result
            

    def get_result_profile(self, wo):
        cdef np.ndarray[np.int8_t] profile = np.zeros(self.N, dtype=np.int8)
        cdef np.int8_t[:, ::1] bond_variables = np.zeros((self.N, 4), dtype=np.int8)
        get_bond_variables_3(wo,  self.nearest_neighbors_nos, self.N, bond_variables)
        for i in xrange(self.N):
            for j in xrange(4):
                is_dangling_bond = self.nearest_neighbors_nos[0, i, j] == i and self.nearest_neighbors_nos[2, i, j] == 13
                if is_dangling_bond:
                    profile[i] = bond_variables[i, j] 
        return profile

    def print_allowed_dangling_bond_profiles(self):
        self.initialize_symmetry_operations()
        self.load_invariants()
        #self.symmetry_operations = filter_symmetry_operations_by_dangling_bond_profile(self.symmetry_operations, profile.base)
        cdef np.int8_t[:, ::1] profiles = self.get_allowed_dangling_bond_profiles()
        if profiles is not None:
            for i, profile in enumerate(profiles.base):
                print "%i: %s" % (i, profile)
                print_("%i: %s" % (i, profile), self.logfile)
        else:
            print "The handled molecule has no three-fold coordinated molecules"

    def get_allowed_dangling_bond_profiles(self, profile_limit = -1):
        """
            Goes through all dangling bond profiles, removes symmetric  profiles and 
            profiles that are impossible because of the sum rule of dangling bonds 
        """
        
        cdef np.uint8_t M, N = self.N, O
        cdef int i, j, k, l, profile_count
        # get indeces of three-fold coordinated moleculs
        cdef list three_coordinated = []
        for i in xrange(N):
            if self.get_coordination_number(i) == 3:
                three_coordinated.append(i)
        M = len(three_coordinated)
        cdef np.int8_t total
        
        cdef np.ndarray[np.int8_t, ndim=2] profiles = np.zeros((1, N), dtype=np.int8), new_profiles
        cdef np.ndarray[np.int8_t] plusandminus = np.empty(2, dtype=np.int8)
        cdef np.int8_t charge = self.charge
        cdef np.ndarray[np.int8_t, ndim=1] profile#, profile_2
        cdef ResultGroup result_group
        cdef list all_invariants = self.graph_invariants
        cdef bint is_preset
        
        cdef list invariants = get_invariants(self.symmetry_operations,  self.nearest_neighbors_nos, True, False, self.get_folder(), False, self, self.invariant_count, True)
        print_invariants(invariants)
        if len(invariants) > self.invariant_count:
            invariants = invariants[:self.invariant_count]
        
        conversion_table = get_index_conversion_table(self.nearest_neighbors_nos)
        initialize_new_indexing(conversion_table, invariants)
        #cdef list invariants = get_dangling_bond_invariants(self.graph_invariants)
        print "%i db invariants found" % len(invariants) 
        cdef np.int_t[:, :, ::1] nearest_neighbors_nos = self.nearest_neighbors_nos
        cdef int result_group_level = len(invariants)
        plusandminus[0] = 1
        plusandminus[1] = -1
        
        # go through all three-fold coordinated molecules
        for i in xrange(M):
           
            l = three_coordinated[i]
            is_preset = self.preset_bond_values is not None and l in self.preset_bond_values and l in self.preset_bond_values[l] and 13 in self.preset_bond_values[l][l]
            # if the currently handled molecule is not preset
            if not is_preset:
                new_profiles = np.zeros((profiles.shape[0]*2, N), dtype=np.int8)
                # go through all profiles
                for j in xrange(profiles.shape[0]):
                    # and try both values (1, -1)
                    for k in xrange(2):
                        profile = profiles[j].copy()
                        profile[l] = plusandminus[k]
                        if i >= M / 2:
                            total = memoryview_sum(profile, N)
                            if abs(total + charge) <= M - (i + 1):
                                new_profiles[profile_count] = profile
                                profile_count += 1
                        else:
                            new_profiles[profile_count] = profile
                            profile_count += 1
                #print "Profile count %i" % profile_count
                profiles = new_profiles[:profile_count]
            # is preset, set profile value to be preset value for all the profiles
            else:
                profiles[:, l] = self.preset_bond_values[l][l][13]
            # Symmetry Check
            result_group = ResultGroup(invariants, 0)
            profiles = self.remove_symmetric_profiles(result_group, nearest_neighbors_nos, result_group_level, profiles, profile_limit = profile_limit)
            profile_count = 0
        
        # Symmetry Check
        #result_group = ResultGroup(invariants, 0)
        #profiles = self.remove_symmetric_profiles(result_group, nearest_neighbors_nos, result_group_level, profiles)
        #for profile in profiles:
        #    print profile
        #raw_input()
        print "Profile count %i" % profiles.shape[0]
        print profiles
        return profiles
        
    def remove_equal_profiles(self, profiles):
        result = []
        count_before = len(profiles)
        for profile in profiles:
            equal_found = False
            for profile2 in result:
                if wos_are_equal(profile, profile2, len(profile)):
                    equal_found = True
                    break
            if not equal_found:
                result.append(profile)
        print "Removed %i equal profiles" % (count_before - len(result))
        return np.array(result, dtype=np.int8)
        
    def remove_symmetric_profiles(self, ResultGroup result_group, np.int_t[:, :, ::1] nearest_neighbors_nos, int result_group_level, np.ndarray[np.int8_t, ndim = 2] profiles, profile_limit = -1):
        cdef SymmetryOperation symmetry_operation
        
        cdef list symmetry_operations = self.symmetry_operations
        
        cdef int i, j, k, O = len(symmetry_operations), profile_count = 0, N = profiles.shape[0], M
        cdef bint equal_found
        cdef np.int8_t[:, ::1] new_profiles = np.empty((profiles.shape[0], profiles.shape[1]), dtype=np.int8), original_profiles = profiles, group_profiles
        cdef np.int8_t[::1] profile, profile_2
        cdef np.int8_t[:, ::1] bond_variables = np.empty((profiles.shape[1], 4), dtype=np.int8)
        cdef ResultGroup res_group
        # go through all profiles
        for i in xrange(N):
            profile = original_profiles[i]
            get_profile_bond_variables(profile,  nearest_neighbors_nos, nearest_neighbors_nos.shape[1], bond_variables)
            # group profiles with graph invariants
            res_group = result_group._try_subgroups(bond_variables, result_group_level)
            group_profiles = res_group.get_wos()
            equal_found = False
            if group_profiles is not None:
                M = group_profiles.shape[0]
                # go through all symmetry operations
                for j in range(O):
                    symmetry_operation = <SymmetryOperation> symmetry_operations[j]
                    # skip identity operator
                    if symmetry_operation.type == <bytes> 'E':
                        continue
                    # check if symmetric profile is in group's results
                    for k in range(M):
                        profile_2 = group_profiles[k]
                        profile_2 = symmetry_operation.apply_for_dangling_bond_profile(profile_2)
                        if profile_2 is not None and profiles_are_equal(profile, profile_2, original_profiles.shape[1]):
                            equal_found = True
                            break
                    if equal_found:
                        break
            if not equal_found:
                new_profiles[profile_count] = profile
                profile_count += 1
                res_group.add_result(profile, None)
                if profile_count == profile_limit:
                    break
        return new_profiles.base[:profile_count] 
        
            


    cdef public np.ndarray[np.int8_t, ndim=2] perform_2(self, np.ndarray[np.int8_t, ndim=2] water_orientations, np.uint16_t i, np.ndarray[np.uint16_t, ndim=1] order, np.ndarray[np.uint8_t, ndim=1] do_symmetry_check, signed char group, signed char invariant_level, list self_symmetry_groups):
        """
            The main method in the performing of the enumeration of proton configurations. Is an recursive method.
            Parameters:
                water_orientations  : the water orientations of previous step, can be None at the first step
                i                   : the number of molecule handled
                order               : the order at which the molecules are handled
                do_symmetry_check   : boolean array (1 or 0) that states at which molecules the symmetry is checked. Defaults to 1 for all if None is given
                group               : the number of the group handled. Only needed if starting from middle of the process
                invariant_level     : the depth of the invariants used
                self_symmetry_groups: the self symmetry groups found on previous iteration  
            
            Returns all possible water configurations for the given oxygen raft  
        """
        self.print_time_stats()
        cdef np.int_t[:, :, ::1] nearest_neighbors_nos = self.nearest_neighbors_nos
        cdef np.int_t[::1] nn = nearest_neighbors_nos[0, i]
        cdef np.ndarray[np.int8_t, ndim=2] new_water_orientations
        cdef int discarded, l
        cdef list symmetry_operations, original_symmetry_operations
        cdef float portion
        cdef bint save_self_symmetry_groups, init
        cdef SymmetryOperation so
        cdef np.uint8_t [:, :, ::1] possible_combinations = self.calculate_possible_combinations(i)
        self.possible_combinations = possible_combinations
        # Have all symmetryoperations been 'found' before. i.e., have all symmetry operations had
        #  complete sub symmetries.
        cdef bint all_symmetries_found = all_found(self.symmetry_operations)
        if order == None:
            order = np.arange(0, self.N, 1, dtype=np.uint16)
        if do_symmetry_check == None:
            do_symmetry_check = np.ones(self.N, dtype=np.uint8)
        if i == None:
            i = order[0]

        # Determine the order number of i
        number = 0
        for l,  n in enumerate(order):
            if n == i:
                number = l
                break        
        
        # Loaded from i certain state
        if i != order[0] and group == None and water_orientations == []:
            if rank == 0:
                print self.get_folder()+"intermediate_%i.txt" % i
                new_water_orientations = np.loadtxt(self.get_folder()+"intermediate_%i.txt" % i)
                new_water_orientations = np.array(new_water_orientations, dtype=np.int8)
            else:
                new_water_orientations = None
            if size > 1:
                new_water_orientations = comm.bcast(new_water_orientations, root = 0)
        else:
            print_parallel("Handling molecule number %i (order number %i)." % (i, number), self.logfile)
            print_("Handling molecule number %i (order number %i)." % (i, number), self.logfile)
            
            
            if group == None or group == -1:
                # Do the iteration
                #  - First handle charges
                if (self.charge != 0 or self.dissosiation_count != 0) and i == order[0]:
                    water_orientations = self.handle_charges()
                    self.result_group = ResultGroup(list(self.graph_invariants[invariant_level:]),  0)
                new_water_orientations = np.ndarray((0, water_orientations.shape[1]), dtype=np.int8)
                
                # handle the actual iteration of the orientations
                s = time()
                discarded = 0
                if water_orientations.shape[0] == 0 and i == order[0]:
                    if rank == 0:
                        init = True
                        new_water_orientations = self.handle_molecule_algorithm_2(i, None, possible_combinations, nn, init, &discarded)
                    if size > 1:
                        new_water_orientations = comm.bcast(new_water_orientations,  root = 0)
                else:
                    init = False
                    if not all_symmetries_found and size > 1:
                        water_orientations = comm.scatter(split_list(water_orientations,  size),  root = 0)
                        #print "All symmetries are not found, scattering %i wos" % len(water_orientations)
                    #print "Processor %i handles %i wos" % (rank, len(water_orientations))
                    new_water_orientations = self.handle_molecule_algorithm_2(i, water_orientations, possible_combinations, nn, init, &discarded)
                    
                    if not all_symmetries_found and size > 1:
                        new_water_orientations = merge_water_orientations(comm.allgather(new_water_orientations))
                        #print "All symmetries are not found, gathering %i wos" % len(new_water_orientations)
                if size > 1:
                    discarded = gather_and_sum(discarded)
                #print_parallel("Discarded %i geometries due to breakage of ice rules. Results %i geometries." % (discarded, new_water_orientations.shape[0]), self.logfile)
                self.iteration_time += time()-s
            else:
                if rank == 0:
                    print self.get_folder()+"group_%i_%i.txt" % (i, group)
                    new_water_orientations = np.loadtxt(self.get_folder()+"group_%i_%i.txt" % (i, group))
                    new_water_orientations = np.array(new_water_orientations, dtype=np.int8)
                else:
                    new_water_orientations = None
                if size > 1:
                    new_water_orientations = comm.bcast(new_water_orientations, root = 0)
            
            
            s = time()
            
            
            #Remove the symmetries
            # -First find the symmetries that can be possible, because of molecule changes 
            # -And handle the self symmetric groups (water orientations that have a parent that is symmetric with itself)
            # -Remove the symmetry operations that cannot be possible because their parents are not related by these symmetry operations
            # -And finally handle the normal cases
            if do_symmetry_check[i] == 1:
                
                if i == self.order[self.order.shape[0]-1]:
                    save_self_symmetry_groups = False
                else:
                    save_self_symmetry_groups = self.save_self_symmetry_groups
                
               
                if rank == 0:
                    original_symmetry_operations = do_symmetry_operation_filtering(self.symmetry_operations, i, order)
                    if self.save_self_symmetry_groups:
                        symmetry_operations = remove_earlier_found(original_symmetry_operations)
                        print_parallel("Removed %i earlier found operations" % (len(original_symmetry_operations) - len(symmetry_operations)))
                    else:
                        symmetry_operations = original_symmetry_operations
                else:
                    original_symmetry_operations = []
                    symmetry_operations = []
                if size > 1:
                    original_symmetry_operations = comm.bcast(original_symmetry_operations, root = 0)
                    symmetry_operations = comm.bcast(symmetry_operations, root = 0)
                
                if len(symmetry_operations) > 0:
                    portion  = (<float>  len(symmetry_operations) / <float> len(self.symmetry_operations)) 
                    if portion < 0.25:
                        portion = 0.25
                    else:
                        portion = 0.25
                    portion = 1.0 


                if self.graph_invariants is not None:
                    self.result_group = ResultGroup(list(self.graph_invariants[invariant_level:]),  0)
                    result_group_count = <int> ( portion * <float> len(self.graph_invariants) - 1)
                #print "Lenght before ssg %i, %i, %i" % (new_water_orientations.shape[0], len(self_symmetry_groups), rank)

                if len(self_symmetry_groups) > 0:
                    #print_("Total number of grouped wos at main thread before ssg-execution: %i" % self.result_group.get_total_number_of_grouped_wos(result_group_count), self.logfile)
                    print_parallel("Handling %i self symmetry groups" % len(self_symmetry_groups))
                    self_symmetry_groups, new_water_orientations = self.handle_self_symmetry_groups(new_water_orientations, i,  self_symmetry_groups, original_symmetry_operations, result_group_count, save_self_symmetry_groups, all_symmetries_found, possible_combinations)

                if len(symmetry_operations) > 0:
                    print_parallel("Checking symmetry for %i water orientations and %i symmetry operations" % (new_water_orientations.shape[0], len(symmetry_operations)))
                    
                    if self.invariant_count > 0:
                        new_water_orientations = self.remove_symmetric_results(new_water_orientations, i, symmetry_operations, result_group_count, self_symmetry_groups, save_self_symmetry_groups, [], None, 1, not all_symmetries_found)
                        #print_("Total number of grouped wos at main thread after everything: %i" % self.result_group.get_total_number_of_grouped_wos(result_group_count), self.logfile)
                    else:
                        new_water_orientations = self.remove_symmetries_no_invariants(new_water_orientations,   nearest_neighbors_nos, i, original_symmetry_operations, self_symmetry_groups, save_self_symmetry_groups)
                    
                        
                    if save_self_symmetry_groups:
                        print_parallel("Marking %i symmetry operations earlier found" % len(symmetry_operations))
                        mark_earlier_found(symmetry_operations, self.symmetry_operations)                        
                        all_symmetries_found = all_found(self.symmetry_operations)
                        # If all symmetry operations have been found, do the final scatter
                        if all_symmetries_found:
                            if size > 1:
                                new_water_orientations = comm.scatter(split_list(new_water_orientations,  size),  root = 0)
                            #self_symmetry_groups = comm.scatter(split_list(self_symmetry_groups, size), root = 0)
                else:
                    print_parallel("Skipping symmetry checking.", self.logfile)
                    print_("Skipping symmetry checking.", self.logfile)
                    print "Finally %i geometries in main thread." % len(new_water_orientations), None
                
              
            
            #self.symmetry_total_time = (time()-s)
            self.save_results(i, new_water_orientations, order, all_symmetries_found = all_symmetries_found)
        
        
        
        #Call the next round (i=i + 1) or return the results
        if(number+1 != len(nearest_neighbors_nos[0])):
            #print "Here again?"
            return self.perform_2(new_water_orientations, order[number+1], order, do_symmetry_check, -1, 0, self_symmetry_groups)
        else:
            self.print_time_stats()
            return new_water_orientations

    cdef int get_coordination_number(self, int i):
        
        cdef np.int_t[:, :, ::1] nearest_neighbors_nos = self.nearest_neighbors_nos
        cdef int j, N = 4, result = 4
        for j in range(N):
            if nearest_neighbors_nos[0, i, j] == i and not nearest_neighbors_nos[1, i, j]:
                result -= 1
        return result

    def get_sample_charge(self, water_orientations):
        result = 0
        for orientation in water_orientations:
            if orientation > 9:
                result += 1
            elif orientation > 5:
                result -= 1
        return result

    def handle_charges(self):
        cdef np.int8_t H3O_count = self.dissosiation_count + (abs(self.charge) + self.charge) / 2
        cdef np.int8_t OH_count = self.dissosiation_count + (abs(self.charge) - self.charge) / 2
        if OH_count == 0 and H3O_count == 0:
            return np.zeros((0, self.N),  dtype=np.int8)

        cdef np.ndarray[np.int8_t, ndim=2] water_orientations = np.zeros((0, self.N),  dtype=np.int8)
        if H3O_count != 0:
            print_parallel("Handling H3O+ molecules", self.logfile)
            print_parallel("---------------------------------------------", self.logfile)
            water_orientations = self.handle_charge(water_orientations, H3O_count, np.array([10, 11, 12, 13], dtype=np.int8))
        if OH_count != 0:
            print_parallel("Handling OH- molecules", self.logfile)
            print_parallel("---------------------------------------------", self.logfile)
            water_orientations = self.handle_charge(water_orientations, OH_count, np.array([6, 7, 8, 9], dtype=np.int8))
        
        return water_orientations

    
        

    def handle_charge(self, np.ndarray[np.int8_t, ndim=2]  water_orientations, np.int8_t charge_left, np.ndarray[np.int8_t, ndim=1] orientations):
        """
            charge_left means the number of charges particles left to be set
            orientations means the orientations possible for this particle (OH or H30)
        """    
        
        iteration = 1
        while charge_left > 0:
            
            print_parallel("##    Handling charged particle no %i     ##" % (iteration), self.logfile)
            print_parallel("--------------------------------------------", self.logfile)
            charge_left -= 1
            iteration += 1
            discarded = 0
            new_water_orientations = np.zeros((0, self.N),  dtype=np.int8)
            if len(water_orientations) == 0:
                if rank == 0:
                    water_orient = np.ndarray((len(self.nearest_neighbors_nos[0])), dtype=np.int8)
                    water_orient.fill(-1)
                    new_water_orientations, discarded = self.handle_single_charge(water_orient, new_water_orientations, orientations)
                    if size > 1:
                        new_water_orientations = comm.bcast(new_water_orientations,  root = 0)
            else:
                if size > 1:
                    water_orientations = comm.scatter(split_list(water_orientations,  size),  root = 0)
                for l, water_orient in enumerate(water_orientations):
                    new_water_orientations, d = self.handle_single_charge(water_orient, new_water_orientations, orientations)
                    discarded += d
                new_water_orientations = merge_water_orientations(comm.allgather(new_water_orientations))
            if size > 1:
                discarded = gather_and_sum(discarded)
            print_parallel("Discarded %i geometries due to breakage of ice rules" % (discarded), self.logfile)
            
            print_(new_water_orientations, self.logfile) 
            
            self.result_group = ResultGroup(self.graph_invariants,  0)
            water_orientations = self.remove_symmetric_results(new_water_orientations,  -1, self.symmetry_operations, -1, [], False, [], None, 20, True) #self.remove_symmetries_no_invariants(new_water_orientations,   nearest_neighbors_nos, i)    
            self.print_time_stats()
        return water_orientations

    def handle_single_charge(self, np.ndarray[np.int8_t, ndim=1] water_orient, np.ndarray[np.int8_t, ndim=2] water_orientations, np.ndarray[np.int8_t, ndim=1] orientations):
        cdef int discarded = 0, i, coodination_number
        cdef np.int8_t orientation
        for i in range(self.N):
            coordination_number = self.get_coordination_number(i)
            if water_orient[i] != -1 or (coordination_number == 4 and orientations[0] > 9):
                continue
            for orientation in orientations:
                s = time()
                if self.water_orientation_is_valid(orientation, water_orient, i):
                    water_orientations_trial = np.copy(water_orient)
                    water_orientations_trial[i] = orientation
                    water_orientations = np.vstack((water_orientations,  water_orientations_trial))
                else:
                    discarded += 1
                self.iteration_time += time()-s
        return water_orientations, discarded
    
    cdef np.uint8_t[:, :, ::1] calculate_possible_combinations(self, np.uint16_t molecule_no):
        cdef np.int_t[::1] nn = self.nearest_neighbors_nos[0, molecule_no]
        cdef np.int8_t[::1] trial, empty_water_orient = np.ndarray( self.N, dtype=np.int8)
        empty_water_orient[:] = -1
        cdef int max_wo
        cdef bint allowed_by_charge
        if self.charge != 0 or self.dissosiation_count != 0:
            max_wo = 14
        else:
            max_wo = 6
        cdef np.uint8_t[:, :, ::1] result = np.ndarray((nn.shape[0], max_wo, max_wo), dtype=np.uint8) 
        cdef int wo, wo_2, i, neighbor_no, N = nn.shape[0]
        cdef bint all_impossible, allowed_by_preset
        cdef np.int8_t[::1] allowed_by_presets = np.ndarray(max_wo, dtype=np.int8)
        for wo from 0 <= wo < max_wo:
            allowed_by_preset = self.wo_is_allowed_by_preset_bonds(molecule_no, wo)
            allowed_by_presets[wo] = allowed_by_preset
            # is allowed by charge and dissosiation?
            if wo > 5 and wo < 10 and self.charge >= 0 and self.dissosiation_count == 0:
                allowed_by_presets[wo] = 0
            elif wo >= 10 and self.charge <= 0 and self.dissosiation_count == 0:
                allowed_by_presets[wo] = 0
            #if not allowed_by_preset:
            #    print "Bond %i - %i, wo %i not allowed by preset" % (molecule_no, neighbor_no, wo)
        for i from 0 <= i < N:
            neighbor_no = nn[i]
            for wo from 0 <= wo < max_wo:
                # next row is needed for zero, one, and two coordinated molecules
                all_impossible = self.wo_is_equal_with_any_of_previous_wos(molecule_no, wo)
                allowed_by_preset = allowed_by_presets[wo]
                for wo_2 from 0 <= wo_2 < max_wo:
                    trial = empty_water_orient.copy()
                    trial[neighbor_no] = wo_2
                    allowed_by_charge = wo_2 < 6 or (wo_2 > 5 and wo_2 < 10 and (self.charge < 0 or self.dissosiation_count != 0)) or (wo_2 >= 10 and (self.charge > 0 or self.dissosiation_count != 0))
                    if allowed_by_charge and not all_impossible and allowed_by_preset and self.water_orientation_is_valid(wo, trial, molecule_no):
                        result[i, wo, wo_2] = 1
                    else:
                        result[i, wo, wo_2] = 0  
        return result

    def preset_bonds_list_to_dict(self, list preset_bonds_list):
        """
            Converts preset bonds list to dict, the input list has to be
            two dimensional and the second dimension must have either 3 or 4 values
            where first and second are the molecules of the bond and the last is the value
            of the bond.
        """
        cdef dict result = None
        if preset_bonds_list != None:
            result = {}
            for p in preset_bonds_list:
                if len(p) < 3 or len(p) > 4:
                    import sys
                    sys.exit("Invalid preset bonds: The preset bonds list value (%s) is of the wrong size." % p)
                if len(p) == 3 and p[2] != 1 and p[2] != -1:
                    import sys
                    sys.exit("Invalid preset bonds: The preset bond value (%i) for bond (%i-%i) is not valid. The value should be 1 or -1." % (p[2], p[0], p[1]))
                if len(p) == 4 and p[3] != 1 and p[3] != -1:
                    import sys
                    sys.exit("Invalid preset bonds: The preset bond value (%i) for bond (%i-%i_%i) is not valid. The value should be 1 or -1." % (p[2], p[0], p[1], p[2]))
                if p[0] not in result:
                    result[p[0]] = {}
                if p[1] not in result[p[0]]:    
                    if len(p) == 4:
                        result[p[0]][p[1]] = {}
                        result[p[0]][p[1]][p[2]] = p[3]
                    else:
                        result[p[0]][p[1]] = p[2]
                else:
                    if len(p) == 4:
                        result[p[0]][p[1]][p[2]] = p[3]
                    else:
                        result[p[0]][p[1]] = p[3]
                        #raise Exception("There are two values for same bond in preset bonds list")
                        
        return result
        
            
        

    cdef bint wo_is_allowed_by_preset_bonds(self, np.uint16_t molecule_no,  int water_orientation):
        if self.preset_bond_values == None:
            return True
        cdef np.int8_t[::1] bvv = get_bond_variable_values_from_water_orientation(water_orientation)
        cdef np.int_t[::1] nn = self.nearest_neighbors_nos[0, molecule_no]
        cdef np.int_t[::1] nnp = self.nearest_neighbors_nos[2, molecule_no]
        cdef np.int_t[:, :, ::1] nearest_neighbors_nos = self.nearest_neighbors_nos
        cdef list ones = []
        cdef bint equals, found
        cdef dict preset_bond_values = self.preset_bond_values
        cdef int i, j, k, wo = 0, N = bvv.shape[0]
        cdef np.uint8_t neighbor_no
        for k from 0 <= k < N:
            neighbor_no = nn[k]
            if molecule_no in self.preset_bond_values and neighbor_no in self.preset_bond_values[molecule_no]:
                nn = nearest_neighbors_nos[0, molecule_no]
                nnp = nearest_neighbors_nos[2, molecule_no]
                found = False
                for i from 0 <= i < N:
                    if not found and nn[i] == neighbor_no:
                        if (type(preset_bond_values[molecule_no][neighbor_no]) == dict and nnp[i] in preset_bond_values[molecule_no][neighbor_no]): 
                            found = True
                            if (bvv[i] != preset_bond_values[molecule_no][neighbor_no][nnp[i]]):
                                return False
                        elif (type(preset_bond_values[molecule_no][neighbor_no]) != dict and nnp[i] == 13): 
                            found = True
                            if bvv[i] != preset_bond_values[molecule_no][neighbor_no]:
                                return False
                        
                        
            elif neighbor_no in preset_bond_values and molecule_no in preset_bond_values[neighbor_no]:
                nn = self.nearest_neighbors_nos[0, molecule_no]
                nnp = self.nearest_neighbors_nos[2, molecule_no]
                for i from 0 <= i < N:
                    if nn[i] == neighbor_no:
                        if (type(preset_bond_values[neighbor_no][molecule_no]) == dict and  get_opposite_periodicity_axis_number(nnp[i]) in preset_bond_values[neighbor_no][molecule_no]):         
                            if (bvv[i] == preset_bond_values[neighbor_no][molecule_no][get_opposite_periodicity_axis_number(nnp[i])]):
                                return False
                        elif (type(preset_bond_values[neighbor_no][molecule_no]) != dict and nnp[i] == 13): 
                            if bvv[i] == self.preset_bond_values[neighbor_no][molecule_no]:
                                return False  
        return True

    cdef bint wo_is_equal_with_any_of_previous_wos(self, np.uint16_t molecule_no, int water_orientation):
        cdef int coordination_number = self.get_coordination_number(molecule_no)
        cdef np.int8_t[::1] nbbv, bvv = get_bond_variable_values_from_water_orientation(water_orientation)
        cdef np.int_t[:, :, ::1] nearest_neighbors_nos = self.nearest_neighbors_nos
        cdef list ones = []
        cdef bint equals
        cdef int i, wo = 0, N = bvv.shape[0]
        if coordination_number < 3:
            for i in xrange(N):
                if bvv[i] == 1:
                    ones.append(nearest_neighbors_nos[0, molecule_no, i])
            while wo < water_orientation:
                equals = True
                nbbv = get_bond_variable_values_from_water_orientation(wo)
                for i in xrange(N):
                    if nbbv[i] == 1 and nearest_neighbors_nos[0, molecule_no, i] not in ones:
                        equals = False
                        break
                if equals:
                    return True
                
                wo += 1
        return False

    cpdef bint additional_requirements_met(self, signed char water_orientation, np.int8_t [::1] water_orient, np.uint16_t molecule_no):
        return True
            

    cdef bint water_orientation_is_valid(self, signed char water_orientation, np.int8_t[::1] water_orient, np.uint16_t molecule_no):
        """
             go through all neighbors and check that there are no breakage of ice rules
                if there is: return False
                else return True

            not used by the actual process but used by calculate_possible_combinations -method. So it can not be deleted!
        """
        cdef np.int_t[::1] nn = self.nearest_neighbors_nos[0, molecule_no]
        cdef np.int_t[:, :, ::1]  nearest_neighbors_nos = self.nearest_neighbors_nos
        cdef np.int_t[::1] nps = self.nearest_neighbors_nos[1, molecule_no]
        cdef np.int8_t[::1] bvv = get_bond_variable_values_from_water_orientation(water_orientation)
        cdef np.int8_t[::1] nbbv
        #assert not self.periodic or len(nn)==4
        cdef np.uint8_t periodic, periodicity_axis,  nearest_neighbor_no, opposite, n
        cdef np.int8_t neighbor_bv_value
        cdef bint neighbor_set
        cdef int l, x, N = nn.shape[0], M
        cdef int coordination_number = self.get_coordination_number(molecule_no)
        if coordination_number == 0:
            return True
        for l from 0 <= l < N:
            nearest_neighbor_no = nn[l]
            periodic = nps[l]
            periodicity_axis = nearest_neighbors_nos[2, molecule_no, l]
            neighbor_set = True
            neighbor_bv_value = 0
            opposite = get_opposite_periodicity_axis_number(periodicity_axis)
            # get corresponding water_orientation_value of neighbor and store it to neighbor_bv_value 
            #   -first check that the neighbor has a value
            #      - if not then no breakage happens
            if water_orient[nearest_neighbor_no] != -1:
                # current neighbors bond variable values
                nbbv = get_bond_variable_values_from_water_orientation(water_orient[nearest_neighbor_no])
                M = nearest_neighbors_nos[0, nearest_neighbor_no].shape[0]
                for x from 0 <= x < M:
                    n = nearest_neighbors_nos[0, nearest_neighbor_no, x]
                    # find the neighbor that corresponds to the molecule currently handled (i) [and has the same periodicity and the periodicity axis correspond]
                    if n == molecule_no and nearest_neighbors_nos[1, nearest_neighbor_no, x] == periodic and nearest_neighbors_nos[2, nearest_neighbor_no, x] == opposite:                       
                        neighbor_bv_value = nbbv[x]
                        break
            else:
                neighbor_set = False
            if water_orientation > 9 and nearest_neighbor_no == molecule_no and bvv[l] != -1:
                return False
             # if both bond variables have the same value, then we have not succeeded
            if neighbor_set and bvv[l] == neighbor_bv_value:
                return False 
                
        return True

    cdef inline int water_orientation_is_valid_using_possible_combinations(self, signed char water_orientation, np.ndarray[np.int8_t, ndim=1] water_orient, np.uint16_t molecule_no):
        """
            Checks if the given water orientation for molecule with number molecule_no
             is valid in environment of orientations given in water_orient
            DEPRECATED! The 'static' method is used.
        """
        cdef np.ndarray[np.int_t, ndim=1] nn = self.nearest_neighbors_nos[0, molecule_no] 
        cdef int neighbor_no, i, combination_possible, N = nn.shape[0]
        cdef np.int8_t neighbor_orientation
        cdef np.ndarray[np.uint8_t, ndim=3] possible_combinations = self.possible_combinations 
        for i from 0 <= i < N:
            neighbor_no = nn[i]
            neighbor_orientation = water_orient[neighbor_no]
            if neighbor_orientation == -1:
                continue
            combination_possible = possible_combinations[i, water_orientation, neighbor_orientation]
            if combination_possible == 0:
                return 0
        return 1     

    cdef np.int8_t[::1] get_first_possible_combination(self, np.uint8_t [:, :, :, ::1] possible_combinations):
        """
            Handles the 'ice rules' check part only to get one water orientations
        """
        cdef int i = 0, N = self.N, wo_max = 6, totalOH = 0, totalH3O = 0, maxH3O, maxOH, charge = self.charge, dissosiation_count = self.dissosiation_count, key, water_orientation
        maxH3O = dissosiation_count
        maxOH = dissosiation_count
        if charge > 0:
            maxH3O += charge
        elif charge < 0:
            maxOH -= charge
        cdef np.int8_t[::1] water_orientations = np.zeros(N, dtype=np.int8)
        cdef np.int8_t[::1] is_preset = np.zeros(N, dtype=np.int8)
       
        water_orientations[:] = -1
        cdef np.int_t[::1] nn
        
        cdef bint preset
        cdef np.int_t[:, :, ::1] nearest_neighbors_nos = self.nearest_neighbors_nos
        cdef bint allowed_by_charge
        if self.preset_water_orientations is not None:
            for key in self.preset_water_orientations:
                water_orientation = self.preset_water_orientations[key]
                is_preset[key] = 1
                water_orientations[key] = water_orientation
                if water_orientation > 9:
                    totalH3O += 1
                elif water_orientation > 5:
                    totalOH += 1
    
        water_orientation = -1            
        if (self.charge != 0 or dissosiation_count != 0) and maxOH != totalOH and maxH3O != totalH3O:
            wo_max = 14
        while i < N:
            preset = is_preset[i]
            nn = nearest_neighbors_nos[0, i]
            if i < 5:
                print(water_orientations.base)
                #sys.stdout.write("\033[F") # Cursor up one line
            
            if water_orientation == wo_max - 1:
                while True:
                    if not is_preset[i]:
                        water_orientations[i] = -1
                    i -= 1
                    
                    if i < 0:
                        print_parallel("Water orientation for given profile does not exist.", self.logfile)
                        return None
                    if not is_preset[i]:
                        water_orientation = water_orientations[i]
                        if water_orientation > 9:
                            totalH3O -= 1
                        elif water_orientation > 5:
                            totalOH -= 1
                        break
                    
            else:
                water_orientation += 1
                if water_orientation < 6:
                    allowed_by_charge = True
                elif water_orientation > 9:
                    allowed_by_charge = totalH3O < maxH3O
                else:
                    allowed_by_charge = totalOH < maxOH
                    
                if allowed_by_charge and water_orientation_is_valid_using_possible_combinations(water_orientation, water_orientations, i, possible_combinations[i], nn, False) and self.additional_requirements_met(water_orientation, water_orientations, i):
                    water_orientations[i] = water_orientation 
                    if water_orientation > 9:
                        totalH3O += 1
                    elif water_orientation > 5:
                        totalOH += 1
                    water_orientation = -1
                    while True:
                        i += 1
                        if i > N -1:
                            break
                        if not is_preset[i]:
                            break
                        
                        
        return water_orientations             
        
    cdef np.ndarray[np.int8_t, ndim=2] handle_molecule_algorithm_2(self, np.uint16_t i, np.int8_t[:, ::1] water_orientations, np.uint8_t [:, :, ::1] possible_combinations, np.int_t[::1] nn, bint init, int *discarded):
        """
            Handles the 'ice rules' check part of the enumeration. Also initializes the water_orientations object if the input is None.
            Parameters:
                i                    : number of molecule in the raft
                water_orientations   : the water orientations lists in two dimensional (numpy) array
                possible_combinations: the combinations that are possible for this molecule and the neighbors orientations, calculated by  'calculate_possible_combinations' -method
        """
        cdef int j, count = 0, N
        cdef np.int8_t[:, ::1] new_water_orientations 
        cdef np.int8_t water_orientation
        cdef np.int8_t[::1] water_orientations_trial
        cdef np.int8_t[::1] water_orient
        if init:
            N = 1
            water_orientations = np.ndarray((1, self.N), dtype=np.int8)
            water_orientations[:] = -1
            new_water_orientations = np.ndarray((6, self.N), dtype=np.int8)
        else:
            if water_orientations is not None:
                N = water_orientations.shape[0]
            else:
                N = 0
            new_water_orientations = np.ndarray((6*N, self.N), dtype=np.int8)
            
        for j from 0 <= j < N:
            water_orient = water_orientations[j]
            if water_orient[i] != -1:
                new_water_orientations[count] =  water_orient
                count += 1
            else:
                for water_orientation from 0 <= water_orientation < 6:
                    #if  self.water_orientation_is_valid(water_orientation, water_orient, i) and self.additional_requirements_met(water_orientation, water_orient, i):
                    
                    if  water_orientation_is_valid_using_possible_combinations(water_orientation, water_orient, i, possible_combinations, nn, init) == 1 and self.additional_requirements_met(water_orientation, water_orient, i):
                        water_orientations_trial = water_orient.copy()
                        water_orientations_trial[i] = water_orientation
                        new_water_orientations[count] = water_orientations_trial
                        count += 1
                    else:
                        discarded[0] += 1
                      
                    #if success:# and not self.is_symmetric_with_another_result(water_orientations_trial, wo, [bond_variables_trial,  periodic_bond_variables_trial],  bv,   nearest_neighbors_nos):
                    #new_results.append(result_trial)
        return new_water_orientations.base[0:count]

    cdef ResultGroup single_wo_initial_grouping(self, np.int8_t[::1] water_orientation, np.int8_t[:, ::1] bond_variables):
        """
            Groups a single water orientation list until the first level of result groups
            Parameters:
                the water orientations of a single proton configuration
            Returns the group in  which the water orientation belongs to
        """
        cdef ResultGroup rg
        #cdef float s
        
        #s = time()
        cdef int N = water_orientation.shape[0]
        cdef np.int_t[:, :, ::1] nearest_neighbors_nos = self.nearest_neighbors_nos
        get_bond_variables_3(water_orientation, nearest_neighbors_nos, N, bond_variables)
        #self.conversion_time += time() - s
        rg = self.result_group._try_subgroups(bond_variables,  1)
        
        rg.add_result(water_orientation, None)
        return rg



    cdef public tuple do_initial_grouping(self, np.ndarray[np.int8_t, ndim=2] water_orientations, int depth_limit, bint scatter):
        if scatter and size > 1:
            water_orientations = comm.scatter(split_list(water_orientations, wanted_parts=size), root=0)
        cdef int wo_count = water_orientations.shape[0]
        cdef np.int8_t[:, ::1] bond_variables = np.ndarray((self.N, 4), dtype=np.int8), wos
        cdef np.int8_t[::1] water_orientation
        cdef ResultGroup group, rg
        cdef list groups, new_groups
        # Get the result groups from 1st level
        # NOTE: this can't be done to any other level
        cdef int i, counter = 0
        for i from 0 <= i < wo_count:
            water_orientation = water_orientations[i]
            rg = self.single_wo_initial_grouping(water_orientation, bond_variables)
        groups = self.result_group.get_subgroups_from_level(0)
        cdef int depth = 1
        cdef int N = water_orientations.shape[1]
        #for f, group in enumerate(groups):
        #    print_parallel("-Group %i has %i results" % (f, len(group.get_wos())))
        
        #if not scatter:
        #    groups = merge(comm.allgather(groups))
        #    groups = merge_groups(groups)
        if size > 1 and scatter:
            # Remove equal groups
            groups = comm.gather(groups, root = 0)
            if rank == 0:
                # merge the list
                groups = merge(groups)
                #for f, group in enumerate(groups):
                #    print_parallel("Group %i has %i results" % (f, len(group.get_wos())), self.logfile)
                # merge equal groups
                
                groups = merge_groups(groups)
                group_count = len(groups)
            else:
                group_count = -1
                groups = []

            #for f, group in enumerate(groups):
            #    print_parallel("Group %i has %i results" % (f, len(group.get_wos())))
            groups = comm.scatter(split_list(groups,  size),  root = 0)
            group_count = comm.bcast(group_count, root = 0)
            #print_parallel("First scatter done: split %i groups for %i processors. Total %i wos handled." % (group_count, size, counter))
            
            # Do scattering as long as each processor has at least 3 groups
            #   - if number of results is over 500
            # This is done to equalize the load for each processor
            
            while wo_count > 500 and group_count < 3*size and depth < 20 and depth < depth_limit and size != 1:
                new_groups = []
                for group in groups:
                    wos = group.get_wos()
                    for i in range(wos.shape[0]):
                        if group.bvs != None and len(group.bvs) > 0:
                            rg = group._try_subgroups(group.bvs[i], 1)
                            rg.add_result(wos[i], group.bvs[i])
                        else:
                            get_bond_variables_3(wos[i],  self.nearest_neighbors_nos, N, bond_variables)
                            rg = group._try_subgroups(bond_variables.base, 1)
                            rg.add_result(wos[i], None)
                    group.bvs = None
                    group.wos = None
                    new_groups.extend(group.get_subgroups_from_level(0))
                groups = comm.gather(new_groups, root = 0)
                group_count = -1
                if rank == 0:
                    groups = merge(groups)
                    group_count = len(groups)
                    if group_count > 3*size:
                        for i, group in enumerate(groups):
                            self.save_results(0, group.get_wos(), None, group_save=True, group_number=i)
                    groups = split_list(groups, size)
                
                group_count = gather_and_max(group_count)
                
                groups = comm.scatter(groups,  root = 0)
                depth += 1
            
            #print_parallel("Results split in %i groups after %i iterations" % (group_count, depth))

        return groups, depth

    cdef tuple handle_self_symmetry_group(self, SelfSymmetryGroup self_symmetry_group, list symmetry_operations, list new_symmetry_operations, np.int8_t[:, ::1] water_orientations, list new_self_symmetry_groups, int result_group_count, np.uint16_t molecule_no, ResultGroup main_result_group, bint save_self_symmetry_groups, bint all_symmetries_found, np.int_t[::1] nn, int child_level, np.uint8_t [:, :, ::1] possible_combinations):
        """
            Handles a single self symmetry group and it's subgroups
            Parameters:
                symmetry_operations: all possible symmetry operations at current phase of execution, before removing earlier found
                self_symmetry_group: the self symmetry group handled
                new_symmetry_operations: the operations that are new in the current phase of execution
                water_orientations: the water orientations from main thread
                new_symmetry_groups: new symmetry groups found during handling of all self symmetry groups
        """
       
        cdef np.int8_t[:, ::1] new_water_orientations, wos
        cdef list tested_symmetry_operations, leftover_symmetry_operations
        cdef int i, j, k, l, O, Q, P = len(self_symmetry_group.child_self_symmetry_groups), discarded = 0
        cdef list child_self_symmetry_groups = []
        cdef bint scatter = False
        #cdef np.ndarray[np.int8_t, ndim=2] bond_variables
        cdef ResultGroup rg, group
        cdef SelfSymmetryGroup child_self_symmetry_group, parent_self_symmetry_group
        tested_symmetry_operations, leftover_symmetry_operations = self_symmetry_group.get_tested_and_leftover_symmetry_operations(symmetry_operations)
        cdef list graph_invariants = self.graph_invariants
        # Do iteration of water orientations
        if size > 1:
            wos = comm.scatter(split_list(self_symmetry_group.water_orientations,  size),  root = 0)
        else:
            wos = self_symmetry_group.water_orientations
        new_water_orientations = self.handle_molecule_algorithm_2(molecule_no, wos, possible_combinations, nn, False, &discarded)
        if size > 1:
            discarded = gather_and_sum(discarded)
            new_water_orientations = merge_water_orientations(comm.allgather(new_water_orientations.base))
        #print_parallel(new_water_orientations.base, self.logfile)
        #print_parallel("Discarded %i geometries due to breakage of ice rules, resulted %i geometries" % (discarded, new_water_orientations.shape[0]), self.logfile)
        self_symmetry_group.water_orientations = <np.ndarray[np.int8_t, ndim=2]>new_water_orientations.base
        group = None
        # Determine the right level of indentation in output
        cdef str child_empty = ""
        cdef int child_l = 0
        while child_l <= child_level:
            child_empty += "    "
            child_l += 1

        # Only using the current result group if the group is left alive, otherwise a fresh result group is used
        if  len(leftover_symmetry_operations) == 0 and not self_symmetry_group.finalized:
            group = self.result_group
            if size > 1:
                scatter = True
            self.result_group = ResultGroup(graph_invariants,  0)
            #print_(child_empty + "Using fresh result group, rank %i" % rank, self.logfile)
        
        
        # Handle child self symmetry groups
        for i from 0 <= i < P:
            #print_(child_empty + "-------------------------------------------------", self.logfile)
            #print_(child_empty + "Handling child self symmetry group %i of %i " % (i+1, P), self.logfile)
            #print_(child_empty + "-------------------------------------------------", self.logfile)
            child_self_symmetry_group = <SelfSymmetryGroup>self_symmetry_group.child_self_symmetry_groups[i]
            self.handle_self_symmetry_group(child_self_symmetry_group, symmetry_operations, new_symmetry_operations, water_orientations, new_self_symmetry_groups, result_group_count, molecule_no, main_result_group, save_self_symmetry_groups, all_symmetries_found, nn, child_level + 1, possible_combinations)
            #print_(child_empty + "-------------------------------------------------", self.logfile)
        cdef SymmetryOperation symo
        new_water_orientations = self_symmetry_group.water_orientations
        #print P, rank, new_water_orientations
        if new_water_orientations.shape[0] > 0:
            
            # Although ther might not be any in tested_symmetry_operations the symmetry check must be performed
            # to group the results in this group, so that symmetries with parent groups water orientaions and 
            # symmetry operations can be detected.
            tested_symmetry_operations.extend(new_symmetry_operations)
            tested_symmetry_operations.extend(self_symmetry_group.get_active_parent_symmetry_operations(symmetry_operations))
            
            new_water_orientations = self.remove_symmetric_results(new_water_orientations, molecule_no, tested_symmetry_operations, result_group_count, child_self_symmetry_groups, save_self_symmetry_groups, None, self_symmetry_group, 20, scatter)
               
            # Clear the initial grouping
            self.result_group.clear_wos_from_level(1)

            if len(leftover_symmetry_operations) > 0 and len(new_symmetry_operations) > 0:
                new_water_orientations = self.remove_symmetric_results(new_water_orientations, molecule_no, new_symmetry_operations, result_group_count, child_self_symmetry_groups, save_self_symmetry_groups, None, self_symmetry_group, 1, False)
        
        self.result_group.clear_wos_from_level(1)
        self_symmetry_group.child_self_symmetry_groups.extend(child_self_symmetry_groups)   
        new_self_symmetry_groups.extend(child_self_symmetry_groups)
        if len(leftover_symmetry_operations) == 0 and not self_symmetry_group.finalized:
            # Finalize the group
            # Switch back to original ResultGroup
            self.result_group = group
            #print_(child_empty + "Falling back to previous group, rank %i" % rank, self.logfile)
            
            parent_self_symmetry_group = self_symmetry_group.get_active_parent_self_symmetry_group()
            self_symmetry_group.finalized = True
            #print_parallel(child_empty + "FINALISING SELF SYMMETRY GROUP %i" % rank, self.logfile)
            if parent_self_symmetry_group is None:
                # all parent groups are finalized, adding wos to main thread
                #print_(child_empty + "to main thread, rank: %i" % rank, self.logfile)
                if all_symmetries_found and size > 1:
                    water_orientations = stack(water_orientations, comm.scatter(split_list(new_water_orientations.base,  size),  root = 0))
                else:
                    water_orientations = stack(water_orientations, new_water_orientations)
            else:
                # parent group found, putting wos to it
                #print_(child_empty + "to parent, rank: %i" % rank, self.logfile)
                parent_self_symmetry_group.water_orientations = <np.ndarray[np.int8_t, ndim=2]> stack(parent_self_symmetry_group.water_orientations, new_water_orientations).base
            #print_(child_empty + "%s" % new_water_orientations.base, self.logfile) 
            # Finally, assert that there are no leftovers
            self_symmetry_group.water_orientations = None #np.ndarray((0, water_orientations.shape[1]), dtype=np.int8)
            
            
            
             
            # Group the new child groups to main ResultGroup
            self.result_group = main_result_group
            Q = len(child_self_symmetry_groups)
            #print_symmetry_operations(tested_symmetry_operations)
            #print_(child_empty + "Grouping %i child self symmetry groups to main result group" % Q, self.logfile)
            for j from 0 <= j < Q:
                child_self_symmetry_group = <SelfSymmetryGroup> child_self_symmetry_groups[j]
                child_self_symmetry_group.water_orientations = self.remove_symmetric_results(child_self_symmetry_group.water_orientations, molecule_no, tested_symmetry_operations, result_group_count, None, False, None, child_self_symmetry_group, 1, False)
                #print_(child_self_symmetry_group.water_orientations, self.logfile)
                self.result_group.clear_wos_from_level(1)
            #    #bond_variables = get_bond_variables_3(child_self_symmetry_group.water_orientations[0],  self.nearest_neighbors_nos)
            #    #rg = main_result_group.try_subgroups(bond_variables, 5)
            #    #rg.add_result(child_self_symmetry_group.water_orientations[0], None)
            self.result_group = group
                   
        else:
            #print_(child_empty + "GROUP LEFT ALIVE", self.logfile)
            self_symmetry_group.symmetry_operations = leftover_symmetry_operations 
            self_symmetry_group.water_orientations = new_water_orientations.base
            new_self_symmetry_groups.append(self_symmetry_group)
        
        return new_self_symmetry_groups, water_orientations
            
                   
    
    cdef tuple handle_self_symmetry_groups(self, np.int8_t[:, ::1] water_orientations, np.uint16_t molecule_no, list self_symmetry_groups, list symmetry_operations, int result_group_count, bint save_self_symmetry_groups, bint all_symmetries_found, np.uint8_t [:, :, ::1] possible_combinations):
        
        cdef int i, j, k, l, m, discarded, N = len(self_symmetry_groups), M, L, O, P, total = 0, tested
        cdef SymmetryOperation symmetry_operation
        cdef SelfSymmetryGroup self_symmetry_group, child_self_symmetry_group, parent_self_symmetry_group
        cdef ResultGroup group
        
        cdef np.uint8_t are_symmetric, is_self_symmetric
        cdef bool add
        cdef list new_self_symmetry_groups = []
        
        cdef list nchsg, all_child_self_symmetry_groups, all_new_children, new_cssgs
        cdef list new_symmetry_operations
        cdef list tested_symmetry_operations, leftover_symmetry_operations, active_symmetry_operations
        cdef np.int_t[::1] nn = self.nearest_neighbors_nos[0, molecule_no]
        if molecule_no == self.order[self.order.shape[0]-1]:
            save_self_symmetry_groups = False
        
        if rank == 0:
            new_symmetry_operations = remove_earlier_found(symmetry_operations)
        else:
            new_symmetry_operations = None
        if size > 1:
            new_symmetry_operations = comm.bcast(new_symmetry_operations, root = 0)
        s = time()
        cdef tuple res
        cdef ResultGroup result_group = self.result_group
        for i from 0 <= i < N:
            discarded = 0
            if True:
                print_("-------------------------------------------------", self.logfile)
                print_("Handling self symmetry group %i of %i " % (i+1, N), self.logfile)
                print_("-------------------------------------------------", self.logfile)
            
            self_symmetry_group = <SelfSymmetryGroup>self_symmetry_groups[i]
            res = self.handle_self_symmetry_group(self_symmetry_group, symmetry_operations, new_symmetry_operations, water_orientations,  new_self_symmetry_groups, result_group_count, molecule_no, result_group, save_self_symmetry_groups, all_symmetries_found, nn, 0, possible_combinations)
            new_self_symmetry_groups = <list>res[0] 
            water_orientations = res[1] 
            print_("-------------------------------------------------", self.logfile)
       
        self.self_symmetry_time += time() - s     
        self.result_group.clear_wos_from_level(1)# = ResultGroup(list(self.graph_invariants),  0)
        #print "Total %i water orientations in self symmetry_groups" % (total + len(new_self_symmetry_groups))
        cdef list result = []
        cdef int o, E = len(new_self_symmetry_groups)
        for o from 0 <= o < E:
            self_symmetry_group = <SelfSymmetryGroup>new_self_symmetry_groups[o]
            parent_self_symmetry_group = self_symmetry_group.get_active_parent_self_symmetry_group()
            if parent_self_symmetry_group is None and not self_symmetry_group.finalized:
                result.append(self_symmetry_group)
        
        #if size > 0:
        #    print "MERGING"
        #    result = merge(comm.allgather(result)) 
        self.result_group = result_group
        print_("Total number of grouped wos at main thread after ssg-execution: %i" % self.result_group.get_total_number_of_grouped_wos(result_group_count), self.logfile)
        return result, water_orientations.base
        
    
            
            
                
    
    cdef np.ndarray[np.int8_t, ndim=2] remove_symmetric_results(self, np.int8_t[:, ::1] water_orientations, np.uint16_t molecule_no, list symmetry_operations, int result_group_level, list self_symmetry_groups, bint save_self_symmetry_groups, list pending_self_symmetry_operations, SelfSymmetryGroup parent_self_symmetry_group, int depth_limit, bint scatter):
        
        #print "Performing symmetry check for %i geometries and %i symmetry operations" % (length_before, len(symmetry_operations))
        #if len(symmetry_operations) == 0:
        #    return water_orientations.base
        cdef int length_before = water_orientations.shape[0]
        cdef list new_self_symmetry_groups = []
        # Split the water orientations for the processors
        cdef list groups
        cdef int depth
        cdef int O = water_orientations.shape[0]
        groups, depth = self.do_initial_grouping(<np.ndarray[np.int8_t, ndim=2]>water_orientations.base, depth_limit, scatter)
        cdef ResultGroup group
        cdef SelfSymmetryGroup self_symmetry_group
        cdef int i, l, loaded, total_loaded = 0
        cdef int N = len(groups)
        cdef int M, count = 0 
        cdef int P, j
        cdef np.ndarray[np.int8_t, ndim=2] result
        cdef np.int8_t[:, ::1] new_water_orientations = np.ndarray((O, water_orientations.shape[1]), dtype=np.int8)
        #cdef list merged_groups = []
        # Finally do the symmetry checking
        #print "Processor number %i handles %i result groups" % (rank, len(groups))
        for l from 0 <= l < N:
            group = <ResultGroup>groups[l]
            if group.get_wos() is not None:
                self.symmetries_loaded = 0
                new_water_orientations = self.remove_symmetries_for_single_group(group, new_water_orientations, result_group_level, molecule_no, symmetry_operations, new_self_symmetry_groups, save_self_symmetry_groups, pending_self_symmetry_operations, parent_self_symmetry_group, &count)
                total_loaded += self.symmetries_loaded
                #merged_groups.append(group)
                
       
        

        if scatter and size > 1:
            new_self_symmetry_groups = merge(comm.allgather(new_self_symmetry_groups))
            result =  merge_water_orientations(comm.allgather(new_water_orientations.base[:count]))
            #print "gathered %i, %i" % (new_water_orientations.shape[0], rank)
        else:
            result = new_water_orientations.base[:count]

        if self_symmetry_groups is not None:
            P = len(new_self_symmetry_groups)
            for j from 0 <= j < P:
                self_symmetry_group = <SelfSymmetryGroup> new_self_symmetry_groups[j]
                self_symmetry_group.parent_self_symmetry_group = parent_self_symmetry_group
            self_symmetry_groups.extend(new_self_symmetry_groups)
        
            
        #print_("%i self symmetry groups, rank: %i" % (len(self_symmetry_groups), rank), self.logfile)
        
        #print "  -Loaded symmetries during process for %i wos" % total_loaded
        #print "  -Removed  %i geometries due to symmetry" % (length_before - new_water_orientations.shape[0])
        #print "  -Finally %i geometries" % result.shape[0]
        
        return result

    cdef inline np.int8_t[:, ::1] remove_symmetries_for_single_group(self, ResultGroup group, np.int8_t[:, ::1] new_water_orientations, int result_group_level, int molecule_no, list symmetry_operations, list self_symmetry_groups, bint save_self_symmetry_groups, list pending_self_symmetry_operations, SelfSymmetryGroup parent_self_symmetry_group, int *count):
        cdef np.int8_t[::1] wo
        cdef np.int8_t[:, ::1] wos = group.get_wos()
        cdef int i, M 
        if wos is not None:
            M = wos.shape[0]
        else:
            M = 0
        #print_("Processor %i: Current group has %i wos and invariant value is %i" % (rank, len(group.get_wos()), group.value), self.logfile)
        #print_(group.wos, self.logfile)
        cdef np.int8_t[:, ::1] bond_variables = np.ndarray((self.N, 4), dtype=np.int8)
        for i from 0 <= i < M:
            wo = wos[i]
            if not self.is_symmetric_with_another_result(group,  wo, molecule_no, False, symmetry_operations, result_group_level, self_symmetry_groups, save_self_symmetry_groups, pending_self_symmetry_operations, parent_self_symmetry_group, bond_variables):
                new_water_orientations[count[0]] = wo
                count[0] += 1
        #print_("Processor %i finished  group, resulted %i symmetry distinct results, loaded symmetries for %i wos" % (rank, count[0], self.symmetries_loaded), self.logfile)
        return new_water_orientations

    cdef np.ndarray[np.int8_t, ndim=2] remove_symmetries_no_invariants(self, np.int8_t[:, ::1] water_orientations,  np.int_t[:, :, ::1] nearest_neighbors_nos, np.uint16_t molecule_no, list symmetry_operations, list self_symmetry_groups, bool save_self_symmetry_groups):
        """ 
            Removes symmetries wihout using invariants
            NOTE: Very slow and should only be used for debugging purposes
        """
        cdef int length_before
        length_before = len(water_orientations)

        cdef np.int8_t[:, ::1] results = np.zeros((water_orientations.shape[0], water_orientations.shape[1]), dtype=np.int8)
        #cdef np.ndarray[np.int8_t, ndim=2] symmetries
       
        cdef np.int8_t[::1] symmetry, wo, wo_2
        cdef int i, j, k, l, m, N = water_orientations.shape[0], M, L = len(symmetry_operations), J, I, result_count = 0
        cdef np.int8_t symmetry_found, self_symmetry_found, copy_found
        cdef SymmetryOperation symmetry_operation
        cdef list self_symmetry_operations 
        cdef SelfSymmetryGroup self_symmetry_group
        cdef list new_self_symmetry_groups = []
        
        print_parallel("Performing symmetry check for %i geometries and %i symmetry operations" % (length_before, L), self.logfile)
        
        for i from 0 <= i < N: 
            self_symmetry_operations = []
            wo = water_orientations[i]
            
            #symmetries = self.get_symmetries(wo, nearest_neighbors_nos, symmetry_operations)
            symmetry_found = 0
            copy_found = 0
            M = results.shape[0]
            for k from 0 <= k < L:
                symmetry_operation = symmetry_operations[k] 
                symmetry = symmetry_operation.apply(wo)
                if symmetry is not None:
                    for j from 0 <= j < M: 
                        wo_2 = results[j]
                        copy_found = wos_are_equal(wo_2, wo, water_orientations.shape[1])
                        symmetry_found = wos_are_equal(wo_2, symmetry, water_orientations.shape[1])
                        if symmetry_found:
                            print_("SYMMETRY FOUND ", self.logfile)
                            print_(symmetry_operation, self.logfile)
                            
                            print_(symmetry_operation.additional_requirements, self.logfile)
                            
                            print_(symmetry_operation, self.logfile)
                            
                            print_(water_orientations.base[i], self.logfile)
                            print_(results.base[j], self.logfile)
                            print_("---", self.logfile)
                            """if symmetry_operation.additional_requirements is not None:
                                print wo
                                print wo_2
                                print symmetry_operation.molecule_change_matrix
                                print np.dot(symmetry_operation.translation_vector, self.atoms.get_cell())
                                print symmetry_operation
                                print symmetry_operation.additional_requirements
                                raw_input()"""
                            #self.check_symmetricity(wo, wo_2, symmetry_operation)
                            break
                        if  copy_found:
                            print_("Copy found", self.logfile)
                            print_(wo, self.logfile)
                            print_("---", self.logfile)
                            break
                    if symmetry_found or copy_found:
                        break
                    J = len(new_self_symmetry_groups)
                    for l from 0 <= l < J:
                        self_symmetry_group = new_self_symmetry_groups[l]
                        I = self_symmetry_group.water_orientations.shape[0]
                        for m from 0 <= m < I:
                            wo_2 = self_symmetry_group.water_orientations[m]
                        
                            symmetry_found = np.all(np.equal(wo_2, symmetry))
                            if symmetry_found:
                                print_("SYMMETRY FOUND ", self.logfile)
                                print_(symmetry_operation, self.logfile)
                                print_(symmetry_operation.nn_change_matrix, self.logfile) 
                                print_(symmetry_operation, self.logfile)
                                print_(wo, self.logfile)
                                print_(wo_2, self.logfile)
                                print_("---", self.logfile)
                                break
                        if symmetry_found:
                            break
                    if symmetry_found or copy_found:
                        break
                    self_symmetry_found = wos_are_equal(wo, symmetry, water_orientations.shape[1])
                    if save_self_symmetry_groups and self_symmetry_found:
                        print_("SELF SYMMETRY FOUND", self.logfile)
                        print_(symmetry_operation.molecule_change_matrix, self.logfile) 
                        print_(symmetry_operation.additional_requirements, self.logfile)
                        #sos = "%s %i %i %s %s" % (symmetry_operation.type, symmetry_operation.magnitude, symmetry_operation.center, symmetry_operation.translation_vector)
                        #print_(sos, self.logfile)
                        print_water_orientations(wo, self.logfile)
                        print_("---", self.logfile)
                        self_symmetry_operations.append(symmetry_operation)
            if not symmetry_found and len(self_symmetry_operations) > 0:
                new_self_symmetry_groups.append(SelfSymmetryGroup(self_symmetry_operations, wo, None))
            elif not  symmetry_found:
                results[result_count] = wo
                result_count += 1
        self_symmetry_groups.extend(new_self_symmetry_groups)
        print_parallel("  -Removed  %i geometries due to symmetry" % (length_before-result_count), self.logfile)
        print_parallel("  -Finally %i geometries" % result_count, self.logfile)
        return results.base[:result_count]
    

    cdef bint is_symmetric_with_another_result(self, ResultGroup result_group, np.int8_t[::1] water_orientations, signed int current_no, bint find_single, list symmetry_operations, int result_group_level, list self_symmetry_groups, bint save_self_symmetry_groups, list pending_self_symmetry_operations, SelfSymmetryGroup parent_self_symmetry_group, np.int8_t[:, ::1] bond_variables):
        """
            Checks if water orientation list is symmetric with some other water orientation list.
            The process is initialized by getting the bond variables which are used in getting the
            corrent result group, which contains the possible combinations for symmetric orientation
            list. After this the water orientations symmetricity is compared agaist the members of this
            group and added to the group if no symmetric result is found. 
                The method also finds the self symmetric results and initializes the SelfSymmetryGroup
            objects required in the handling of children.
        
            Parameters:
                result_group                    : the initial result group which contains the currentry handled waterorientation list grouped to a certain level
                water_orientations              : the currently handled water orientation list
                current_no                      : the number of the molecule currently handled
                find_single                     : (DEPRECATED) set to False 
                symmetry_operations             : the list of SymmetryOperation objects checked
                result_group_level              : the level the result groups are checked to
                self_symmetry_groups            : the list containing all SelfSymmetryGroup objects and in which to new instances are added
                save_self_symmetry_groups       : if the self symmetry groups are saved in the process
                pending_self_symmetry_operations: the symmetry operations that are pending for all the newly found self symmetry groups
                parent_self_symmetry_group      : the parent self symmetry group for all newly found self symmetry groups
                bond_variables                  : just an empty bond_variables table (goes with because of execution speed)

            Returns
                0 if symmetries are not found
                1 if a symmetry is found, or if self symmetry is found and save_self_symmetry_groups is True
                 
                 
        """
        cdef unsigned int i, N, M = len(symmetry_operations), O = water_orientations.shape[0]
        cdef np.uint8_t are_symmetric, is_self_symmetric
        cdef np.ndarray[np.int_t, ndim=3] nearest_neighbors_nos = <np.ndarray[np.int_t, ndim=3]> self.nearest_neighbors_nos 
        #if current_no != -1 and result_group.bvs is not None and len(result_group.bvs) > 0:
        #    bond_variables = result_group.bvs[current_no]
        #else:
        s = time()
        get_bond_variables_3(water_orientations,  nearest_neighbors_nos, O, bond_variables)
        self.conversion_time += time() - s
        # Find the result group
        s = time()
        cdef ResultGroup res_group = result_group._try_subgroups(bond_variables, result_group_level)
        self.result_group_tryout_time += time()-s
        cdef list self_symmetries = []
        cdef np.int8_t[::1] symmetry
        cdef np.int8_t[::1] wator
        
        #print "Result group has %i results" % len(res_group.wos)
        # Load symmetries only if there are other results to be checked with
        #if res_group.get_wos() != None and len(res_group.get_wos()) >0:
        #    #print "Result group has %i water orientations" % len(res_group.get_wos())
        #    symmetries = self.get_symmetries(water_orientations, nearest_neighbors_nos, symmetry_operations)
        #    
        #    self.symmetries_loaded += 1
        #    #print "Symmetry Load %i" % *symmetries_loaded
        #    M = symmetries.shape[0]
        #s = time()
        
        cdef SymmetryOperation symmetry_operation
        # go through all water orientations with same invariant values
        #  in other words all water orientations that belong to same group
        #if res_group.wos is not None and len(res_group.get_wos()) >0:
        #self.symmetries_loaded += 1
        cdef np.int8_t[:, ::1] wos =  res_group.get_wos()
        cdef WaterAlgorithm wa = self
        if wos is None:
            N = 0
        else:
            N = wos.shape[0]
        cdef bint inverse = False
        
        
        
        if  save_self_symmetry_groups or N != 0:
            for n from 0 <= n < M: 
                s = time()
                symmetry_operation = <SymmetryOperation>symmetry_operations[n]
                if symmetry_operation.type == <bytes> 'E':
                    continue
                symmetry = symmetry_operation.apply(water_orientations) 
                self.symmetry_load_time += time()-s 
                if symmetry is not None:
                    s = time()
                    for i from 0 <= i < N: 
                        wator = wos[i]
                        are_symmetric = wos_are_equal(wator, symmetry, O) # np.all(np.equal(wator, symmetry)) #
                        if are_symmetric:
                            #print_("SYMMETRY FOUND ", self.logfile)
                            #print_(symmetry_operation, self.logfile)
                            #print_water_orientations(water_orientations, self.logfile)
                            #self.verify_invariants(wator)
                            #print_("%s" % symmetry_operation.molecule_change_matrix, self.logfile)
                            #print_(symmetry_operation.nn_change_matrix, self.logfile) 
                            #self.view_result(water_orientations)    
                            #self.view_result(wator)
                            #if np.all(np.equal(water_orientations, symmetry)):
                            #    print "They are equal, should not be"
                            #    raw_input()
                            #print_(water_orientations.base, self.logfile)
                            #print_(wator, self.logfile)
                            #print_(symmetry, self.logfile)
                            #print wator
                            #print self.nearest_neighbors_nos
                            #raw_input()
                            #symmetry_operation.found_earlier = True
                            #self.symmetry_check_time += time()-s
                            #if find_single:
                            #    return wator
                            return True
                    if save_self_symmetry_groups:
                        is_self_symmetric =  wos_are_equal(water_orientations, symmetry, O) # np.all(np.equal(water_orientations, symmetry)) #
                        if is_self_symmetric:
                            #print_("SELF SYMMETRY FOUND", self.logfile)
                            #sos = "%s %i %i %s %s" % (symmetry_operation.type, symmetry_operation.magnitude, symmetry_operation.center, symmetry_operation.translation_vector)
                            #print_(sos, self.logfile)
                            #if wos is not None:
                            #    print_(wos.base)
                            #print_(symmetry_operation.molecule_change_matrix, self.logfile) 
                            #print_(symmetry_operation.additional_requirements, self.logfile)
                            #print_(symmetry_operation.molecule_change_matrix, self.logfile) 
                            #print_water_orientations(water_orientations, self.logfile)
                            self_symmetries.append(symmetry_operation)
                    self.symmetry_check_time += time()-s
        res_group.add_result(water_orientations, None) 
        if save_self_symmetry_groups and  len(self_symmetries) > 0 and self_symmetry_groups is not None:
            if pending_self_symmetry_operations is not None:
                self_symmetries.extend(pending_self_symmetry_operations)
            self_symmetry_groups.append(SelfSymmetryGroup(self_symmetries, water_orientations, None)) 
            return True
        #if find_single:
        #    return None
        return False

    

    def finalize_grouping(self, group, current_no = -1):
        """ Finalizes the initial grouping for given group made with do_initial_grouping """
        cdef np.int8_t[:, ::1] bond_variables = np.ndarray((self.N, 4), dtype=np.int8), wos = group.get_wos()
        cdef int i, N = wos.shape[0]
        for i in range(N):
            if current_no != -1 and group.bvs != None and len(group.bvs) > 0:
                bond_variables = group.bvs[current_no]
            else:
                s = time()
                get_bond_variables_3(wos[i],  self.nearest_neighbors_nos, self.N, bond_variables)
                self.conversion_time += time() - s

            res_group = group.try_subgroups(bond_variables, -1)
            res_group.add_result(wos[i], None)
            print_("Result group length %i" % len(res_group.wos), self.logfile)

    

    def execute_commands(self, options, args = None):
        if options.wa_method != None:
            self.run_command(options.wa_method, options.do_symmetry_check, options.invariant_count, options.dissosiation_count, args = args)

    def run_command(self, command, do_symmetry_check = True, invariant_count = 20, dissosiation_count = 0 , no_self_symmetry_groups = False, args = None):
        self.logfile = open(self.get_folder()+str(command)+'_log%i.txt' % rank, 'w+')
            
        if no_self_symmetry_groups:
            self.save_self_symmetry_groups = False
        else:
            self.save_self_symmetry_groups = True

        self.initialize_symmetry_check(do_symmetry_check)
        self.invariant_count = invariant_count
        self.dissosiation_count = dissosiation_count
        method = getattr(self, str(command))
        if args is not None:
            method(*args)
        else:
            method()
            

    # Reverse engineering method
    def find_equivalent(self, filename):
        cdef np.int8_t[:, ::1] bond_variables
        oxygens_0 = self.atoms.copy()
        atoms = ase.io.read(filename)
        oxygens = remove_hydrogens(atoms)
        
        ivalues, ivectors = get_moments_of_inertia(self.atoms)
        i2values, i2vectors  = get_moments_of_inertia(oxygens)
        new_atoms = atoms.copy()
        new_oxygens = oxygens.copy()
        maxvalue = max(ivalues)
        selected_axis = -1
        evectors = np.zeros_like(i2vectors)
        print ivectors
        print ivalues
        print i2vectors
        print i2values
        equivalence = [-1, -1, -1]
        for i, value in enumerate(ivalues):
            if value == maxvalue:
                selected_axis = i
            for i2, value2 in enumerate(i2values):
                print "%i and %i are equal?  %s" % (i, i2, are_equal(value, value2, error_tolerance=100))
                if are_equal(value, value2, error_tolerance=350) and i2 not in equivalence:
                    equivalence[i] = i2
                    evectors[i] = i2vectors[i2]
                    break
        print equivalence
        # find coordinates  in inertia basis for both
        basis_1 = np.reshape(ivectors, ivectors.size, order='F').reshape((3, 3))
        basis_2 = np.reshape(evectors, evectors.size, order='F').reshape((3, 3))
        
        com = self.atoms.get_center_of_mass()
        com2 = oxygens.get_center_of_mass()
        c1 = get_coordinates_in_basis(self.atoms.get_positions()-com, ivectors)
        c2 = get_coordinates_in_basis(oxygens.get_positions()-com2, i2vectors)
        c2 = get_coordinates_in_basis(c2, ivectors)
        c3 = np.zeros_like(c2)
        
        for i, e in enumerate(equivalence):
            c3[:,i] = c2[:,e]
        self.atoms.set_positions(c1)
        new_oxygens.set_positions(c3)
        moi1 = get_moments_of_inertia(self.atoms)[1]
        moi2 = get_moments_of_inertia(new_oxygens)[1]
        c1[:, 2] = -c1[:, 2]
        c1[:, 1] = c1[:, 1]
        #c1[:, 0] = -c1[:, 0]
        #closest_position = get_closest_position(c1[0], c2)
        
        self.atoms.set_positions(c1)
        new_oxygens.set_positions(c3)
        #self.atoms = rotate_to(self.atoms, c1[0], closest_position)
        ase.visualize.view(new_oxygens) 
        ase.visualize.view(self.atoms)
        
        equals = None # get_equals(self.atoms, get_oxygens(new_oxygens).get_positions(), error_tolerance=1.2)
        cp = get_closest_positions(c1, c3)
        cdef int loaded = 0
        if equals != None:
            nearest_neighbors_nos, bvv, elongated_bonds, elongated_hydrogen_bonds = self.get_bond_varibles_from_atoms(atoms)
            water_orientations = self.get_water_orientations_from_bond_variables_and_molecule_change_matrix(bvv, equals)
            self.view_result(water_orientations)
            wos = load_results(self.folder)
            self.initialize_symmetry_operations()
            self.load_invariants()
            self.result_group = ResultGroup(list(self.graph_invariants),  0)
            self.print_time_stats()
            self.do_initial_grouping(wos, 20, True)
            #bond_variables = get_bond_variables_3(water_orientations,  self.nearest_neighbors_nos)
            rg = self.result_group.try_subgroups(bond_variables.base,  1)
            print "Result groups has %i results" % len(rg.wos)
            self.finalize_grouping(rg)
            bond_variables = np.ndarray((self.N, 4), dtype=np.int8)
            symmetry = self.is_symmetric_with_another_result(rg,  water_orientations, -1, True, self.symmetry_operations, -1, [], False, [], None, bond_variables)

            assert symmetry != None
            for i, wo in enumerate(wos):
                if all(np.equal(wo, symmetry)):
                    print "Structure is symmetric with structure number %i" % i 

    def verify_results(self):
        wos = load_results(self.folder)
        initial_count = wos.shape[0]
        self.initialize_symmetry_operations()
        cdef SymmetryOperation symm = self.symmetry_operations[2]
        self.print_time_stats()
        if self.invariant_count > 0:
            self.load_invariants()
            self.result_group = ResultGroup(list(self.graph_invariants),  0)
            wos = self.remove_symmetric_results(wos, self.N-1, self.symmetry_operations, len(self.graph_invariants), [], False, [], None, 20, True) 
        else:
            wos = self.remove_symmetries_no_invariants(wos,   self.nearest_neighbors_nos, self.N-1, self.symmetry_operations, [], False)
      
        
        print "Originally %i results, removed %i results, finally %i results" % (initial_count, initial_count - len(wos), len(wos))
        #profiles = self.get_result_profiles(wos)   
        #profiles = self.remove_equal_profiles(profiles)
        #profiles = self.remove_symmetric_profiles(profiles)
        #for i, profile in enumerate(profiles):
        #    print i, ": ", profile
        #self.remove_symmetric_results(wos, -1, self.symmetry_operations)
        return wos

    def verify_results_and_overwrite(self):
        wos = self.verify_results()
        self.save_results(self.order[len(self.order)-1], wos, self.order)
    
    
    def print_nearest_neighbors_nos(self, bond_variables = None, symmetry_operation = None):
        result = ""
        for i in range(self.N):
            result += "(%i): " % i
            if i < 10:
                result += " "
            for j in range(len(self.nearest_neighbors_nos[0][i])):
                result += "%i" % self.nearest_neighbors_nos[0][i][j]
                if self.nearest_neighbors_nos[2][i][j] != 13:
                    result += "_%i" % self.nearest_neighbors_nos[2][i][j]
                    if bond_variables != None:
                        result +=": "
                    if self.nearest_neighbors_nos[2][i][j] < 10:
                        result += " "
                else:
                    if bond_variables != None or symmetry_operation is not None:
                        result +=":"
                    result += "    "
                if self.nearest_neighbors_nos[0][i][j] < 10:
                    result += " "
                if bond_variables != None:
                    if bond_variables[i][j] < 0: 
                        result += "%i  " % bond_variables[i][j]
                    else:
                        result += " %i  " % bond_variables[i][j] 
                if symmetry_operation is not None:
                    new_no = symmetry_operation.molecule_change_matrix[i]
                    k = symmetry_operation.nn_change_matrix[i][j]
                    result += " %i-%i" % (new_no, self.nearest_neighbors_nos[0, new_no, k])
                        
                    if self.nearest_neighbors_nos[2, new_no, k] != 13:
                        result += "_%i" % self.nearest_neighbors_nos[2, new_no, k]
                        if self.nearest_neighbors_nos[2, new_no, k] < 10:
                            result += " "
                    else:
                        result += "   "
                    if new_no < 10:
                        result += " "
                    if self.nearest_neighbors_nos[0, new_no, k] < 10:
                        result += " "
                    result += "  " 
            result += "\n"
        print result

    def print_nearest_neighbors_nos_wo(self, water_orientation):
        cdef np.int8_t[:, ::1] bond_variables = np.ndarray((self.N, 4), dtype=np.int8)
        get_bond_variables_3(water_orientation, self.nearest_neighbors_nos, self.N, bond_variables)
        self.print_nearest_neighbors_nos(bond_variables)
        
    
    def verify_invariants(self, water_orientation):
        if self.graph_invariants is None:
            self.initialize_symmetry_operations()
            self.load_invariants()
        
        cdef  np.int8_t[:, ::1] bond_variables_1 = np.empty((water_orientation.shape[0], 4), dtype=np.int8), bond_variables_2 = np.empty((water_orientation.shape[0], 4), dtype=np.int8)
        get_bond_variables_3(water_orientation, self.nearest_neighbors_nos, water_orientation.shape[0], bond_variables_1)
        cdef np.int8_t[::1] water_orientation_2
        #self.print_nearest_neighbors_nos(bond_variables_1)
        #self.print_nearest_neighbors_nos(bond_variables_2)
        #self.print_nearest_neighbors_nos(symmetry_operation = symmetry_operation)
        cdef Invariant invariant
        cdef bint valid
        cdef list new_invariants = []
        #cdef InvariantTerm invariant_term
        cdef SymmetryOperation symmetry_operation
        cdef np.int_t[:, :, ::1] nearest_neighbors_nos = self.nearest_neighbors_nos
        for i, invariant in enumerate(self.graph_invariants):
            valid = True
            for j, symmetry_operation in enumerate(self.symmetry_operations):
                water_orientation_2 = symmetry_operation.apply(water_orientation)
                if water_orientation_2 is not None:
                    get_bond_variables_3(water_orientation_2, nearest_neighbors_nos, water_orientation.shape[0], bond_variables_2)
                    value_1 = invariant._get_value(bond_variables_1)
                    value_2 = invariant._get_value(bond_variables_2)
                    if not ice_rules_apply(bond_variables_2, nearest_neighbors_nos):
                        continue
                    if  value_1 != value_2:
                    
                        print "------------------------"
                        print "Invariant %i values differ %i, %i" % (i, value_1, value_2)
                        print symmetry_operation.molecule_change_matrix, symmetry_operation.translation_vector, symmetry_operation.type, symmetry_operation.magnitude, symmetry_operation.order_no, symmetry_operation.center, symmetry_operation.vector
                        invariant.debug_value(bond_variables_1)
                        self.print_nearest_neighbors_nos(bond_variables_1)
                        invariant.debug_value(bond_variables_2)
                        self.print_nearest_neighbors_nos(bond_variables_2)
            
                        valid = False
                        if invariant.original_indeces[0, 0] == 0 and invariant.original_indeces[0, 1] == 2:
                            raw_input()
                        break
            if valid:
                new_invariants.append(invariant)
        self.graph_invariants = new_invariants
                        

                        
    def generate_graph_invariant_value(self, water_orientation, i, j, k, l, axis1 = 13, axis2 = 13):
        cdef SymmetryOperation symmetry_operation
        cdef  np.int8_t[:, ::1] bond_variables_2 = np.ndarray((self.N, 4), dtype=np.int8)
        #cdef  np.int8_t[:, ::1] bv = np.ndarray((self.N, 4), dtype=np.int8)
        cdef np.ndarray[np.int8_t, ndim=1] water_orientation_2
        for m, nn in enumerate(self.nearest_neighbors_nos[0][i]):
            if nn == j and self.nearest_neighbors_nos[2][i][m] == axis1:
                index_j = m
        for n, nn in enumerate(self.nearest_neighbors_nos[0][k]):
            if nn == l and self.nearest_neighbors_nos[2][k][n] == axis2:
                index_l = n 
        value = 0        
        cdef str invariant_str = ""
        cdef str value_str = ""
        inverse = False
        cdef np.ndarray[np.int8_t, ndim=3] indeces = np.zeros((len(self.symmetry_operations), 2, 3), dtype=np.int8)
        #print "b_{%i, %i} * b_{%i, %i}" % (i, index_j, k, index_l)
        for m, symmetry_operation in enumerate(self.symmetry_operations):
            #print symmetry_operation.molecule_change_matrix
            #print symmetry_operation.nn_change_matrix
            
            water_orientation_2 = symmetry_operation.apply(water_orientation)
            get_bond_variables_3(water_orientation_2, self.nearest_neighbors_nos, self.N, bond_variables_2)

            current_value = bond_variables_2[i][index_j] * bond_variables_2[k][index_l]
            value += current_value
            if inverse:
                indeces[m, 0] = np.array([symmetry_operation.inverse_molecule_change_matrix[i], self.nearest_neighbors_nos[0, symmetry_operation.inverse_molecule_change_matrix[i], symmetry_operation.inverse_nn_change_matrix[i][index_j]], self.nearest_neighbors_nos[2, symmetry_operation.inverse_molecule_change_matrix[i], symmetry_operation.inverse_nn_change_matrix[i][index_j]]], dtype=np.int8)
                indeces[m, 1] = np.array([symmetry_operation.inverse_molecule_change_matrix[k], self.nearest_neighbors_nos[0, symmetry_operation.inverse_molecule_change_matrix[k], symmetry_operation.inverse_nn_change_matrix[k][index_l]], self.nearest_neighbors_nos[2, symmetry_operation.inverse_molecule_change_matrix[k], symmetry_operation.inverse_nn_change_matrix[k][index_l]]], dtype=np.int8)
                
                #value_str += " (%i * %i = %i) " % (bond_variables_2[i][index_j], bond_variables_2[k][index_l], current_value)
                #invariant_str += "(%i), " % current_value
            else:
                indeces[m, 0] = np.array([symmetry_operation.molecule_change_matrix[i], self.nearest_neighbors_nos[0, symmetry_operation.molecule_change_matrix[i], symmetry_operation.nn_change_matrix[i][index_j]], self.nearest_neighbors_nos[2, symmetry_operation.molecule_change_matrix[i], symmetry_operation.nn_change_matrix[i][index_j]]], dtype=np.int8)
                indeces[m, 1] = np.array([symmetry_operation.molecule_change_matrix[k], self.nearest_neighbors_nos[0, symmetry_operation.molecule_change_matrix[k], symmetry_operation.nn_change_matrix[k][index_l]], self.nearest_neighbors_nos[2, symmetry_operation.molecule_change_matrix[k], symmetry_operation.nn_change_matrix[k][index_l]]], dtype=np.int8)
        order_list = []
        order_of_invariant = 2
        N = len(indeces)
        multipliers = []
        constants = []
        for m from 0 <= m < N:
            constant = False
            multiplier = 1
            for n, index in enumerate(indeces[m]):
                if index[0] > index[1]:
                    new_index_2 = get_opposite_periodicity_axis_number(index[2])
                    indeces[m, n] = np.array([index[1], index[0], new_index_2], dtype=np.int8)
                    multiplier *= -1
            multipliers.append(multiplier)
            if indeces[m, 0, 0] > indeces[m, 1, 0]:
                indeces[m] = np.array([indeces[m, 1], indeces[m, 0]], dtype=np.int8)
            elif indeces[m, 0, 0] == indeces[m, 1, 0]:
                if indeces[m, 0, 1] > indeces[m, 1, 1]:
                    indeces[m] = np.array([indeces[m, 1], indeces[m, 0]], dtype=np.int8)
                elif indeces[m, 0, 1] == indeces[m, 1, 1]:
                    if indeces[m, 0, 2] > indeces[m, 1, 2]:
                        indeces[m] = np.array([indeces[m, 1], indeces[m, 0]], dtype=np.int8)
                    elif indeces[m, 0, 2] == indeces[m, 1, 2]:
                        constant = True
            constants.append(constant)
            
            order_list.append(indeces[m].flatten())
        np_order_list = np.array(order_list, dtype=np.uint8)
        l = order_of_invariant -1
        counter = 0
        lists = []
        while l >= 0:
            lists.append(np_order_list[:, l*3+2])
            lists.append(np_order_list[:,  l*3+1])
            lists.append(np_order_list[:,  l*3+0])
             
            l -= 1
            counter += 1
        ind = np.lexsort(lists)   
        previous_term = None
        multiplier = 0
        c = 0
        
        bv = get_bond_variables_2(water_orientation, self.nearest_neighbors_nos)
        old_value = 0
        old_value_str = ""
        for i in ind:
            mul = multipliers[i]
            current_term = indeces[i]
            constant = constants[i]
            if not constant:
                if previous_term is None:
                    previous_term = current_term
                    multiplier = mul
                else:
                    if np.all(current_term == previous_term):
                        multiplier += mul
                    else:
                        invariant_str += " + %i * b_{%i, %i_[%i]} * b_{%i, %i_[%i]}" % (multiplier, previous_term[0][0], previous_term[0][1], previous_term[0][2], previous_term[1][0], previous_term[1][1], previous_term[1][2])
                        old_value += multiplier * bv[previous_term[0][0]][previous_term[0][1]][previous_term[0][2]] * bv[previous_term[1][0]][previous_term[1][1]][previous_term[1][2]]
                        if multiplier != 0:                        
                            old_value_str += " + %i * %i * %i" % (multiplier, bv[previous_term[0][0]][previous_term[0][1]][previous_term[0][2]], bv[previous_term[1][0]][previous_term[1][1]][previous_term[1][2]])
                        multiplier = mul
 
                        previous_term = current_term
            else:
                #print  "+ %i * b_{%i, %i_[%i]} * b_{%i, %i_[%i]} is constant" % (mul, current_term[0][0], current_term[0][1], current_term[0][2], current_term[1][0], current_term[1][1], current_term[1][2])
                c += mul
                previous_term  = None
        if previous_term is not None:
            invariant_str += " + %i * b_{%i, %i_[%i]} * b_{%i, %i_[%i]}" % (multiplier, previous_term[0][0], previous_term[0][1], previous_term[0][2], previous_term[1][0], previous_term[1][1], previous_term[1][2])
            old_value += multiplier * bv[previous_term[0][0]][previous_term[0][1]][previous_term[0][2]] * bv[previous_term[1][0]][previous_term[1][1]][previous_term[1][2]]
            if multiplier != 0:
                old_value_str += " + %i * %i * %i" % (multiplier, bv[previous_term[0][0]][previous_term[0][1]][previous_term[0][2]], bv[previous_term[1][0]][previous_term[1][1]][previous_term[1][2]])
        if c != 0:
            invariant_str += " + %i" % c
            old_value_str  += " + %i" % c
            old_value += c
             
        print invariant_str
        #print value_str
        print "Value generated from old bond variables %i " % old_value
        print old_value_str
        return value
                
        
    def check_symmetricity(self, water_orientation_1, water_orientation_2, symmetry_operation):
        if self.graph_invariants is None:
            self.initialize_symmetry_operations()
            self.load_invariants()
        cdef  np.int8_t[:, ::1] bond_variables_1 = np.ndarray((self.N, 4), dtype=np.int8), bond_variables_2 = np.ndarray((self.N, 4), dtype=np.int8)
        get_bond_variables_3(water_orientation_1, self.nearest_neighbors_nos, self.N, bond_variables_1)
        get_bond_variables_3(water_orientation_2, self.nearest_neighbors_nos, self.N, bond_variables_2)
        self.print_nearest_neighbors_nos(bond_variables_1)
        self.print_nearest_neighbors_nos(bond_variables_2)
        self.print_nearest_neighbors_nos(symmetry_operation = symmetry_operation)
        cdef Invariant invariant
        cdef InvariantTerm invariant_term
        for i, invariant in enumerate(self.graph_invariants):
            value_1 = invariant._get_value(bond_variables_1.base)
            value_2 = invariant._get_value(bond_variables_2.base)
            if value_1 != value_2:
                print "Invariant %i values differ %i, %i" % (i, value_1, value_2)
                print invariant
                print self.generate_graph_invariant_value(water_orientation_1, invariant.original_indeces[0, 0], invariant.original_indeces[0, 1], invariant.original_indeces[1, 0], invariant.original_indeces[1, 1], invariant.original_indeces[0, 2], invariant.original_indeces[1, 2])
                print self.generate_graph_invariant_value(water_orientation_2, invariant.original_indeces[0, 0], invariant.original_indeces[0, 1], invariant.original_indeces[1, 0], invariant.original_indeces[1, 1], invariant.original_indeces[0, 2], invariant.original_indeces[1, 2])
                raw_input()
                #for invariant_term in invariant.invariant_terms:
                #    value_1 = invariant_term.get_value(bond_variables_1)
                #    value_2 = invariant_term.get_value(bond_variables_2)
                #    if value_1 != value_2:
                #        print "Invariant term (%s) values differ %i, %i" % (invariant_term, value_1, value_2)
                        
        self.view_result(water_orientation_1)
        self.view_result(water_orientation_2)
        



cdef dict get_bond_variables_2(np.ndarray[np.int8_t, ndim=1] water_orientations, np.ndarray[np.int_t, ndim=3]  nearest_neighbors_nos):
    cdef dict result = {}
    cdef unsigned char i, j
    cdef signed char periodic_axis, water_orientation
    cdef np.ndarray[np.int_t, ndim=1] nn 
    cdef np.int8_t[::1] bvv
    cdef np.uint8_t neighbor_no
    for i,  nn in enumerate(nearest_neighbors_nos[0]):
        result[i] = {}
        water_orientation = water_orientations[i]
        bvv = get_bond_variable_values_from_water_orientation(water_orientation)
        for j,  neighbor_no in enumerate(nn):
            if neighbor_no not in result[i]:
                result[i][neighbor_no] = {}
            periodic_axis = nearest_neighbors_nos[2][i][j]
            assert periodic_axis not in result[i][neighbor_no]
            result[i][neighbor_no][periodic_axis] = bvv[j]
    return result

cdef dict get_index_conversion_table(np.ndarray[np.int_t, ndim=3]  nearest_neighbors_nos):
    cdef dict result = {}
    cdef unsigned char i, j
    cdef signed char water_orientation
    cdef np.ndarray[np.int_t, ndim=1] nn 
    cdef int neighbor_no, periodic_axis
    
    for i,  nn in enumerate(nearest_neighbors_nos[0]):
        result[i] = {}
        for j,  neighbor_no in enumerate(nn):
            if neighbor_no not in result[i]:
                result[i][neighbor_no] = {}
            periodic_axis = nearest_neighbors_nos[2, i, j]
            
            #assert periodic_axis not in result[i][neighbor_no]
            result[i][neighbor_no][periodic_axis] = np.array([i, j], dtype=np.uint8)
    return result

cdef inline void get_bond_variables_3(np.int8_t[::1] water_orientations, np.int_t[:, :, ::1]  nearest_neighbors_nos, int N, np.int8_t[:, ::1] bond_variables):  
    cdef np.int8_t water_orientation
    cdef int neighbor_no, i
    cdef np.int8_t[::1] bvv
    for i from 0 <= i < N:
        water_orientation = water_orientations[i]
        bvv = get_bond_variable_values_from_water_orientation(water_orientation)
        for j from 0 <= j < 4:
            bond_variables[i, j] = bvv[j]

cdef inline void get_profile_bond_variables(np.int8_t[::1] profile, np.int_t[:, :, ::1]  nearest_neighbors_nos, int N, np.int8_t[:, ::1] bond_variables):  
    cdef int neighbor_no, profile_value, i, j
    cdef np.int8_t[::1] bvv
    for i from 0 <= i < N:
        profile_value = profile[i]
        if profile_value != 0:
            for j from 0 <= j < 4:
                if nearest_neighbors_nos[0, i, j] == i and nearest_neighbors_nos[2, i, j] == 13:
                    bond_variables[i, j] = profile_value
                else:
                    bond_variables[i, j] = 0
        else:
            for j from 0 <= j < 4:
                bond_variables[i, j] = 0

cdef inline bint ice_rules_apply(np.int8_t[:, ::1] bond_variables, np.int_t[:, :, ::1] nearest_neighbors_nos):
    cdef int i, j, k, N = nearest_neighbors_nos.shape[1], neighbor_no, neighbor_of_neighbor, axis, axis_of_neighbor
    for i in range(N):
        for j in range(4):
            neighbor_no = nearest_neighbors_nos[0, i, j]
            axis = nearest_neighbors_nos[2, i, j]
            if neighbor_no != i:
                for k in range(4):
                    neighbor_of_neighbor = nearest_neighbors_nos[0, neighbor_no, k]
                    axis_of_neighbor = get_opposite_periodicity_axis_number(nearest_neighbors_nos[2, neighbor_no, k])
                    if neighbor_of_neighbor == i and axis_of_neighbor == axis:
                        if bond_variables[i, j] == bond_variables[neighbor_no, k]: 
                            return False
    return True



cdef  np.ndarray[np.int8_t, ndim=3] get_bond_variables(water_orientations, nearest_neighbors_nos):
    cdef np.ndarray[np.int8_t, ndim=3] bond_variables = np.zeros((2, len(nearest_neighbors_nos[0]),  len(nearest_neighbors_nos[0])),  dtype='int8')
    cdef np.ndarray[np.int8_t, ndim=1] bvv
    for i,  nn in enumerate(nearest_neighbors_nos[0]):
        water_orientation = water_orientations[i]
        bvv = get_bond_variable_values_from_water_orientation(water_orientation)
        for l,  neighbor_no in enumerate(nn):
            periodic = nearest_neighbors_nos[1][i][l]
            if neighbor_no > i:
                bond_variables[periodic][i][neighbor_no][0] = bvv[l]
                bond_variables[periodic][i][neighbor_no][1] = nearest_neighbors_nos[2][i][l]
                bond_variables[periodic][neighbor_no][i][0] = -bvv[l]
                bond_variables[periodic][neighbor_no][i][1] = get_opposite_periodicity_axis_number(nearest_neighbors_nos[2][i][l])
            elif neighbor_no == i:
                bond_variables[periodic][i][neighbor_no][0] = bvv[l]
                bond_variables[periodic][i][neighbor_no][1] = nearest_neighbors_nos[2][i][l]
    return bond_variables

cdef inline bint wos_are_equal(np.int8_t[::1] orientation1, np.int8_t[::1] orientation2, int N):
    """
        Checks if two water orientation lists are equal
        Parameter:
            orientation1 : first orientation list
            orientation2 : second orientation list
            N            : The number of molecules in raft
        Returns
            0 if are not equal
            1 if they are equal
    """
    cdef int i
    for i from 0 <= i < N:
        if orientation1[i] != orientation2[i]:
            return 0
    return 1

cdef inline bint profiles_are_equal(np.int8_t[::1] orientation1, np.int8_t[::1] orientation2, int N):
    """
        Checks if two water orientation lists are equal
        Parameter:
            orientation1 : first orientation list
            orientation2 : second orientation list
            N            : The number of molecules in raft
        Returns
            0 if are not equal
            1 if they are equal
    """
    cdef int i
    for i from 0 <= i < N:
        if orientation1[i] != orientation2[i]:
            return 0
    return 1

cdef inline bint water_orientation_is_valid_using_possible_combinations(signed char water_orientation, np.int8_t [::1] water_orient, np.uint16_t molecule_no, np.uint8_t [:, :, ::1] possible_combinations, np.int_t[::1] nn, bint init):
    """
       Checks if the ice rules are fulfilled for given water orientation for the current molecule (molecule_no)
       in given water orientation environment water_orient. The process is done using the given possible combinations.
            Parameters:
                water_orientation    : water orientation tried for current molecule
                water_orient         : water orientation list that describes the water orientation environment
                molecule_no          : the number of the molecule that is being handled
                possible_combinations: the possible water orientation combinations for current molecule (molecule_no) and its neighbors
                nn                   : nearest neighbors numbers for molecule handled (molecule_no)
    """
    cdef int neighbor_no, i,  N = nn.shape[0]
    cdef bint combination_possible
    cdef np.int8_t neighbor_orientation
    if init:
        for i from 0 <= i < N:
            for neighbor_orientation from 0 <= neighbor_orientation < 6:
                combination_possible = possible_combinations[i, water_orientation, neighbor_orientation]
                if combination_possible:
                    return True
        return False
    else:
        for i from 0 <= i < N:
            neighbor_no = nn[i]
            neighbor_orientation = water_orient[neighbor_no]
            if neighbor_orientation == -1:
                continue
            combination_possible = possible_combinations[i, water_orientation, neighbor_orientation]
            if combination_possible == 0:
                return False
        return True      

cdef inline np.int8_t[:, ::1] stack(np.int8_t[:, ::1] a, np.int8_t[:, ::1] b):
    cdef unsigned int a_length = a.shape[0], b_length = b.shape[0], N = a.shape[1], new_length
    cdef np.int8_t[:, ::1] result
    cdef tuple newsize
    if b_length != 0:
        new_length = a_length+b_length
        newsize = (new_length, N)
        result = np.ndarray(newsize, dtype=np.int8)
        result[:a_length] = a
        result[a_length:] = b
        return result 
    else:
        return a


cdef gather_and_sum(parameter):
    parameter = np.array([parameter], dtype=type(parameter))
    parameter = np.sum(merge(comm.allgather(parameter)))
    return parameter

cdef gather_and_max(parameter):
    parameter = np.array([parameter], dtype=type(parameter))
    parameter = max(merge(comm.allgather(parameter)))
    return parameter

cdef inline void print_water_orientations(np.int8_t[::1] water_orientations, logfile):
    result = ""
    cdef int i, N = water_orientations.shape[0]
    for i in range(N):
       result += "%i " % water_orientations[i]
    print_(result, logfile)
    

def print_parallel(text, file = None):
    if rank == 0:
        if file is not None:
            print >> file, text
        print text
                    


include "help_methods_cython.pyx"
