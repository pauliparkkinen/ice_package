import os, ase
import numpy as np
from optparse import OptionParser
from system_commons import SystemCommons, rank, size

allowed_programs = ['TurboMole', 'cp2k', 'GPAW']
allowed_methods = ['dft', 'mp2']

"""
try:
    from gpaw import *
    import gpaw
    size = gpaw.mpi.size
    rank = gpaw.mpi.rank
    comm = gpaw.mpi.world
    allowed_programs.append("GPAW")
except ImportError:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    try:
        size = comm.size
        rank = comm.rank
    except ImportError:
        size = 1
        rank = 0
"""

class EnergyCommons(SystemCommons):
    def __init__(self, **kwargs):
        SystemCommons.__init__(self, folder = kwargs['folder'], debug = kwargs['debug'])
        self.program = kwargs['program']
        self.original_program = kwargs['original_program']
        self.method = kwargs['method']
        self.check_program_input()
        self.check_method_input()
        self.number_of_relaxation = kwargs['number_of_relaxation']
        self.xc = kwargs['xc']
        self.basis_set = kwargs['basis_set']
        self.fmax = kwargs['fmax']
        self.extension = kwargs['file_type']
        

    def check_program_input(self):
        if self.program not in allowed_programs:
            raise Exception("Program '"+self.program+"' is not in allowed programs (%s)" % allowed_programs)
        if self.original_program is not None and self.original_program not in allowed_programs:
            raise Exception("Program '"+self.original_program+"' is not in allowed programs (%s)" % allowed_programs)
    
    def check_method_input(self):
        if self.method not in allowed_methods:
            raise Exception("Method  '"+self.method+"' is not in allowed method (%s)" % allowed_methods)

    def _get_file_paths(self, number, extension = "traj", number_of_relaxation = None, lattice_constant = None, basis_set = None, program = None, method = None, xc = None, additional_atoms_calculation = False):
        dir, filename = self.get_directory_and_filename(number, loading = True, extension = 'traj', lattice_constant = lattice_constant, number_of_relaxation = 0, basis_set = basis_set, program = program, method = method, additional_atoms_calculation = additional_atoms_calculation)
        result = []
        if os.path.exists(dir):
            filenames = os.listdir(dir)
            
            for filename in filenames:
                valid, parameters =  self._file_path_is_valid(filename, extension = extension, number_of_relaxation = number_of_relaxation, lattice_constant = lattice_constan, basis_set = basis_set, program = program, method = method, xc = xc)
                    
                if valid:
                    result.append(dir+"/"+filename)
        return result
        
    
    def _file_path_is_valid(self, filename, extension = "traj", number_of_relaxation = None, lattice_constant = None, basis_set = None, program = None, method = None, xc = None):
        """
            Check if the parameters contained by the file path are equal with 
            the input parameters
            
            This method also returns the file path parameters
        """
        if not filename.endswith(extension):
            return False, None
        if filename.endswith("_start."+extension):
            return False, None
        parameters = self.parse_parameters_from_filename(filename)
        if basis_set is not None and parameters['basis_set'] != basis_set:
            return False, parameters
        if program is not None and parameters['program'] != program:
            return False, parameters
        if method is not None and parameters['method'] != method:
            return False, parameters
        if xc is not None and parameters['xc'] != xc:
            return False, parameters
        if  number_of_relaxation is not None and parameters['number_of_relaxation'] != number_of_relaxation:
            return False, parameters
        if lattice_constant is not None and parameters['lattice_constant'] != lattice_constant:
            return False, parameters
        return True, parameters
        
        
    def get_latest_file_paths(self, number, extension = "traj", number_of_relaxation = None, lattice_constant = None, basis_set = None, program = None, method = None, xc = None, additional_atoms_calculation = False):
        """
            Removes files with otherwise same parameters, but with different
            numbers of relaxation. The latest is preserved, if such cases are found
        """
        
        dir, filename = self.get_directory_and_filename(number, loading = True, extension = 'traj', lattice_constant = lattice_constant, number_of_relaxation = 0, basis_set = basis_set, program = program, method = method, additional_atoms_calculation = additional_atoms_calculation)
        result = []
        if os.path.exists(dir):
            filenames = os.listdir(dir)
            for filename in filenames:
                valid, parameters =  self._file_path_is_valid(filename, extension = extension, number_of_relaxation = number_of_relaxation, lattice_constant = lattice_constant, basis_set = basis_set, program = program, method = method, xc = xc)
                    
                if valid:
                    filepath = self._get_latest_file_path(number, extension = extension, lattice_constant = parameters['lattice_constant'], basis_set = parameters['basis_set'], program = parameters['program'], method = parameters['method'], xc = parameters['xc'], additional_atoms_calculation = additional_atoms_calculation)
                    result.append(filepath)
        return result
   
    


    def _get_latest_file_path(self, number, extension = "traj", lattice_constant = None, basis_set = None, program = None, method = None, xc = None, additional_atoms_calculation = False, ice_structure_calculation = False):
        dir, filename = self.get_directory_and_filename(number, loading = True, extension = 'traj', lattice_constant = lattice_constant, number_of_relaxation = 0, basis_set = basis_set, program = program, method = method, additional_atoms_calculation = additional_atoms_calculation, ice_structure_calculation = ice_structure_calculation)
        latest_path = None
        # for the storage of the number of relaxation for the most recent calculation
        current_number_of_relaxation = -1
        if self.debug:
            print dir
        if os.path.exists(dir):
            filenames = os.listdir(dir)
            for filename in filenames:
                if self.debug:
                    print filename
                # check if the file fulfills the given conditions
                valid, parameters = self._file_path_is_valid(filename, extension = extension, lattice_constant = lattice_constant, basis_set = basis_set, program = program, method = method, xc = xc)
                # check if the handled file is more recent than the previous recent
                if valid and parameters['number_of_relaxation'] > current_number_of_relaxation:
                    latest_path = dir+"/"+filename
                    current_number_of_relaxation = parameters['number_of_relaxation']
        if self.debug:
            print latest_path
        return latest_path
            
    def get_latest_file_path(self, number, extension = "traj", fallback = True, lattice_constant = None, basis_set = None, program = None, method = None, xc = None, additional_atoms_calculation = False, ice_structure_calculation = False):
        """ 
            Reads the most fitting and recently executed file path with given input (basis_set, program, method, xc)
            for the given file extension (default: traj)
        """
        file_path = self._get_latest_file_path(number,  extension=extension, lattice_constant = lattice_constant, basis_set = basis_set, program = program, method = method, xc = xc, additional_atoms_calculation = additional_atoms_calculation, ice_structure_calculation = ice_structure_calculation)
        if file_path is None and fallback:
            # try with same method, but no xc
            file_path = self._get_latest_file_path(number, extension=extension, lattice_constant = lattice_constant, basis_set = basis_set, program = program, method = method, xc = None, additional_atoms_calculation = additional_atoms_calculation, ice_structure_calculation = ice_structure_calculation)
            if file_path is None:
                #try with same all but with no program input
                file_path = self._get_latest_file_path(number,  extension=extension, lattice_constant = lattice_constant, basis_set = basis_set, program = None, method = method, xc = xc, additional_atoms_calculation = additional_atoms_calculation, ice_structure_calculation = ice_structure_calculation)
                if file_path is None:
                    #try with same all but with no basis_set input
                    file_path = self._get_latest_file_path(number,  extension=extension, lattice_constant = lattice_constant, basis_set = None, program = program, method = method, xc = xc, additional_atoms_calculation = additional_atoms_calculation, ice_structure_calculation = ice_structure_calculation)
                    if file_path is None:
                        #try with same all but with no basis_set, xc or program input
                        file_path = self._get_latest_file_path(number,  extension=extension, lattice_constant = lattice_constant, basis_set = None, program = None, method = method, xc = None, additional_atoms_calculation = additional_atoms_calculation, ice_structure_calculation = ice_structure_calculation)
                        if file_path is None:
                            #finally try with no input
                            file_path = self._get_latest_file_path(number,  extension=extension, lattice_constant = lattice_constant, additional_atoms_calculation = additional_atoms_calculation, ice_structure_calculation = ice_structure_calculation)
        return file_path
        
        
        
    def get_labels_from_file_paths(self, file_paths):
        """
            Find uncommon factors from file paths, and give labels to the files
            according to them. For example, if two calculations in the filepaths
            have different xc-functionals, the xc-functionals are included in the
            labels. If all the basis-sets are the same, then basis sets are not
            included in the labels.
        """
        result = None
        order_indices = None
        # list that will contain all the parameters for all the file_paths
        if len(file_paths) > 0:
            all_parameters = []
            for file_path in file_paths:
                parameters = self.parse_parameters_from_filename(file_path)
                all_parameters.append(parameters)
        
        if len(all_parameters) > 0:
            label_keys = []
            keys = all_parameters[0].keys()
            # Go through all parameters 
            for key in keys:
                different_values = None
                # Go through all parameter sets
                # note, parameters is a dict
                for parameters in all_parameters:
                    if different_values is None:
                        different_values = [parameters[key]]
                    elif parameters[key] not in different_values:
                        label_keys.append(key)
                        break
            order = ['number', 'program', 'method', 'xc', 'basis_set', 'number_of_relaxation']            
            # do this separately to get a decent order
            order_lists = []
            for key in order:
                if key in label_keys:
                    if result is None:
                        initialized = False
                        result = []
                    else:
                        initialized = True
                    order_list = []
                   
                    # get the label values as csv strings
                    for i, parameters in enumerate(all_parameters):
                        if initialized:
                            result[i] += ", %s" % parameters[key]
                        else:
                            result.append("%s" % parameters[key]) 
                        order_list.append(parameters[key])
                    order_lists.append(order_list)
            # order in reversed order (we will get it the right way by this)
            order_indices = np.lexsort(order_lists[::-1])
        return result, order_indices

    def read_latest_structure_from_file(self, number, fallback = True, lattice_constant = None, basis_set = None, program = None, method = None, xc = None, additional_atoms_calculation = False, ice_structure_calculation = False):
        trajectory_path = self.get_latest_file_path(number, fallback = fallback, lattice_constant = lattice_constant, basis_set = basis_set, program = program, method = method, xc = xc, additional_atoms_calculation = additional_atoms_calculation, ice_structure_calculation = ice_structure_calculation)
        if trajectory_path is not None:
            print trajectory_path
            return self._read_structure_with_filename(trajectory_path)
        return None

    def read_latest_cube_from_file(self, number, extension = ".cube", fallback = True, lattice_constant = None, basis_set = None, program = None, method = None, xc = None, additional_atoms_calculation = False, ice_structure_calculation = False):
        from misc.cube import read_cube
        cube_path = self.get_latest_file_path(number, extension = extension, fallback = fallback, lattice_constant = lattice_constant, basis_set = basis_set, program = program, method = method, xc = xc, additional_atoms_calculation = additional_atoms_calculation, ice_structure_calculation = ice_structure_calculation)
        if cube_path is not None:
            return read_cube(cube_path)
        return None

    def read_structure_from_file(self, number, fallback = True, extension = None, number_of_relaxation = None, lattice_constant = None, basis_set = None, additional_atoms_calculation = False, xc = None, program = None, method = None, ice_structure_calculation = False):

        if extension == None:
            extension = self.extension
        if number_of_relaxation == None:
            number_of_relaxation = self.number_of_relaxation
        dir, filename = self.get_directory_and_filename(number, loading = True, extension = extension, lattice_constant = lattice_constant, number_of_relaxation = number_of_relaxation, basis_set = basis_set, program = program, method = method, additional_atoms_calculation = additional_atoms_calculation, ice_structure_calculation = ice_structure_calculation)
        filename = dir + "/" + filename

        # Fall back to the gpaw pbe calculated structure, if the special cases are not found  --- DISABLED, not very good habit to program things like this
        #if fallback and self.number_of_relaxation != -1 and not os.path.exists(filename):
        #    filename = self.get_folder()+"relaxed/relaxed_structure_%i/relaxed_structure_%i_%i.traj" % (number, number, number_of_relaxation)

        return self._read_structure_with_filename(filename)
    
    def _read_structure_with_filename(self, filename):
        if self.debug:
            print "Trying to read structure with path \"%s\"" % filename
        if os.path.exists(filename):
            try:
                if self.debug:
                    print " - Path exists"
                atoms = ase.io.read(filename)
                if self.number_of_relaxation == -1 and filename.endswith('xyz'):
                    atoms.center(vacuum=5)
                    atoms.pbc = False
                if self.debug:
                    print " - Reading file was successful"
                return atoms
            except IOError:
                if self.debug:
                    print " - Reading file failed"
                return None
        else:
            if self.debug:
                print " - Path does not exist"
            return None

    def get_image_file_name(self, file_identifier, universal = False, xc = None, method = None, program = None, basis_set = None, extension = "png"):
        return self.get_image_folder()+self.get_identifier_for_result_file(file_identifier, universal = universal, xc = xc, method = method, program = program, basis_set = basis_set)+'.'+extension 

    def get_text_file_name(self, file_identifier, universal = False, xc = None, method = None, basis_set = None, program = None):
        return self.get_text_folder()+self.get_identifier_for_result_file(file_identifier, universal = universal, xc = xc, method = method, program = program, basis_set = basis_set)+'.txt' 

    
    def get_data_file_name(self, file_identifier, universal = False, xc = None, method = None, basis_set = None, program = None):
        return self.get_data_folder()+self.get_identifier_for_result_file(file_identifier, universal = universal, xc = xc, method = method, program = program, basis_set = basis_set)+'.data' 

    def parse_parameters_from_filename(self, filename):
        # make sure that the directory is not included in the filename, as it contains only redundant
        # data (number)
        directory, filename = self.split_directory_and_filename(filename)
        parameters = {'number': None, 'number_of_relaxation': None, 'program': None, 'basis_set': None, 'xc': None, 'method': None, 'lattice_constant': None}
        words = filename.split("_")
        for word in words:
            if is_int(word):
                if parameters['number'] is not None:
                    parameters['number'] = int(word)
                else:
                    parameters['number_of_relaxation'] = int(word)
            if word.startswith("lc"):
                parameters['lattice_constant'] = float(word[2:])
            if word in ['tml', 'cp2k']:
                if word == 'tml':
                    parameters['program'] = 'TurboMole'
                else:
                    parameters['program'] = 'cp2k'
            elif word in ['BLYP', 'B3LYP', 'LDA']:
                parameters['xc'] = word
            if word in ['dzvp', 'dzvp-sr', 'tzvp', 'tzvp-sr', 'tzv2p', 'tzv2p-sr']:
                parameters['basis_set'] = word
            if word in ['mp2']:
                parameters['method'] = word
                
        # set this program defaults
        if parameters['method'] is None or parameters['method'] == 'dft':
            parameters['method'] = 'dft'
            if parameters['xc'] is None:
                parameters['xc'] = 'PBE'
        return parameters
        
    def split_directory_and_filename(self, filename):
        if "/" in filename:
            split_index = filename.rfind('/')
            return filename[0 : split_index+1], filename[split_index+1 : ]
        else:
            return None, filename
                

    def get_identifier_for_result_file(self, identifier, universal = False, xc = None, method = None, program = None, basis_set = None):
        if universal:
            return identifier
        result = identifier
        if method is None:
            method = self.method
        if program is None:
            program = self.program
        if basis_set is None:
            basis_set = self.basis_set
        
        if program is not None and program != 'GPAW':
            if program == 'TurboMole':
                result += "_tml"
            elif program == 'cp2k':
                result += "_cp2k"
            
        if method is None or method == 'dft':
            if xc is None:
                xc = self.xc
            if xc is not None and xc != 'PBE':
                result += "_" + xc
        else:
            result += "_" + method
            
        # basis_set
        if program is not None and program != 'GPAW' and basis_set is not None:
            result += "_" + basis_set
        return result

    def write_dict_to_file(self, dictionary, filename,  universal = False, xc = None, method = None, program = None):
        import pickle
        if type(dictionary) == list:
            result = {}
            for i, value in enumerate(dictionary):
                result[i] = value
            dictionary = result            
        output = open(self.get_data_file_name(filename, universal = universal, xc = xc, method = method, program = program), 'wb')
        pickle.dump(dictionary, output)
        output.close()


    def get_filename(self, number, number_of_relaxation = None, loading = False, extension = None, xc = None, program = None, method = None, folder = None, lattice_constant = None, basis_set = None, additional_atoms_calculation = False, ice_structure_calculation = False):
        dir, filename = self.get_directory_and_filename(number, number_of_relaxation = number_of_relaxation, loading = loading, extension = extension, xc = xc, program = program, method = method, folder = folder, lattice_constant = lattice_constant, basis_set = basis_set, additional_atoms_calculation = additional_atoms_calculation, ice_structure_calculation = False)
        return dir +"/"+ filename
        
    def get_directory_and_filename(self, number = None, number_of_relaxation = None, loading = False, extension = None, xc = None, program = None, method = None, folder = None, lattice_constant = None, basis_set = None, additional_atoms_calculation = False, ice_structure_calculation = False):
        if additional_atoms_calculation:
            return self.get_additional_atoms_directory_and_filename(number_of_relaxation = number_of_relaxation, loading = loading, extension = extension, xc = xc, program = program, method = method, folder = folder, lattice_constant = lattice_constant, basis_set = basis_set)
        elif ice_structure_calculation:
            return self.get_ice_structure_calculation_directory_and_filename(number, number_of_relaxation = number_of_relaxation, loading = loading, extension = extension, xc = xc, program = program, method = method, folder = folder, lattice_constant = lattice_constant, basis_set = basis_set)
        else:
            return self.get_structure_calculation_directory_and_filename(number, number_of_relaxation = number_of_relaxation, loading = loading, extension = extension, xc = xc, program = program, method = method, folder = folder, lattice_constant = lattice_constant, basis_set = basis_set)

    def get_additional_atoms_directory_and_filename(self, number_of_relaxation = None, loading = False, extension = None, xc = None, program = None, method = None, folder = None, lattice_constant = None, basis_set = None):
        if folder == None:
            folder = self.get_folder()
        if extension == 'eps' or extension == 'png':
            dir = self.get_folder()+"images"
        else:
            if lattice_constant is not None:
                dir = folder + "lattice_constant/additional_atoms"
            else:
                dir = folder + "relaxed/additional_atoms"

        if number_of_relaxation == None:
            number_of_relaxation = self.number_of_relaxation
       
        if loading and number_of_relaxation == -1:
            if extension == None:
                extension = self.extension
            filename = self.additional_atoms
            dir = folder
        else:
            if not loading:
                number_of_relaxation += 1
            if lattice_constant is None:
                filename = "additional_atoms_%i" % (number_of_relaxation)
            else:
                filename = "additional_atoms_%i_lc%.4f" % (number_of_relaxation, round(lattice_constant,4))
                
            filename = self.get_identifier_for_result_file(filename, xc = xc, method = method, program = program, basis_set = basis_set)
            

            if extension != None and extension != '':
                filename += "." + extension
        if not loading and rank == 0:
            if not os.path.exists(dir):
                os.makedirs(dir)
        return dir, filename

    def get_ice_structure_calculation_directory_and_filename(self, number, number_of_relaxation = None, loading = False, extension = None, xc = None, program = None, method = None, folder = None, lattice_constant = None, basis_set = None):
        """
            loading : if the structure is the initial structure, from which the relaxation is started (True) or the final result structure (False)
            extension : the file type of the structure (png, eps, traj, xyz)
            
        """
        if folder == None:
            folder = self.get_folder()
        if extension == 'eps' or extension == 'png':
            dir = self.get_folder()+"images"
        else:
            if lattice_constant is not None:
                dir = folder + "lattice_constant/ice_structure_%i" % number
            else:
                dir = folder + "ice_structure/relaxed_structure_%i" % number

        if number_of_relaxation == None:
            number_of_relaxation = self.number_of_relaxation
        
        if program == None:
            if loading and self.original_program is not None:
                program = self.original_program
            else:
                program = self.program
        if method == None:
            method = self.method

        
       
        if loading and number_of_relaxation == -1:
            if extension == None:
                extension = self.extension
            filename = "Water_structure_%i.%s" % (number, extension)
            dir = folder+"originals/"
        else:
            if not loading:
                number_of_relaxation += 1
            if lattice_constant is None:
                filename = "relaxed_structure_%i_%i" % (number, number_of_relaxation)
            else:
                filename = "relaxed_structure_%i_%i_lc%.4f" % (number, number_of_relaxation, round(lattice_constant,4))

            filename = self.get_identifier_for_result_file(filename, xc = xc, method = method, program = program, basis_set = basis_set)
            
            if extension != None and extension != '':
                filename += "." + extension
        if not loading and rank == 0:
            if not os.path.exists(dir):
                os.makedirs(dir)
        return dir, filename

        
    def get_structure_calculation_directory_and_filename(self, number, number_of_relaxation = None, loading = False, extension = None, xc = None, program = None, method = None, folder = None, lattice_constant = None, basis_set = None):
        """
            loading : if the structure is the initial structure, from which the relaxation is started (True) or the final result structure (False)
            extension : the file type of the structure (png, eps, traj, xyz)
            
        """
        if folder == None:
            folder = self.get_folder()
        if extension == 'eps' or extension == 'png':
            dir = self.get_folder()+"images"
        else:
            if lattice_constant is not None:
                dir = folder + "lattice_constant/structure_%i" % number
            else:
                dir = folder + "relaxed/relaxed_structure_%i" % number

        if number_of_relaxation == None:
            number_of_relaxation = self.number_of_relaxation
        
        if program == None:
            if loading and self.original_program is not None:
                program = self.original_program
            else:
                program = self.program
        if method == None:
            method = self.method

        
       
        if loading and number_of_relaxation == -1:
            if extension == None:
                extension = self.extension
            filename = "Water_structure_%i.%s" % (number, extension)
            dir = folder+"originals/"
        else:
            if not loading:
                number_of_relaxation += 1
            if lattice_constant is None:
                filename = "relaxed_structure_%i_%i" % (number, number_of_relaxation)
            else:
                filename = "relaxed_structure_%i_%i_lc%.4f" % (number, number_of_relaxation, round(lattice_constant,4))

            filename = self.get_identifier_for_result_file(filename, xc = xc, method = method, program = program, basis_set = basis_set)
            
            if extension != None and extension != '':
                filename += "." + extension
        if not loading and rank == 0:
            if not os.path.exists(dir):
                os.makedirs(dir)
        return dir, filename

    def calculation_finished(self, number = None, folder=None, number_of_relaxation=None, xc=None, lattice_constant = None, basis_set = None, additional_atoms_calculation = False, ice_structure_calculation = False):
        if self.debug:
            print "Checking if calculation was finished (additional_atoms_calculation: %s)" % (additional_atoms_calculation) 
        if additional_atoms_calculation:
            return self.additional_atoms_calculation_finished(folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant, basis_set = basis_set)
        elif ice_structure_calculation:
            return self.ice_structure_calculation_finished(number, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant, basis_set = basis_set)
        else:
            return self.structure_calculation_finished(number, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant, basis_set = basis_set)

    def calculation_started(self, number = None, folder=None, number_of_relaxation=None, xc=None, lattice_constant = None, basis_set = None, additional_atoms_calculation = False, ice_structure_calculation = False):
        if self.debug:
            print "Checking if calculation was started (additional_atoms_calculation: %s)" % (additional_atoms_calculation) 
        if additional_atoms_calculation:
            return self.additional_atoms_calculation_started(folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant, basis_set = basis_set)
        elif ice_structure_calculation:
            return self.ice_structure_calculation_started(number, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant, basis_set = basis_set)
        else:
            return self.structure_calculation_started(number, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant, basis_set = basis_set)

    def additional_atoms_calculation_finished(self, folder=None, number_of_relaxation=None, xc=None, lattice_constant = None, basis_set = None):
        if number_of_relaxation == None:
            number_of_relaxation = self.number_of_relaxation
        if xc == None:
            xc = self.xc
        if self.program == 'cp2k':
            return self._cp2k_additional_atoms_calculation_finished(0, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant, basis_set = basis_set)
        elif (self.program == 'GPAW'):
            return self._gpaw_additional_atoms_calculation_finished(0, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant)
        else:
            return self._ase_additional_atoms_calculation_finished(0, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant, basis_set = basis_set)

    def additional_atoms_calculation_started(self, folder=None, number_of_relaxation=None, xc=None, lattice_constant = None, basis_set = None):
        if number_of_relaxation == None:
            number_of_relaxation = self.number_of_relaxation
        if xc == None:
            xc = self.xc
        if self.program == 'cp2k':
            return self._cp2k_additional_atoms_calculation_started(0, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant, basis_set = basis_set)
        elif (self.program == 'GPAW'):
            return self._gpaw_additional_atoms_calculation_started(0, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant)
        else:
            return self._ase_additional_atoms_calculation_started(0, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant, basis_set = basis_set)

    def ice_structure_calculation_finished(self, number, folder=None, number_of_relaxation=None, xc=None, lattice_constant = None, basis_set = None):
        if number_of_relaxation == None:
            number_of_relaxation = self.number_of_relaxation
        if xc == None:
            xc = self.xc
        if self.program == 'cp2k':
            return self._cp2k_ice_structure_calculation_finished(number, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant, basis_set = basis_set)
        elif (self.program == 'GPAW'):
            return self._gpaw_ice_structure_calculation_finished(number, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant)
        else:
            return self._ase_ice_structure_calculation_finished(number, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant, basis_set = basis_set)

    def structure_calculation_finished(self, number, folder=None, number_of_relaxation=None, xc=None, lattice_constant = None, basis_set = None):
        if number_of_relaxation == None:
            number_of_relaxation = self.number_of_relaxation
        if xc == None:
            xc = self.xc
        if self.program == 'cp2k':
            return self._cp2k_structure_calculation_finished(number, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant, basis_set = basis_set)
        elif (self.program == 'GPAW'):
            return self._gpaw_structure_calculation_finished(number, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant)
        else:
            return self._ase_structure_calculation_finished(number, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant, basis_set = basis_set)

    def structure_calculation_started(self, number, folder=None, number_of_relaxation=None, xc=None, lattice_constant = None, basis_set = None):
        if number_of_relaxation == None:
            number_of_relaxation = self.number_of_relaxation
        if xc == None:
            xc = self.xc
        if self.program == 'cp2k':
            return self._cp2k_structure_calculation_started(number, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant, basis_set = basis_set)
        elif (self.program == 'GPAW'):
            return self._gpaw_structure_calculation_started(number, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant)
        else:
            return self._ase_structure_calculation_started(number, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant, basis_set = basis_set)

    def ice_structure_calculation_started(self, number, folder=None, number_of_relaxation=None, xc=None, lattice_constant = None, basis_set = None):
        if number_of_relaxation == None:
            number_of_relaxation = self.number_of_relaxation
        if xc == None:
            xc = self.xc
        if self.program == 'cp2k':
            return self._cp2k_ice_structure_calculation_started(number, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant, basis_set = basis_set)
        elif (self.program == 'GPAW'):
            return self._gpaw_ice_structure_calculation_started(number, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant)
        else:
            return self._ase_ice_structure_calculation_started(number, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc, lattice_constant = lattice_constant, basis_set = basis_set)

    def read_energy(self, number, folder=None, number_of_relaxation=None, xc=None, lattice_constant = None, basis_set = None, additional_atoms_calculation = False, ice_structure_calculation = False):
        filename = self.get_filename(number, folder=folder, loading = True, number_of_relaxation=number_of_relaxation, xc=xc, extension = "log", lattice_constant = lattice_constant, basis_set = basis_set, additional_atoms_calculation = additional_atoms_calculation)
        print filename
        if self.calculation_finished(number, folder=folder, number_of_relaxation=number_of_relaxation, xc=xc, lattice_constant = lattice_constant, basis_set = basis_set, additional_atoms_calculation = additional_atoms_calculation, ice_structure_calculation = ice_structure_calculation):
            file = open(filename, 'r')
            lines = file.readlines()
            if lines == None or len(lines) == 0:
                return None
            lastline = lines[-1]
            energy = float(lastline.split()[3])
            file.close()
            #print "%s: Finished" % filename
            return energy
        else:
            return None

    def read_latest_energy(self, number = None, folder=None, number_of_relaxation=None, xc=None, lattice_constant = None, basis_set = None, additional_atoms_calculation = False, ice_structure_calculation = False):
        """
            Returns the latest calculated finished energy of a structure

            Used in sublimation energy calculations
        """
        number_of_relaxation = 20
        latest_energy = None
        while number_of_relaxation >= 0:
            energy = self.read_energy(number, number_of_relaxation=number_of_relaxation, folder = folder, xc = xc, lattice_constant = lattice_constant, basis_set = basis_set, additional_atoms_calculation = additional_atoms_calculation, ice_structure_calculation = ice_structure_calculation)
            if energy != None:
                return energy
            number_of_relaxation -= 1
        return None


    def _ase_ice_structure_calculation_finished(self, number, folder = None, number_of_relaxation = None, xc = None, lattice_constant = None, basis_set = None):
        dir, filename = self.get_directory_and_filename(number, number_of_relaxation, True, "log", xc = xc, folder = folder, lattice_constant = lattice_constant, basis_set = basis_set, ice_structure_calculation = True)
        filename =  dir + "/" + filename
        return self._ase_calculation_finished(filename)
    
    def _ase_structure_calculation_finished(self, number, folder = None, number_of_relaxation = None, xc = None, lattice_constant = None, basis_set = None):
        dir, filename = self.get_directory_and_filename(number, number_of_relaxation, True, "log", xc = xc, folder = folder, lattice_constant = lattice_constant, basis_set = basis_set)
        filename =  dir + "/" + filename
        return self._ase_calculation_finished(filename)
    
    def _ase_calculation_finished(self, filename):
        """
            filename, the relative path to the .log-file resulting from ase calculation
        """
        if os.path.exists(filename):
            file = open(filename, 'r')
            lines = file.readlines()
            if lines == None or len(lines) == 0:
                return False
            lastline = lines[-1]
            force = float(lastline.split()[4])
            file.close()
            return (force < self.fmax)
        else:
            return False 

    def _gpaw_ice_structure_calculation_finished(self, number, folder = None, number_of_relaxation = None, xc = None, lattice_constant = None):
        dir, filename = self.get_directory_and_filename(number, number_of_relaxation, True, "gpw", xc = xc, folder = folder, lattice_constant = lattice_constant, ice_structure_calculation = True)
        gpwname =  dir + "/" + filename
        return self._gpaw_calculation_finished(gpwname)

    def _gpaw_structure_calculation_finished(self, number, folder = None, number_of_relaxation = None, xc = None, lattice_constant = None):
        dir, filename = self.get_directory_and_filename(number, number_of_relaxation, True, "gpw", xc = xc, folder = folder, lattice_constant = lattice_constant)
        gpwname =  dir + "/" + filename
        return self._gpaw_calculation_finished(gpwname)

    def _gpaw_calculation_finished(self, gpwname):
        if gpwname is not None and os.path.exists(gpwname):
            return True
        return False

    def _cp2k_additional_atoms_calculation_finished(self, number, folder=None, number_of_relaxation = None, xc = None, lattice_constant = None, basis_set = None):
        dir, filename = self.get_directory_and_filename(number, number_of_relaxation, True, "out", xc = xc, folder = folder, lattice_constant = lattice_constant, basis_set = basis_set, additional_atoms_calculation = True)
        outfilename = dir + "/" + filename
        dir, logfilename = self.get_directory_and_filename(number, number_of_relaxation, True, "log", xc = xc, folder = folder, lattice_constant = lattice_constant, basis_set = basis_set, additional_atoms_calculation = True)
        logfilename = dir + "/" + logfilename
        return self._cp2k_calculation_finished(outfilename, logfilename)

    def _cp2k_structure_calculation_finished(self, number, folder=None, number_of_relaxation = None, xc = None, lattice_constant = None, basis_set = None):
        dir, filename = self.get_directory_and_filename(number, number_of_relaxation, True, "out", xc = xc, folder = folder, lattice_constant = lattice_constant, basis_set = basis_set)
        outfilename = dir + "/" + filename
        dir, logfilename = self.get_directory_and_filename(number, number_of_relaxation, True, "log", xc = xc, folder = folder, lattice_constant = lattice_constant, basis_set = basis_set)
        logfilename = dir + "/" + logfilename
        return self._cp2k_calculation_finished(outfilename, logfilename)

    def _cp2k_ice_structure_calculation_finished(self, number, folder=None, number_of_relaxation = None, xc = None, lattice_constant = None, basis_set = None):
        dir, filename = self.get_directory_and_filename(number, number_of_relaxation, True, "out", xc = xc, folder = folder, lattice_constant = lattice_constant, basis_set = basis_set, ice_structure_calculation = True)
        outfilename = dir + "/" + filename
        dir, logfilename = self.get_directory_and_filename(number, number_of_relaxation, True, "log", xc = xc, folder = folder, lattice_constant = lattice_constant, basis_set = basis_set, ice_structure_calculation = True)
        logfilename = dir + "/" + logfilename
        return self._cp2k_calculation_finished(outfilename, logfilename)
        
    def _cp2k_calculation_finished(self, out_filename, log_filename): 
        
        if self.debug:
            print "Checking if CP2K-calculation with filename \"%s\" finished" % out_filename
        if os.path.exists(out_filename):
            if self.debug:
                print "  -File exists"
            
            if os.path.exists(log_filename):
                text = open(out_filename, 'r').read()
                lines = iter(text.split('\n'))
                for line in lines:
                    if line.startswith(' ***                    GEOMETRY OPTIMIZATION COMPLETED'):
                        if self.debug:
                            print "  -Calculation was finished"
                        return True
                    else:
                        if self.debug:
                            print "  -Calculation was not finished"
        else:
            if self.debug:
                print "  -File does not exist"
            
        return False

    def _cp2k_additional_atoms_calculation_started(self, number, folder=None, number_of_relaxation = None, xc = None, lattice_constant = None, basis_set = None):
        dir, filename = self.get_directory_and_filename(number, number_of_relaxation, True, "in", xc = xc, folder = folder, lattice_constant = lattice_constant, basis_set = basis_set, additional_atoms_calculation = True)
        posfilename = dir + "/cp2k-pos-1.xyz"         
        outfilename = dir + "/" + filename
        dir, logfilename = self.get_directory_and_filename(number, number_of_relaxation, True, "log", xc = xc, folder = folder, lattice_constant = lattice_constant, basis_set = basis_set, additional_atoms_calculation = True)
        logfilename = dir + "/" + logfilename
        return self._cp2k_calculation_started(outfilename, posfilename, logfilename)

    def _cp2k_ice_structure_calculation_started(self, number, folder=None, number_of_relaxation = None, xc = None, lattice_constant = None, basis_set = None):
        dir, filename = self.get_directory_and_filename(number, number_of_relaxation, True, "in", xc = xc, folder = folder, lattice_constant = lattice_constant, basis_set = basis_set, ice_structure_calculation = True)
        posfilename = dir + "/cp2k-pos-1.xyz"         
        outfilename = dir + "/" + filename
        dir, logfilename = self.get_directory_and_filename(number, number_of_relaxation, True, "log", xc = xc, folder = folder, lattice_constant = lattice_constant, basis_set = basis_set, ice_structure_calculation = True)
        logfilename = dir + "/" + logfilename
        return self._cp2k_calculation_started(outfilename, posfilename, logfilename)
    
    def _cp2k_structure_calculation_started(self, number, folder=None, number_of_relaxation = None, xc = None, lattice_constant = None, basis_set = None):
        dir, filename = self.get_directory_and_filename(number, number_of_relaxation, True, "in", xc = xc, folder = folder, lattice_constant = lattice_constant, basis_set = basis_set)
        posfilename = dir + "/cp2k-pos-1.xyz"         
        outfilename = dir + "/" + filename
        dir, logfilename = self.get_directory_and_filename(number, number_of_relaxation, True, "log", xc = xc, folder = folder, lattice_constant = lattice_constant, basis_set = basis_set)
        logfilename = dir + "/" + logfilename
        return self._cp2k_calculation_started(outfilename, posfilename, logfilename)

    def _cp2k_calculation_started(self, in_filename, pos_filename, log_filename): 
        
        if self.debug:
            print "Checking if CP2K-calculation with filename \"%s\" started" % in_filename
        if os.path.exists(in_filename):
            if self.debug:
                print "  -File exists"
            
            if os.path.exists(log_filename) or os.path.exists(pos_filename):
                if self.debug:
                    print "  -Calculation was started"
                return True
        else:
            if self.debug:
                print "  -File does not exist"
            
        return False


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
        

def add_options(parser, folder = None):
    from optparse import OptionGroup
    from system_commons import add_options
     
    
    group = OptionGroup(parser, "Energetic parameters", "Parameters that are used in energetic calculation and analysis.")  
    group.add_option("--program", 
                      action="store", type='string', dest="program", default="GPAW",
                      help="The program used in calculations. Supported programs: %s" % allowed_programs)
    group.add_option("--original_program", 
                      action="store", type='string', dest="original_program", default=None,
                      help="The program used the starting point calculation. Supported programs: %s" % allowed_programs)
    group.add_option("--method", 
                      action="store", type='string', dest="method", default="dft",
                      help="The method used in calculations. Supported methods: %s" % allowed_methods)
    group.add_option("-x", "--exchange_correlation", type="string",
                      action="store", dest="xc", default="PBE",
                      help="The exchange and correlation method used.")
    group.add_option("--extension", type="string",
                      action="store", dest="file_type", default="traj",
                      help="The extension of the stored original files, e.g. 'traj' or 'xyz'")
    group.add_option("-b", "--basis_set", type="string",
                      action="store", dest="basis_set", default=None,
                      help="The basis set used in Turbomole/cp2k calculations.")
    group.add_option("-s", "--number_of_relaxation", type="int",
                      action="store", dest="number_of_relaxation", default=-1,
                      help="The order number of the stucture handled or loaded as initial guess.")
    group.add_option("-m", "--maximum_force", type="float",
                      action="store", dest="fmax", default=0.05,
                      help="The maximum force tolerance eV per Angstrom")

    parser.add_option_group(group) 

    add_options(parser, folder)
    
     
        
