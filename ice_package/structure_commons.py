# outer imports
import os, ase, numpy as np
# local imports
from structure_commons_cython import get_water_orientations_from_bond_variables_and_molecule_change_matrix, add_periodic_neighbors, get_single_molecule_hydrogen_coordinates, get_vector_from_periodicity_axis_number, get_selector_hydrogen_coordinates, get_coordination_numbers, get_bond_variables_from_atoms as get_bond_variables_from_atoms_cython
from energy_commons import EnergyCommons
from system_commons import rank
from help_methods import handle_periodic_input, get_oxygens, get_only_oxygens, get_oxygen_indeces, get_hydrogens, get_distance, get_periodic_distance, get_periodic_distances, get_opposite_periodicity_axis_number
import pickle

# type definitions
DTYPE = np.uint8
DTYPE2 = np.int8
DTYPE4 = np.float

class StructureCommons(EnergyCommons):
    """
        Class containing all the information related to structural parameters of the ice 
        cluster, and the additional atoms
        - Most significant functions include the nearest neighbor search and
          structure alteration checks
        - Should not be used as such, but through subclasses like Classification (of classification.py),
          HandleResults (handle_results.py) etc.
    """
    def __init__(self, **kwargs):
        EnergyCommons.__init__(self, **kwargs)
        # set structure parameter values, interface value is preferred over options value,
        # which is preferred over old settings value, and then hard coded
        # values are used. As a default the options value is None.      
        self.load_structure_settings()
        self.set_parameters(kwargs)
        if self.O_H_distance is None:
            self.O_H_distance = 1.0
        if self.O_O_distance is None:
            self.O_O_distance = 2.7
        if self.vacuum is None:
            self.vacuum = 5.0
        self.save_structure_settings()
        self.atoms = self.read_oxygen_raft_from_file()
        self.number_of_results = None
        
        if self.atoms is not None:
            self.oxygen_coordinates = self.atoms.get_positions()
            self.oxygen_coordinates += [self.x_offset, self.y_offset, self.z_offset]
            self.periodic = self.atoms.get_pbc()
            self.initialize_cell(self.atoms.get_cell())
            self.get_nearest_neighbors_nos()
            
            original_O_O_distance = self.get_average_O_O_distance(self.oxygen_coordinates, self.cell)
            # original_O_O_distance could be 0 if there is only one oxygen
            if original_O_O_distance == 0:
                scale_factor = 1.0
            else:
                scale_factor = self.O_O_distance / original_O_O_distance
            self.cell *= scale_factor
            self.oxygen_coordinates *= scale_factor
            self.atoms = ase.Atoms("O%i" % len(self.oxygen_coordinates), positions=self.oxygen_coordinates, cell=self.cell, pbc = self.periodic)
            #self.write_oxygen_raft_to_file()
        else:
            print "No oxygen raft detected. Asserting that this is an additional atoms project."
            self.nearest_neighbors_nos = None
        self.results = None
        
            
    def get_nearest_neighbors_nos(self):
        if not hasattr(self, 'nearest_neighbors_nos'):
            self.nearest_neighbors_nos = None

        if self.nearest_neighbors_nos is None and self.atoms is not None:
            self.nearest_neighbors_nos = all_nearest_neighbors_no(self.oxygen_coordinates, self.O_O_distance, periodic=self.periodic, cell=self.cell) 
        return self.nearest_neighbors_nos
              

    def initialize(self, cell = None, periodic = None, oxygen_coordinates = None):
        if periodic is not None or self.atoms is None:
            self.periodic = handle_periodic_input(periodic)
        if cell is not None or self.atoms is None:
            self.initialize_cell(cell)
        if oxygen_coordinates is not None:
            self.oxygen_coordinates = oxygen_coordinates
            self.oxygen_coordinates += [self.x_offset, self.y_offset, self.z_offset]
            self.atoms = ase.Atoms("O%i" % len(self.oxygen_coordinates), positions=self.oxygen_coordinates, cell=self.cell, pbc = self.periodic)
            #self.write_oxygen_raft_to_file()
            self.get_nearest_neighbors_nos()
        
            
            
            
        
        
    def save_structure_settings(self):
        if rank == 0:
            try:
                settings = { "O_H_distance": self.O_H_distance, "O_O_distance":  self.O_O_distance, 'vacuum': self.vacuum, 'charge': self.charge}
                pickle.dump( settings, open( self.get_folder()+"settings.p", "wb" ) )
            except:
                print "Saving structure settings failed"

    def load_structure_settings(self):
        try:
            settings = pickle.load( open( self.get_folder()+"settings.p", "rb" ) )
            self.O_H_distance = settings['O_H_distance']
            self.O_O_distance = settings['O_O_distance']
            self.vacuum = settings['vacuum']
            self.charge = settings['charge']
        except:
            self.O_H_distance = None
            self.O_O_distance = None
            self.vacuum = None
            self.charge = 0

    def structure_altered_during_relaxation(self, number, original_orientation = None, number_of_relaxation = None, lattice_constant = None, O_H_distance = None):
        """
            Check if structure has been altered during relaxation
            Returns a list with many different alteration possibilities
               --altered, dissosiated, assembled, proton_migration, 
               --hydrogen_bonds_broken, elongated_bonds, elongated_hydrogen_bonds, changed
        """ 
        if original_orientation is None:
            original_orientation = self.load_single_result(number)
        atoms = self.read(number, number_of_relaxation = number_of_relaxation, O_O_distance = lattice_constant)
        dissosiated = False
        assembled = False
        proton_migration = False
        hydrogen_bonds_broken = False
        elongated_bonds = False
        elongated_hydrogen_bonds = False
        changed = False
        if atoms == None:
            return None
        try:
            foreign_nearest_neighbors, bvv, elongated_bonds, elongated_hydrogen_bonds, nearest_neighbors_hydrogens, oxygen_indeces = self.get_bond_variables_from_atoms(atoms, O_O_distance = lattice_constant, O_H_distance = O_H_distance)
            wo = get_water_orientations_from_bond_variables_and_molecule_change_matrix(foreign_nearest_neighbors, self.nearest_neighbors_nos, bvv)
            changed = np.any(wo != original_orientation)
            if self.debug:            
                print "  * Proton configuration changed? %s." % changed
                print "  * Elongated bonds? %s" % elongated_bonds
                print "  * Elongated hydrogen bonds? %s" % elongated_hydrogen_bonds
                
            if changed:
                elongated_bond = False
                elongated_hydrogen_bonds = False
                charge2, dc2 = get_charge_and_dissosiation_count(wo)
                charge1, dc1 = get_charge_and_dissosiation_count(original_orientation)
                assert charge1 == charge2
                if dc1 < dc2:
                    dissosiated = True
                elif dc2 < dc1:
                    assembled = True
                else:
                    proton_migration = True
        except:
            import sys, traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            hydrogen_bonds_broken = True

        # altered means that any of the parameters is true
        altered = changed or dissosiated or assembled or proton_migration or hydrogen_bonds_broken or elongated_bonds or elongated_hydrogen_bonds
        return [altered, dissosiated, assembled, proton_migration, hydrogen_bonds_broken, elongated_bonds, elongated_hydrogen_bonds, changed]

    
    def view_oxygen_raft(self):
        oxygens = get_oxygens(self.oxygen_coordinates, cell = self.cell, periodic = self.periodic)
        ase.visualize.view(oxygens)

    def view_additional_atoms(self):
        additional_atoms = self.read_latest_additional_atoms()
        ase.visualize.view(additional_atoms)


    def write_oxygen_raft_to_file(self):
        if rank == 0:
            oxygens = get_oxygens(self.oxygen_coordinates)
            oxygens.pbc = self.periodic
            if self.cell != None:
                oxygens.set_cell(self.cell)
            else:
                oxygens.center(vacuum=5)  
            #print "Writing %soxygen_raft.traj" % (self.get_folder())
            ase.io.write(("%soxygen_raft.traj" % (self.get_folder())), oxygens, format="traj")    
    
    #def get_water_orientations_from_bond_variables_and_molecule_change_matrix(self, bond_variables,  equals = None):
    #    """
    #        Uses the old kind of bond variables (molecule1-molecule2-axis keyed dict) // OLD VERSION
    #    """
    #    if equals is None:
    #        equals = np.arange(0, self.N, dtype='int')
    #    result = []
    #    cdef DTYPE2_t[::1] bvv
    #    for i, new_molecule_no in enumerate(equals):
    #        bvv = np.array([-1, -1, -1, -1], dtype=DTYPE2)
    #        for j, nn in enumerate(self.nearest_neighbors_nos[0][i]):
    #            new_neighbor_no = equals[nn]
    #            axis = self.nearest_neighbors_nos[2][i][j]
    #            if new_neighbor_no == new_molecule_no and axis == 13:
    #                # TAKE care of the missing dangling oxygen bonds
    #                if new_neighbor_no not in bond_variables[new_molecule_no] or axis not in bond_variables[new_molecule_no][new_neighbor_no]:
    #                    bvv[j] = -1
    #                else:
    #                    bvv[j] = bond_variables[new_molecule_no][new_neighbor_no][axis]
    #                    assert bvv[j] == 1
    #            else:
    #                bvv[j] = bond_variables[new_molecule_no][new_neighbor_no][axis]
    #        result.append(get_water_orientation_from_bond_variable_values(bvv))
    #    return result

    def get_all_oxygen_coordinates(self):
        return None

    def get_bond_variables_from_atoms(self, atoms, O_O_distance = None, O_H_distance = None):
        if O_O_distance is None:
            O_O_distance = self.O_O_distance
        if O_H_distance is None:
            O_H_distance = self.O_H_distance
        return get_bond_variables_from_atoms(atoms, O_O_distance = O_O_distance, O_H_distance = O_H_distance, debug = self.debug)

    def get_average_O_O_distance(self, positions, cell):
        count = 0
        total_distance = 0.0
        for i in range(self.nearest_neighbors_nos.shape[1]):
            for j in range(4):
                if self.nearest_neighbors_nos[0, i, j] != i or self.nearest_neighbors_nos[2, i, j] != 13:
                    count += 1
                    p1 = positions[i]
                    p2 = positions[self.nearest_neighbors_nos[0, i, j]] + np.dot(get_vector_from_periodicity_axis_number(self.nearest_neighbors_nos[2, i, j]), cell)
                    total_distance += np.linalg.norm(p1-p2)
        # Do a zero division check if there is only one oxygen
        if count == 0:
            return 0.0
        else:
            return total_distance / count

    def get_degrees_of_freedom(self, preset_bond_values = None):
        return 4 * self.nearest_neighbors_nos.shape[1]   
    
    def load_single_result(self, number):
        return load_single_result(number, self.get_folder())

    def load_results(self):
        return load_results(self.get_folder())
    
    def read_oxygen_raft_from_file(self):
        return read_oxygen_raft_from_file(self.get_folder(), self.debug)

    def view_result(self, water_orientation):
        waters = self.get_atoms(water_orientation)         
        ase.visualize.view(waters)

    def initialize_cell(self, cell):
        self.cell = initialize_cell(cell, self.additional_atoms, self.periodic)

    def read(self, number = None, fallback = True, extension = None, number_of_relaxation = -1, O_O_distance = None, O_H_distance = None, basis_set = None, additional_atoms_calculation = False, ice_structure_calculation = False):
        if additional_atoms_calculation:
            return self.read_additional_atoms(extension = extension, number_of_relaxation = number_of_relaxation, lattice_constant = O_O_distance, basis_set = basis_set)
        elif ice_structure_calculation:
            return self.read_ice_structure(number, fallback = fallback, extension = extension, number_of_relaxation = number_of_relaxation, O_O_distance = O_O_distance, O_H_distance = O_H_distance, basis_set = basis_set)
        else:
            return self.read_whole_structure(number, fallback = fallback, extension = extension, number_of_relaxation = number_of_relaxation, O_O_distance = O_O_distance, O_H_distance = O_H_distance, basis_set = basis_set)


    def read_ice_structure(self, number, fallback = True, extension = None, number_of_relaxation = -1, O_O_distance = None, O_H_distance = None, basis_set = None, center = True):
        """
            Reads ice structure THAT DOES NOT contain anything but ice.
            read structure with given number and number of relaxation, etc.
            number_of_relaxation = -1 means that structure is generated (DEFAULT). 
             -number: number of the proton configuration
             -center: if the generated ice structure is positioned to the center of the cell. 
                This has no effect on structures read from a file 
        """
        if number is None:
            return None
        if number_of_relaxation is None:
            # Get the latest relaxation 
            if self.debug:
                print "Reading the latest ice structure number %i from the result (@structure_commons.read_ice_structure)" % (number)
            return self.read_latest_structure_from_file(number, fallback = fallback, lattice_constant = O_O_distance, basis_set = basis_set, program = self.program, method = self.method, xc = self.xc, ice_structure_calculation = True)
        # generate new structure
        elif number_of_relaxation == -1:
            result = self.load_single_result(number)
            if self.debug:
                print "Reading the initial ice geometry of water orientation %i (@structure_commons.read)"   % number
            atoms = self.get_atoms(result, O_O_distance = O_O_distance, O_H_distance = O_H_distance, add_additional_atoms = False, center = center)
            #self.fix_atoms(atoms)
            return atoms
        else:
            if self.debug:
                print "Reading the ice structure number %i from the result of relaxation number %i (@structure_commons.read_ice_structure)" % (number, number_of_relaxation)
            return self.read_structure_from_file(number, fallback = fallback, extension = extension, number_of_relaxation = number_of_relaxation, lattice_constant = O_O_distance, basis_set = basis_set, ice_structure_calculation = True)


    def read_latest_ice_structure(self, number, fallback = True, extension = None, O_O_distance = None, O_H_distance = None, basis_set = None, center = True):
        """
            Returns the latest relaxed ice structure that DOES NOT contain anything 
            else but water molecules (and OH and H3O+)
            - If structure has not been relaxed, a generated ice structure is returned
            center: if the generated ice structure is positioned to the center
             of the cell, has no effect on structures read on file 
        """
        result = self.read_ice_structure(number, fallback = fallback, number_of_relaxation = None, O_O_distance = O_O_distance, O_H_distance = O_H_distance, basis_set = basis_set)
        if result is None:
            result = self.read_ice_structure(number, fallback = fallback, number_of_relaxation = -1, O_O_distance = O_O_distance, O_H_distance = O_H_distance, basis_set = basis_set, center = center)
        return result

    def read_whole_structure(self, number, fallback = True, extension = None, number_of_relaxation = -1, O_O_distance = None, O_H_distance = None, basis_set = None):
        if number is None:
            return None
        if number_of_relaxation is None:
            # Get the latest relaxation, these will include additional atoms
            if self.debug:
                print "Reading the latest whole structure number %i from the result (@structure_commons.read_whole_structure)" % (number)
            return self.read_latest_structure_from_file(number, fallback = fallback, lattice_constant = O_O_distance, basis_set = basis_set, program = self.program, method = self.method, xc = self.xc)
        elif number_of_relaxation == -1:
            waters = self.read_latest_ice_structure(number, fallback = fallback, O_O_distance = O_O_distance, O_H_distance = O_H_distance, basis_set = basis_set, center = False)
            additional_atoms = self.read_latest_additional_atoms(lattice_constant = O_O_distance, basis_set = basis_set)
            atoms = self.get_atoms_from_waters_and_additional_atoms(waters, additional_atoms)
            if self.debug:
                print "Reading the initial geometry of whole structure with number %i (@structure_commons.read_whole_structure)"   % number
            #self.fix_atoms(atoms)
            return atoms
        else:
            if self.debug:
                print "Reading the structure number %i from the result of relaxation number %i (@structure_commons.read)" % (number, number_of_relaxation)
            return self.read_structure_from_file(number, fallback = fallback, extension = extension, number_of_relaxation = number_of_relaxation, lattice_constant = O_O_distance, basis_set = basis_set)

    def read_dipole_moment(self, number, program = None):
        if program is None:
            program = self.program
        if program is None or program == 'GPAW':
            filename = self.get_filename(number, program = program, extension='out')
            if self.calculation_finished(number):
                
                file = open(filename, 'r')
                result = []
             
                lines = file.readlines()
                for i in range(len(lines)-1, 0, -1):
                    if lines[i].startswith('Dipole Moment:'):
                        start_index = lines[i].find("[")+1
                        end_index = lines[i].find("]")
                        dipolem_string = lines[i][start_index:end_index].split()
                        for s in dipolem_string:
                            result.append(float(s) / 0.20822678)
                        break
                file.close()
                return result
        else:
            raise Exception("The dipole moment reading feature is implemented only for GPAW.")

    def read_dipole_moments(self):
        result = {}
        for number in range(self.get_number_of_results()):
            dipole_moment = self.read_dipole_moment(number)
            if dipole_moment != None:
                result[number] = norm = np.linalg.norm(dipole_moment)
        self.write_dict_to_file(result, "dipole_moment")
        return result

    def fix_atoms(self, atoms):
        if self.fix != None:
            from ase.constraints import FixedPlane
            constraints = []
            for i, atom in enumerate(atoms):
                if atom.symbol == 'O':
                    if self.fix == 'xy':
                        plane = FixedPlane(i, (0, 0, 1))
                        constraints.append(plane)
                    elif self.fix == 'xz':
                        plane = FixedPlane(i, (0, 1, 0))
                        constraints.append(plane)
                    elif self.fix == 'yz':
                        plane = FixedPlane(i, (1, 0, 0))
                        constraints.append(plane)
            atoms.set_constraint(constraints)

    def get_additional_atoms_initial_lattice_constant(self, extension = None):
        atoms = self.read_additional_atoms(number_of_relaxation = -1)
        return self.determine_lattice_constant(atoms)

    def determine_lattice_constant(self, atoms):
        positions = atoms.get_positions()
        chemical_symbols = atoms.get_chemical_symbols()
        position_0 = positions[0]
        symbol_0 = chemical_symbols[0]
        smallest_distance = None
        for i, position in enumerate(positions):
            if i != 0 and symbol_0 == chemical_symbols[i]:
                distance = np.sqrt(np.sum(np.power(position_0 - position, 2)))
                if smallest_distance is None or distance < smallest_distance:
                    smallest_distance = distance

        return smallest_distance

            
            
    def read_additional_atoms(self, extension = None, number_of_relaxation = -1, lattice_constant = None, basis_set = None):
        if number_of_relaxation is None:
            # Get the latest relaxation 
            return self.read_latest_structure_from_file(0, fallback = True, lattice_constant = lattice_constant, basis_set = basis_set, program = self.program, method = self.method, xc = self.xc, additional_atoms_calculation = True)
        elif number_of_relaxation == -1:
            if self.debug:
                print "Reading the initial geometry of additional atoms  (@structure_commons.read)"   
            if self.additional_atoms is not None:        
                additional_atoms_path = self.additional_atoms
            else: # use the default path
                additional_atoms_path = self.get_folder()+"additional_atoms.traj"

            if (os.path.exists(additional_atoms_path)):
                atoms = ase.io.read(additional_atoms_path)
                # scale with the lattice constant if is not none
                if lattice_constant is not None:
                    
                    initial_lattice_constant = self.determine_lattice_constant(atoms)
                    # store constraints
                    constraints = atoms.constraints
                    # remove all constraints, so that we can scale these atoms too
                    atoms.set_constraint()
                    scale_factor = lattice_constant / initial_lattice_constant
                    if self.debug:
                        print "Scaling additional atoms with factor %f" % scale_factor
                    atoms.set_positions(atoms.get_positions() * scale_factor)
                    atoms.set_cell(atoms.get_cell() * scale_factor)
                    # after scaling, reset constraints
                    for constraint in constraints:
                        if constraint.index.dtype == bool: 
                            c = ase.constraints.FixAtoms(mask = constraint.index)
                            atoms.set_constraint(c)
                        else: 
                            raise NotImplementedError
                periodic = atoms.get_pbc()

                # if slab or wire, center in the middle of vacuum
                if type(periodic) == list or type(periodic) == np.ndarray and any(periodic):
                    for j, periodic_on_axis in enumerate(periodic):
                        if not periodic_on_axis:
                            atoms.center(vacuum = self.vacuum, axis = j)  
                #self.fix_atoms(atoms)
            else:
                if self.debug:
                    print "Additional atoms path (%s) does not exist." % additional_atoms_path
                atoms = None
            return atoms
        else:
            if self.debug:
                print "Reading the additional atoms from the result of relaxation number %i (@structure_commons.read)" % (number_of_relaxation)
            return self.read_structure_from_file(0, extension = extension, number_of_relaxation = number_of_relaxation, lattice_constant = lattice_constant, basis_set = basis_set, additional_atoms_calculation = True)

    def read_latest_additional_atoms(self, lattice_constant = None, basis_set = None):
        latest = self.read_additional_atoms(self, number_of_relaxation = None, lattice_constant = lattice_constant, basis_set = basis_set)
        # if latest is None, no calculations have been made for the additional atoms        
        if latest is None:
            latest = self.read_additional_atoms(number_of_relaxation = -1, lattice_constant = lattice_constant, basis_set = basis_set)
        return latest

    def get_atoms_from_waters_and_additional_atoms(self, waters, additional_atoms):
        """
            Extends 'waters' with 'additional_atoms'.
            Both parameters should be ase Atoms objects
              both can be None, but in this case nothing happens
        """
        # add additional atoms 
        if additional_atoms is not None:
            added_index = len(waters)
            if additional_atoms is not None:
                if self.additional_atoms_position is not None:
                    com = additional_atoms.get_center_of_mass()
                    positions = additional_atoms.get_positions()
                    positions -= com
                    positions += self.additional_atoms_position
                    additional_atoms.set_positions(positions)
                waters = waters.extend(additional_atoms)    
            waters.set_positions(waters.get_positions())
            # the additional atoms cell always overrides the waters cell, if it is periodic
            if additional_atoms is not None and (type(additional_atoms.get_pbc()) == list or type(additional_atoms.get_pbc()) == np.ndarray and any(additional_atoms.get_pbc())):
                waters.set_cell(additional_atoms.get_cell())
            
            # set constraints from additional atoms
            if additional_atoms is not None:
                if self.debug:
                    print "Handling additional atoms constraints (@structure_commons.get_atoms_from_waters_and_additional_atoms)", additional_atoms.constraints
                for constraint in additional_atoms.constraints:
                    if constraint.index.dtype == bool: 
                        new_mask = [False for h in range(constraint.index.shape[0] + added_index)]
                        for l, ind in enumerate(constraint.index):
                            new_mask[l + added_index] = ind
                        c = ase.constraints.FixAtoms(mask = new_mask)
                        waters.set_constraint(c)
                    else: 
                        raise NotImplementedError
        # if not periodic, or periodicity is not specified center the structure
        # if slab or wire, center the structure on the non-periodic axis 
        if self.periodic is None or ( type(self.periodic) == bool and not self.periodic ):
            waters.center(vacuum = self.vacuum)
        elif type(self.periodic) == list or type(self.periodic) == np.ndarray:
            for j, periodic_on_axis in enumerate(self.periodic):
                if not periodic_on_axis:
                    waters.center(vacuum = self.vacuum, axis = j)   
            #if self.cell is not None:
            #    waters.set_cell(self.cell)
            #else:
            #    waters.center(vacuum = self.vacuum)  
        return waters
        
    def get_average_atoms(self, O_O_distance = None, O_H_distance = None, add_additional_atoms = True, center = True):
        """
            This method generates an ase Atoms object from containing oxygen atoms and the positions of
            possible hydrogen atoms
            - add_additional_atoms determines if additional atoms are included to the object
            - center m
        """
        # initialize parameters
        if self.cell is not None:        
            cell = self.cell.copy()
        else:
            cell = None

        # initialize OH and OO distances if none is specified
        if O_H_distance == None:
            O_H_distance = self.O_H_distance
        if O_O_distance is None:
            O_O_distance = self.O_O_distance
            oxygen_coordinates = self.oxygen_coordinates
        # if OO distance is specified scale the lattice
        else:
            scale_factor = O_O_distance / self.O_O_distance    
            oxygen_coordinates = self.oxygen_coordinates * scale_factor
            if cell is not None:
                cell *= scale_factor
            O_H_distance *= scale_factor
        # get the ase Atoms object containing only oxygen atoms
        oxygens = get_oxygens(oxygen_coordinates, cell = cell, periodic = self.periodic)
        # get hydrogen positions in a numpy array
        hydrogen_positions = []
        hydrogen_positions = self.get_selector_hydrogen_coordinates(cell = cell, O_H_distance = O_H_distance)
        hydrogen_positions = np.reshape(hydrogen_positions, (-1, hydrogen_positions.shape[-1]))
        # get the ase Atoms object with hydrogen atoms
        hydrogens = get_hydrogens(hydrogen_positions)
        # add oxygens and hydrogens together to form a new ase Atoms object
        waters = oxygens.extend(hydrogens)

        
        if cell is not None:
            waters.set_cell(cell)
        waters.pbc = self.periodic


        # add additional atoms 
        if add_additional_atoms:
            added_index = len(waters)
            additional_atoms = self.read_latest_additional_atoms()
            if additional_atoms != None:
                if self.additional_atoms_position is not None:
                    com = additional_atoms.get_center_of_mass()
                    positions = additional_atoms.get_positions()
                    positions -= com
                    positions += self.additional_atoms_position
                    additional_atoms.set_positions(positions)
                waters = waters.extend(additional_atoms)    
            waters.set_positions(waters.get_positions())

            # the additional atoms cell always overrides the waters cell, if it is periodic
            if additional_atoms is not None and (type(additional_atoms.get_pbc()) == list or type(additional_atoms.get_pbc()) == np.ndarray and any(additional_atoms.get_pbc())):
                waters.set_cell(additional_atoms.get_cell())
            
            # set constraints from additional atoms
            if additional_atoms is not None:
                print "Setting additional atoms constraints"
                for constraint in additional_atoms.constraints:
                    print "Here"
                    if constraint.index.dtype == bool: 
                        new_mask = [False for h in range(constraint.index.shape[0] + added_index)]
                        for l, ind in enumerate(constraint.index):
                            new_mask[l + added_index] = ind
                        c = ase.constraints.FixAtoms(mask = new_mask)
                        waters.set_constraint(c)
                    else: 
                        raise NotImplementedError

        # if not periodic, or periodicity is not specified center the structure
        # if slab or wire, center the structure on the non-periodic axis 
        if center:
            if self.periodic is None or ( type(self.periodic) == bool and not self.periodic ):
                waters.center(vacuum = self.vacuum)
            elif type(self.periodic) == list or type(self.periodic) == np.ndarray:
                for j, periodic_on_axis in enumerate(self.periodic):
                    if not periodic_on_axis:
                        waters.center(vacuum = self.vacuum, axis = j)   
                #if self.cell is not None:
                #    waters.set_cell(self.cell)
                #else:
                #    waters.center(vacuum = self.vacuum)  
        return waters

    def get_atoms(self, water_orientation, O_O_distance = None, O_H_distance = None, add_additional_atoms = True, center = True):
        """
            This method generates an ase Atoms object from water orientatios list
            - add_additional_atoms determines if additional atoms are included to the object
        """
        # initialize parameters
        if self.cell is not None:        
            cell = self.cell.copy()
        else:
            cell = None

        if O_H_distance == None:
            O_H_distance = self.O_H_distance
        if O_O_distance is None:
            O_O_distance = self.O_O_distance
            oxygen_coordinates = self.oxygen_coordinates
        else:
            scale_factor = O_O_distance / self.O_O_distance    
            oxygen_coordinates = self.oxygen_coordinates * scale_factor
            if cell is not None:
                cell *= scale_factor
            O_H_distance *= scale_factor
        # get the ase Atoms object containing only oxygen atoms
        oxygens = get_oxygens(oxygen_coordinates, cell = cell, periodic = self.periodic)
        # get hydrogen positions in a numpy array
        hydrogen_positions = []
        hydrogen_positions = self.get_hydrogen_coordinates(oxygen_coordinates, water_orientation,  self.nearest_neighbors_nos, cell, O_H_distance)
        # get the ase Atoms object with hydrogen atoms
        hydrogens = get_hydrogens(hydrogen_positions)
        # add oxygens and hydrogens together to form a new ase Atoms object
        waters = oxygens.extend(hydrogens)

        
        if cell is not None:
            waters.set_cell(cell)
        waters.pbc = self.periodic


        # add additional atoms 
        if add_additional_atoms:
            added_index = len(waters)
            additional_atoms = self.read_latest_additional_atoms()
            if additional_atoms != None:
                if self.additional_atoms_position is not None:
                    com = additional_atoms.get_center_of_mass()
                    positions = additional_atoms.get_positions()
                    positions -= com
                    positions += self.additional_atoms_position
                    additional_atoms.set_positions(positions)
                waters = waters.extend(additional_atoms)    
            waters.set_positions(waters.get_positions())

            # the additional atoms cell always overrides the waters cell, if it is periodic
            if additional_atoms is not None and (type(additional_atoms.get_pbc()) == list or type(additional_atoms.get_pbc()) == np.ndarray and any(additional_atoms.get_pbc())):
                waters.set_cell(additional_atoms.get_cell())
            
            # set constraints from additional atoms
            if additional_atoms is not None:
                print "Setting additional atoms constraints"
                for constraint in additional_atoms.constraints:
                    print "Here"
                    if constraint.index.dtype == bool: 
                        new_mask = [False for h in range(constraint.index.shape[0] + added_index)]
                        for l, ind in enumerate(constraint.index):
                            new_mask[l + added_index] = ind
                        c = ase.constraints.FixAtoms(mask = new_mask)
                        waters.set_constraint(c)
                    else: 
                        raise NotImplementedError

        # if not periodic, or periodicity is not specified center the structure
        # if slab or wire, center the structure on the non-periodic axis 
        if center:
            if self.periodic is None or ( type(self.periodic) == bool and not self.periodic ):
                waters.center(vacuum = self.vacuum)
            elif type(self.periodic) == list or type(self.periodic) == np.ndarray:
                for j, periodic_on_axis in enumerate(self.periodic):
                    if not periodic_on_axis:
                        waters.center(vacuum = self.vacuum, axis = j)   
                #if self.cell is not None:
                #    waters.set_cell(self.cell)
                #else:
                #    waters.center(vacuum = self.vacuum)  
        return waters

    def remove_additional_atoms(self, atoms):
        """
            Removes additional atoms and returns the result
            -used in charge transfer calculations
        """
        additional_atoms = self.read_additional_atoms()
        result = atoms.copy()
        start = len(atoms) - len(additional_atoms)
        end = len(atoms)
        # remove the start atoms end-start times
        for i in range(start, end):
            result.pop(i = start)
        return result

    def remove_ice_structure(self, atoms):
        """
            Removes atoms of the ice structure and returns the result
            -used in charge transfer calculations
        """
        additional_atoms = self.read_additional_atoms()
        result = atoms.copy()
        start = 0
        end = len(atoms) - len(additional_atoms)
        # Remove the first atom enough times that only the additional atoms are left
        for i in range(start, end):
            result.pop(i = 0)
        return result

    def get_hydrogen_coordinates(self, oxygen_positions, water_orientations, nearest_neighbors_nos, cell, O_H_distance, symmetry_operation=0):
        return get_hydrogen_coordinates(oxygen_positions, water_orientations, nearest_neighbors_nos, cell, O_H_distance, symmetry_operation = symmetry_operation)
        
    def get_selector_hydrogen_coordinates(self, coordination_numbers = None, preset_bond_values = None, O_H_distance = None, cell = None):
        # get parameters from the object if none are given
        if O_H_distance is None:
            O_H_distance = self.O_H_distance
        if cell is None:
            cell = self.cell
        if coordination_numbers is None:
            coordination_numbers = get_coordination_numbers(self.nearest_neighbors_nos)
        return get_selector_hydrogen_coordinates(coordination_numbers, self.oxygen_coordinates, self.nearest_neighbors_nos, cell, O_H_distance, preset_bond_values)

    
    def get_charge_and_dissosiation_count(self, water_orientations):
        return get_charge_and_dissosiation_count(water_orientations)

    def set_parameters(self, kwargs):
        if kwargs is not None:
            #self.write_geometries = kwargs['write_geometries']
            #self.invariant_count = kwargs['invariant_count']
            self.additional_atoms = kwargs['additional_atoms']
            self.folder = kwargs['folder']
            self.y_offset = kwargs['y_offset']
            self.x_offset = kwargs['x_offset']
            self.z_offset = kwargs['z_offset']
            self.x_rotation = kwargs['x_rotation']
            self.y_rotation = kwargs['y_rotation']
            self.z_rotation = kwargs['z_rotation']
            self.atom_radius = kwargs['atom_radius']
            self.image_type = kwargs['image_type']
            self.periodic = handle_periodic_input(kwargs['periodic'])
            if kwargs['additional_atoms_position'] is not None:
                try:
                    self.additional_atoms_position = np.array(kwargs['additional_atoms_position'].split(','), dtype=float)
                except:
                    print "Illegal additional atoms position, please give in format x,y,z (Example: 0.0,1.0,2.0)."
                    self.additional_atoms_position = None
            else:
                self.additional_atoms_position = None
            if kwargs['charge'] is not None:
                self.charge = kwargs['charge']
            if kwargs['vacuum'] is not None:
                self.vacuum = kwargs['vacuum']
            if kwargs['O_H_distance'] is not None:
                self.O_H_distance = kwargs['O_H_distance']
            if kwargs['O_O_distance'] is not None:
                self.O_O_distance = kwargs['O_O_distance']
        else:
            self.x_offset = 0.0
            self.y_offset = 0.0
            self.z_offset = 0.0
            self.x_rotation = 0.0
            self.y_rotation = 0.0
            self.z_rotation = 0.0
            self.atom_radius = 1.0
            self.image_type = "png"
            self.vacuum = 7.5
            self.periodic = False
            self.charge = 0
            self.additional_atoms_position = None

    def read_structures(self):
        """
            Reads structures as ase Atoms, returns dict with structure number as a key
        """
        result = {}
        for number in range(self.get_number_of_results()):
            atoms = self.read(number)
            if atoms != None:
                result[number] = atoms
        return result

    def has_results(self):
        results = self.get_results()
        return results is not None

    def get_results(self):
        if self.results is None:
            self.folder = self.get_folder()
            self.results = self.load_results()
            if self.results is None:
                self.number_of_results = 0
            else:
                self.number_of_results = len(self.results)
        return self.results

    def get_number_of_results(self):
        if self.number_of_results is None:
            result_file_path = self.folder+"allresults.txt"
            if os.path.exists(result_file_path):
                with open(result_file_path) as result_file:
                    count = sum(1 for line in result_file)
                self.number_of_results = count
            else:
                self.number_of_results = 0
        return self.number_of_results

    def read_structures(self, latest=False):
        result = {}
        for number in range(self.get_number_of_results()):
            if self.calculation_finished(number):
                atoms = self.read(number)
                if atoms != None:
                    result[number] = atoms
            
        print "Number of structures %i" % len(result)
        return result

def initialize_cell(cell, additional_atoms_path, periodic):
    additional_atoms = None
    additional_atoms_cell = None
    if (additional_atoms_path is not None and additional_atoms_path != '' and os.path.exists(additional_atoms_path)):
        additional_atoms = ase.io.read(additional_atoms_path)
        additional_atoms_cell = additional_atoms.get_cell()
    if cell is None:
        if additional_atoms_cell is None:
            if type(periodic) != bool or periodic:
                raise Exception("Invalid cell input. The system is set to be periodic, but no cell is available.")
            else:
                result = None
        else:
            result = additional_atoms_cell
    else:
        result = cell
    return result

def get_charge_and_dissosiation_count(water_orientations):
    OH_count = 0
    H3O_count = 0
    for water_orientation in water_orientations:
        if water_orientation > 9:
            OH_count += 1
        elif water_orientation > 5:
            H3O_count += 1
        
    return (H3O_count - OH_count), np.min([H3O_count, OH_count])

def read_oxygen_raft_from_file(folder, debug = False):
    try:
        if debug:
            print "Trying to read raft from %s" % ("%soxygen_raft.traj" % (folder))
            print "traj: exists? %s, xyz: exists? %s" % (os.path.exists("%soxygen_raft.traj" % (folder)), os.path.exists("%soxygen_raft.xyz" % (folder)))
        if os.path.exists("%soxygen_raft.xyz" % (folder)):
            return ase.io.read(("%soxygen_raft.xyz" % (folder)))
        elif os.path.exists("%soxygen_raft.traj" % (folder)):
            return ase.io.read("%soxygen_raft.traj" % (folder))
        else:
            return None
    except IOError:
        raise Exception("Returning to previous execution failed (%s). Please give the folder of the previous execution  with --folder \"FOLDER_NAME\" command or give the new oxygen raft with --oxygen_raft \"FILE_NAME\" command." % folder)
        




def get_bond_variables_from_atoms(atoms, O_O_distance = 2.7, O_H_distance = 1.0, debug = False):  
    """
        Parses nearest neighbors nos, bond variables and the elongated bonds of the ice structure
    """  
    return get_bond_variables_from_atoms_cython(atoms, O_O_distance = O_O_distance, O_H_distance = O_H_distance, debug = debug)
    oxygens = get_only_oxygens(atoms)
    oxygen_indeces = get_oxygen_indeces(atoms)
    nearest_hydrogens = {}
    elongated_bonds = False
    elongated_hydrogen_bonds = False
    periodic = atoms.get_pbc()
    cell = atoms.get_cell()
    seek_periodic = (type(periodic) == list and 1 in periodic) or (type(periodic) == np.ndarray and 1 in periodic) or (type(periodic) == bool and periodic)
    # Find the oxygens that are closest to each hydrogen and store the closest hydrogens indeces for
    # each oxygen atom
    for i, atom in enumerate(atoms):
        if atom.get_symbol() == 'H':
            smallest_distance = None
            smallest_distance_oxygen = -1
            axis_to_closest_oxygen = -1
            for j, oxygen in enumerate(oxygens):
                distance = np.linalg.norm(oxygen.get_position() - atom.get_position())
                if smallest_distance is None or distance < smallest_distance:
                    smallest_distance = distance
                    smallest_distance_oxygen = j
                    axis_to_closest_oxygen = 13
                if seek_periodic:
                    assert cell is not None
                    periodic_distances, axis = get_periodic_distances(atom.get_position(), oxygen.get_position(), cell)
                    for k, periodic_distance in enumerate(periodic_distances):
                        if smallest_distance == None or periodic_distance < smallest_distance:
                            axis_to_closest_oxygen = axis[k]
                            smallest_distance = periodic_distance
                            smallest_distance_oxygen = j
            # initialize the list
            if smallest_distance_oxygen not in nearest_hydrogens:
                nearest_hydrogens[smallest_distance_oxygen] = []
            # check if the bond is considered elongated
            if smallest_distance > 1.15:
                elongated_bonds = True
            # store for the smallest_distance oxygen
            nearest_hydrogens[smallest_distance_oxygen].append(i)

    # Get the bond variables by first finding the second closest oxygen to the closest hydrogen,
    # i.e., by finding the nearest neighbors nos
    nearest_neighbors_nos = np.empty((3, len(oxygens), 4), dtype=np.int_)
    nearest_neighbors_nos.fill(-1)

    # indeces for hydrogens in the atoms object
    nearest_neighbors_hydrogens = np.empty((len(oxygens), 4), dtype=np.int_)
    nearest_neighbors_hydrogens.fill(-1)
    bond_variables = np.empty((len(oxygens), 4), dtype=DTYPE2)
    
    for i in nearest_hydrogens:
        nearest_hydrogens_o = nearest_hydrogens[i]
        # Find the second closest oxygens:
        for hydrogen_no in nearest_hydrogens_o:
            smallest_distance = None # to second closest oxygen
            second_closest_oxygen = -1
            axis_to_second_closest_oxygen = -1
            hydrogen_position = atoms[hydrogen_no].get_position()
            
            for j, oxygen in enumerate(oxygens):
                oxygen_index = oxygen_indeces[j]
                # check that we are not comparing oxygen with itself 
                if i != j:
                    distance = np.linalg.norm(hydrogen_position-oxygens.get_positions()[j])
                    if smallest_distance == None or distance < smallest_distance:
                        axis_to_second_closest_oxygen = 13
                        smallest_distance = distance
                        second_closest_oxygen = j
                if seek_periodic:
                    assert cell != None
                    periodic_distances, axis = get_periodic_distances(atoms[hydrogen_no].get_position(), oxygens.get_positions()[j], cell)
                    for k, periodic_distance in enumerate(periodic_distances):
                        if axis[k] != 13 and (smallest_distance == None or periodic_distance < smallest_distance):
                            axis_to_second_closest_oxygen = axis[k]
                            smallest_distance = periodic_distance
                            second_closest_oxygen = j

            # If there is a second smallest distance oxygen
            if smallest_distance != None:
                if smallest_distance < 2.3:
                    if smallest_distance > 0.74 * O_O_distance:
                        elongated_hydrogen_bonds = True
                        if debug: 
                            print "Second closest oxygen is at %f Ang. LC is %f. O-H is %f" % (smallest_distance, O_O_distance, O_H_distance)
                    # Set the bond
                    #  -Iterate over the nearest neighbors until empty one (-1) is found
                    for j, neighbor_no in enumerate(nearest_neighbors_nos[0, i]):
                        if neighbor_no == -1:
                            nearest_neighbors_nos[0, i, j] = second_closest_oxygen
                            nearest_neighbors_nos[2, i, j] = axis_to_second_closest_oxygen
                            nearest_neighbors_nos[1, i, j] = axis_to_second_closest_oxygen != 13
                            nearest_neighbors_hydrogens[i, j] = hydrogen_no
                            bond_variables[i, j] = 1
                            break
                    # Set the opposite bond:
                    #  -Iterate over the nearest neighbors of second closest oxygen until empty one (-1) is found
                    for j, neighbor_no in enumerate(nearest_neighbors_nos[0, second_closest_oxygen]):
                        if neighbor_no == -1:
                            nearest_neighbors_nos[0, second_closest_oxygen, j] = i 
                            nearest_neighbors_nos[2, second_closest_oxygen, j] = get_opposite_periodicity_axis_number(axis_to_second_closest_oxygen)
                            nearest_neighbors_nos[1, second_closest_oxygen, j] = nearest_neighbors_nos[2, second_closest_oxygen, j] != 13
                            bond_variables[second_closest_oxygen, j] = -1
                            break
                else:  # No second closest oxygen found -> dangling hydrogen bond
                    for j, neighbor_no in enumerate(nearest_neighbors_nos[0, i]):
                        if neighbor_no == -1:
                            nearest_neighbors_nos[0, i, j] = i
                            nearest_neighbors_nos[1, i, j] = 0
                            nearest_neighbors_nos[2, i, j] = 13
                            nearest_neighbors_hydrogens[i, j] = hydrogen_no
                            bond_variables[i, j] = 1
                            break
    
    # Include dangling oxygen bonds to nearest_neighbors_nos
    for i in range(bond_variables.shape[0]):
        for j in range(bond_variables.shape[1]):
            if nearest_neighbors_nos[0, i, j] == -1:
                nearest_neighbors_nos[0, i, j] = i
                nearest_neighbors_nos[1, i, j] = 0
                nearest_neighbors_nos[2, i, j] = 13
                bond_variables[i, j] = -1
    
    return nearest_neighbors_nos, bond_variables, elongated_bonds, elongated_hydrogen_bonds, nearest_neighbors_hydrogens, oxygen_indeces

def load_single_result(number, folder):
    try:
        import linecache
        filename = folder+"allresults.txt"
        linecache.checkcache(filename)
        line = linecache.getline(filename, number +1)
        words = line.split()
        result = np.zeros(len(words), dtype=DTYPE2)
        for i, word in enumerate(words):
            result[i] = int(word)
        return result
    except:
        raise Exception("Failed loading result number %i" % number)
        
def get_hydrogen_coordinates(oxygen_positions, water_orientations, nearest_neighbors_nos, cell, O_H_distance, symmetry_operation=0):
    result = np.zeros((0, 3))
    site = 1
    for i in range(len(oxygen_positions)):
        result = np.vstack((result, get_single_molecule_hydrogen_coordinates(0, water_orientations[i], i, oxygen_positions,  nearest_neighbors_nos[0][i], nearest_neighbors_nos[1][i], nearest_neighbors_nos[2][i], cell, O_H_distance)))
        if site == 4:
            site = 1
        else:
            site = site +1
    return result

        
    
def load_results(folder, nozeros = False):
    """
        Numpy loading of results
    """
    if os.path.exists(folder+"nozero_results.txt") or os.path.exists(folder+"allresults.txt"):
        try:
            if nozeros:
                wo = np.loadtxt(folder+"nozero_results.txt",  dtype=DTYPE2)     
            else:
                wo = np.loadtxt(folder+"allresults.txt",  dtype=DTYPE2)
            #if wo == 0:
            #    wo = [0]
        except:
            return None  
    else:
        return None
    return wo 

def nearest_neighbors(atom_position, atom_positions, periodic=False, cell=None):
    result = np.ndarray(4)
    count = 0
    for i in range(len(atom_positions)):
        distance = get_distance(atom_positions[i], atom_position, periodic=periodic, cell=cell)
        if(distance == 2.76 or (distance > 2.510 and distance < 2.530)):
            result[count] = atom_positions[i]
            count = count + 1
    return result

def nearest_neighbors_no(O_O_distance,  atom_no,  atom_position, atom_positions, periodic=False, cell=None):
    result = np.array([],  dtype=DTYPE)
    periodicity = []
    periodicity_axis = []
    sortlist = np.zeros((0, 3),  dtype=DTYPE)
    count = 0
    min_distance = O_O_distance * 0.85
    max_distance = O_O_distance * 1.15
    for i in range(len(atom_positions)):
        distance = get_distance(atom_position,  atom_positions[i] , periodic=False)
        if (distance > min_distance and distance < max_distance):
            result = np.append(result, i)
            periodicity = np.append(periodicity, False)
            periodicity_axis.append(13)
            count = count + 1
            sortlist = np.vstack((sortlist,  atom_positions[i]))
        if periodic:
            result, periodicity, periodicity_axis, count, sortlist = add_periodic_neighbors(atom_position, atom_positions[i], cell, min_distance, max_distance, i, result, periodicity, periodicity_axis, count, sortlist)
            
    # following lines are deprecated because of the changes in the symmetry finding algorithm
    #ind = water_algorithm.sort_nearest_neighbors(atom_no,  sortlist,  result)
    #if ind != None:
    #    periodicity = np.array([periodicity[i] for i in ind],  dtype=DTYPE)
    #    periodicity_axis = np.array([periodicity_axis[i] for i in ind],  dtype=DTYPE)
    #    result = np.array([result[i] for i in ind],  dtype=DTYPE)
    return result, periodicity, periodicity_axis



def all_nearest_neighbors_no(atom_positions, O_O_distance, periodic=False, cell=None):
    """
        Nearest neighbors search. Searches all nearest neighbors for oxygen positions in
        atom_positions with the given O_O_distance. The python version is quite slow, and the
        version of structure_commons_cython should be used through this method.
    """
    try:
        from structure_commons_cython import find_nearest_neighbors_nos 
        return find_nearest_neighbors_nos(atom_positions, O_O_distance, periodic = periodic, cell = cell)
    except:
        import sys, traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        if type(periodic) == list or type(periodic) == np.ndarray:
            periodic = any(periodic)
     
        result = []
        periodicities = []
        periodicity_axis = []
        count = 0
        for i in range(len(atom_positions)):
            nos, periodicity, periodicity_ax = nearest_neighbors_no(O_O_distance,  i,  atom_positions[i], atom_positions, periodic, cell)
            nos, periodicity, periodicity_ax = add_dangling_bonds_to_nearest_neighbors(i,  nos,  periodicity, periodicity_ax)
            result.append(nos)
            periodicities.append(periodicity)
            periodicity_axis.append(periodicity_ax)
        return np.array([result, periodicities, periodicity_axis], dtype=DTYPE)

def add_dangling_bonds_to_nearest_neighbors(molecule_no,  nearest_neighbors_nos, periodicity, periodicity_axis):
    if len(nearest_neighbors_nos) == 4:
        return nearest_neighbors_nos,  periodicity, periodicity_axis
    result = nearest_neighbors_nos
    while result.shape[0] < 4:
        result = np.append(result, molecule_no)
        periodicity = np.append(periodicity, False)
        periodicity_axis = np.append(periodicity_axis, 13)
    return result,  periodicity, periodicity_axis

def write_results_to_file(oxygen_positions, results, cell,  nearest_neighbors_nos,  folder='', periodic=False, additional_atoms = None, vacuum = 5.0, O_H_distance = 1.0):
    for i, water_orientation in enumerate(results):
        write_result_to_file(oxygen_positions, water_orientation, i, cell, nearest_neighbors_nos, folder=folder, periodic=periodic, additional_atoms = additional_atoms, vacuum = vacuum, O_H_distance = O_H_distance)

def write_result_to_file(oxygen_positions, water_orientation, i, cell,  nearest_neighbors_nos,  folder='', periodic=False, additional_atoms = None, vacuum = 5.0, O_H_distance = 1.0):
    oxygens = get_oxygens(oxygen_positions)
    hydrogen_positions = []
    hydrogen_positions = get_hydrogen_coordinates(oxygen_positions, water_orientation,  nearest_neighbors_nos, cell, O_H_distance)
    hydrogens = get_hydrogens(hydrogen_positions)
    waters = oxygens.extend(hydrogens)
    if additional_atoms != None:
        added_index = len(waters)
        waters = waters.extend(additional_atoms)
        #waters.set_constraint()
        # Handle constraints:
        
        for constraint in additional_atoms.constraints:
            if constraint.index.dtype == bool: 
                new_mask = [False for h in range(constraint.index.shape[0] + added_index)]
                for l, ind in enumerate(constraint.index):
                    new_mask[l + added_index] = ind
                c = ase.constraints.FixAtoms(mask = new_mask)
                waters.set_constraint(c)
            else: 
                raise NotImplementedError
           
    #waters.set_positions(waters.get_positions())
    waters.pbc = periodic
    if cell is not None:
        waters.set_cell(cell)
    else:
        if additional_atoms is not None:
            waters.set_cell(additional_atoms.get_cell())
    if periodic is None or ( type(periodic) == bool and not periodic ):
        waters.center(vacuum=vacuum)
    elif type(periodic) == list or type(periodic) == np.ndarray:
        for j, periodic_on_axis in enumerate(periodic):
            if not periodic_on_axis:
                waters.center(vacuum = vacuum, axis = j)   
    ase.io.write(("%sWater_structure_%i.traj" % (folder,  i)), waters, format="traj") 

def parse_project_structures(atoms, exclude_atoms = None):
    """
        Parses Oxygen Raft and the additional atoms from the project
            Hydrogens are removed automatically
            All Oxygens are considered to be part of oxygen 
            TODO: do it
    """
     #nearest_neighbors_nos, bond_variables, elongated_bonds, elongated_hydrogen_bonds, nearest_neighbors_hydrogens = get_bond_variables_from_atoms(atoms, O_O_distance = 2.7, O_H_distance = 1.0, debug = False)
    #for i, atom in enumerate(atoms):
    #    pass
    
    


def add_options(parser, folder = None):
    from optparse import OptionGroup
    from energy_commons import add_options
    
    group = parser.get_option_group("--view_structure") 
    if group is not None:
        group.add_option("--write", dest='wa_method', const="write", action='store_const',
                          help="Write the geometries to individual files.")
        group.add_option("--write_structure", dest='function', const="write_structure", action='store_const',
                          help="Write the geometry with given number to individual file.")
        group.add_option("--write_structure_image", dest='function', const="write_structure_image", action='store_const',
                          help="Writes an image of the geometry with given number to individual file in the images folder.")
        #group.add_option("--write_no_zeros", dest='wa_method', const="write_no_zeros", action='store_const',
        #                  help="Write the structures that contain only possible O-O-O angles")
        group.add_option("--view_oxygen_raft", dest='function', const="view_oxygen_raft", action='store_const',
                          help="View the oxygen raft.")
        group.add_option("--view_additional_atoms", dest='function', const="view_additional_atoms", action='store_const',
                          help="View the oxygen raft.")
        group.add_option("--print_profiles", dest='wa_method', const="print_result_profiles", action='store_const',
                          help="Print dangling bond profiles of generated structures.")
        

    group = OptionGroup(parser, "Structure parameters", "Parameters used in the writing of proton configurations, i.e., with --write_structure, --write")
    group.add_option("--additional_atoms", dest='additional_atoms',  action='store', type="string", default=None,
                      help="Filename of the xyz/traj file that contains the (non-water) atoms added to the final structures.")
    group.add_option("--additional_atoms_position", dest='additional_atoms_position',  action='store', type="string", default=None,
                      help="The x y and z parameters of the center of mass of additional atoms.")
    group.add_option("--vacuum", dest='vacuum',  action='store', type=float, default=None,
                      help="The amount of vacuum (Angstrom) added to each non periodic direction.")
    group.add_option("--x_offset", dest='x_offset', action='store', default=0.0, type=float,
                          help="Offset of water molecules in x direction")
    group.add_option("--y_offset", dest='y_offset', action='store', default=0.0, type=float,
                          help="Offset of water molecules in y direction")
    group.add_option("--z_offset", dest='z_offset', action='store', default=0.0, type=float,
                          help="Offset of water molecules in z direction")
    group.add_option("--O_O_distance", dest='O_O_distance', action='store', default=None, type=float,
                          help="Distance between two neighboring oxygen atoms")
    group.add_option("--O_H_distance", dest='O_H_distance', action='store', default=None, type=float,
                          help="Distance between oxygen and hydrogens covalently bonded with it.")
    group.add_option("--charge", type="int",
                      action="store", dest="charge", default=None,
                      help="Charge of calculated structures")
    group.add_option("--periodicity", dest='periodic', type='string', action='store', default=None,
                      help="Periodicity of the structure. Options: 'none', 'x-wire', 'y-wire', 'z-wire', 'slab', 'bulk'.")
    parser.add_option_group(group)
    
    group = OptionGroup(parser, "Image parameters", "Parameters used in the writing of proton configuration images, i.e., with --write_structure_image")
    
    group.add_option("--x_rotation", dest='x_rotation', action='store', default=0.0, type=float,
                          help="x-axis rotation applied to the system when writing an image.")
    group.add_option("--y_rotation", dest='y_rotation', action='store', default=0.0, type=float,
                          help="y-axis rotation applied to the system when writing an image.")
    group.add_option("--z_rotation", dest='z_rotation', action='store', default=0.0, type=float,
                          help="z-axis rotation applied to the system when writing an image.")
    group.add_option("--atom_radius", dest='atom_radius', action='store', default=1.0, type=float,
                          help="atom radius when writing an image.")
    group.add_option("--image_type", dest='image_type', action='store', default="png", type='string',
                          help="written image type [png/eps/pov].")
    parser.add_option_group(group)

    add_options(parser, folder)

    
    parser.add_option_group(group)  











