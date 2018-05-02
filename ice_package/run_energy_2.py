import os, copy, ase, random, pickle, sys, traceback
import numpy as np

from optparse import OptionParser
from ase.vibrations import Vibrations
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.infrared import InfraRed
from stools import mycalculators, cp2ktools
from time import time

from ase.io.trajectory import PickleTrajectory
import Queue

from misc.cube import read_cube
from structure_commons import StructureCommons
from structure_commons import add_options
from system_commons import rank, size, comm
from help_methods import remove_all_except, remove_indeces
from calculator.calculator import initialize_calculator


class EnergyCalculations(StructureCommons):
    def __init__(self, **kwargs):
        StructureCommons.__init__(self, **kwargs)
        self.random = kwargs['random']
        calculated_numbers_string = kwargs['calculated_numbers_string']
        if calculated_numbers_string != None:
            splits = calculated_numbers_string.split(",")
            self.calculated_numbers = []
            for split in splits:
                self.calculated_numbers.append(int(split))
        else:
            self.calculated_numbers = None
        #if run_single != None:
        #    self.calculated_numbers = [kwargs['run_single']]
        self.basis_set = kwargs['basis_set']
        self.charge = kwargs['charge']
        self.vibrations = kwargs['vibrations']
        self.calculate_lattice_constant = kwargs['calculate_lattice_constant']
        self.initial_lattice_constant = kwargs['initial_lattice_constant']
        self.additional_atoms_calculation = kwargs['additional_atoms_calculation']
        self.ice_structure_calculation = kwargs['ice_structure_calculation']
        self.wrkdir = kwargs['wrkdir']
        if self.wrkdir is None:
            self.wrkdir = "./"
        self.metals = kwargs['metals']
        self.mulliken = kwargs['mulliken']
        self.fix = kwargs['fix']
        self.calculate_dipole_moment = kwargs['calculate_dipole_moment']
        self.calculate_molecular_dipole_moments = kwargs['calculate_molecular_dipole_moments']
        self.calculate_charge_density = kwargs['calculate_charge_density']
        self.calculate_electron_density = kwargs['calculate_electron_density']
        self.calculate_charge_transfer = kwargs['calculate_charge_transfer']
        self.calculate_wannier_centers = kwargs['calculate_wannier_centers']
        self.calculate_electrostatic_potential = kwargs['calculate_electrostatic_potential']
        self.calculate_hb_charge_transfer = kwargs['calculate_hydrogen_bond_charge_transfer']
        self.calculate_molecule_bonding_energies = kwargs['calculate_bonding_energies']
        if self.calculate_dipole_moment or self.calculate_molecular_dipole_moments or self.calculate_charge_density or self.calculate_wannier_centers or self.calculate_electrostatic_potential or self.calculate_hb_charge_transfer:
            self.calculation_type = 'ENERGY'
        else:
            self.calculation_type = 'GEOMETRY_OPTIMIZATION'
        """if self.program is None or self.program == 'GPAW':
            self.calculator = GPAW()
        elif self.program == 'TurboMole':
            self.calculator = TurboMole()
        elif self.program == 'cp2k':
            self.calculator = TurboMole()"""

    
    
    def get_next(self, number = 0, init=False):
        if self.calculated_numbers != None:
            if rank == 0 and self.debug:
                print "Size of calculated numbers array %i" % len(self.calculated_numbers)
            if init:
                self.order_no = 0
            else:
                self.order_no += 1
            if self.order_no == len(self.calculated_numbers):
                print "All the numbers have been calculated!"
                return None
            else:
                if rank == 0 and self.debug:
                    print "Calculating %i:th number" % self.order_no
                    print "Number is %i" % self.calculated_numbers[self.order_no]
                number = self.calculated_numbers[self.order_no]
            
        elif self.random:
            if rank == 0:
                number = random.randint(0, self.number_of_results)
            else:
                number = 0
            if size > 1:
                number = comm.bcast(number,  root=0)
        else:
            if init:
                number = 0
            else:
                number += 1
            if size > 1:
                number = comm.bcast(number,  root=0)
        return number

    

    def run(self):
        if self.additional_atoms_calculation:
            self.run_additional_atoms()
        else:
            self.run_ice_structures(ice_structure_calculation = self.ice_structure_calculation)
            
    def run_additional_atoms(self):
        atoms = self.read(0, additional_atoms_calculation = True)
        if atoms == None:
            if rank == 0:
                print "Reading additional atoms resulted None, ending the process"
            return
        else:
            if self.vibrations:
                print "Vibrations calculation for additional atoms is not implemented."
                return
                #success = self.calculate_vibrations(atoms,  number)
            elif self.calculate_lattice_constant:
                success = self.optimize_lattice_constant(0, additional_atoms_calculation = True)
            else:
                success = self.calculate(0, additional_atoms_calculation = True)
            

    def run_ice_structures(self, ice_structure_calculation = False):
        number = self.get_next(init=True)
        try_count = 0
        while True:
            atoms = self.read(number, ice_structure_calculation = self.ice_structure_calculation)
            if atoms == None:
                if rank == 0:
                    if number is not None:
                        print "Reading atoms with number %i resulted None, ending the process" % number
                return
            else:
                if self.vibrations:
                    success = self.calculate_vibrations(atoms,  number, ice_structure_calculation = ice_structure_calculation)
                elif self.calculate_lattice_constant:
                    success = self.optimize_lattice_constant(number, ice_structure_calculation = ice_structure_calculation)
                elif self.calculate_charge_transfer:
                    success = self.calculate_charge_transfering(number)
                elif self.calculate_charge_density:
                    success = self.charge_density_calculation(number)
                elif self.calculate_electron_density:
                    success = self.electron_density_calculation(number)
                elif self.calculate_hb_charge_transfer:
                    success = self.calculate_hydrogen_bond_charge_transfer(number)
                elif self.calculate_molecule_bonding_energies:
                    success = self.calculate_bonding_energies(number)
                else:
                    success = self.calculate(number, ice_structure_calculation = ice_structure_calculation)
                if rank == 0:
                    if success:
                        try_count = 0
                    else:
                        try_count += 1
            number = self.get_next(number)
            if size > 1:
                try_count = comm.bcast(try_count,  root=0)
            if try_count > 10000:
                if rank == 0:
                    print "Tried to get next number 10000 times, ending the process"
                return
                
    def charge_density_calculation(self, number):
        """
            Calculates the charge density of the system
        """
        # The calculations are charge density calculations, so let's assure it
        calculator = initialize_calculator(self.program)
        calculator.set_general_parameters(self)
        
        number_of_relaxation = self.number_of_relaxation
        # reload original geometry optimization calculation that should be finished
        calculation = calculator.initialize_calculation(None, self.program, calculator, number, self.folder, number_of_relaxation, self.method, self.xc, None, self.basis_set, charge = self.charge, calculation_type = 'GEOMETRY_OPTIMIZATION', ice_structure_calculation = False, additional_atoms_calculation = False, debug = self.debug)
        # make sure that the calculation is finished
        if calculation.started() and calculation.finished():
            # do the all atoms charge density calculation
            atoms = calculation.get_final_structure()
            atoms_calculation = calculator.initialize_calculation(atoms, self.program, calculator, number, self.folder, number_of_relaxation, self.method, self.xc, None, self.basis_set, charge = self.charge, calculation_type = 'ENERGY', directory = calculation.directory, filename = calculation.filename+"_cd", debug = self.debug)
            atoms_calculation.start()            
        else:
            print "The original geometry optimization for structure number %i is not finished, please finish the original calculation before starting the charge density calculation." % number
            
    def electron_density_calculation(self, number):
        """
            Calculates the electron density of the system
        """
        calculator = initialize_calculator(self.program)
        calculator.set_general_parameters(self)
        
        number_of_relaxation = self.number_of_relaxation
        # reload original geometry optimization calculation that should be finished
        calculation = calculator.initialize_calculation(None, self.program, calculator, number, self.folder, number_of_relaxation, self.method, self.xc, None, self.basis_set, charge = self.charge, calculation_type = 'GEOMETRY_OPTIMIZATION', ice_structure_calculation = False, additional_atoms_calculation = False, debug = self.debug)
        # make sure that the calculation is finished
        if calculation.started() and calculation.finished():
            # do the all atoms charge density calculation
            atoms = calculation.get_final_structure()
            atoms_calculation = calculator.initialize_calculation(atoms, self.program, calculator, number, self.folder, number_of_relaxation, self.method, self.xc, None, self.basis_set, charge = self.charge, calculation_type = 'ENERGY', directory = calculation.directory, filename = calculation.filename+"_ed", debug = self.debug)
            atoms_calculation.start()            
        else:
            print "The original geometry optimization for structure number %i is not finished, please finish the original calculation before starting the charge density calculation." % number

    def calculate_charge_transfering(self, number):
        """
            Calculates the charge transfer between additional atoms and ice structure
        """

        if self.additional_atoms_calculation:
            raise Exception("Calculating charge transfer for additional atoms only is not possible.")
        elif self.ice_structure_calculation:
            raise Exception("Calculating charge transfer for ice structure atoms only is not possible.")
        elif self.read_additional_atoms() is None:
            raise Exception("Calculating charge transfer for structure that does not contain additional atoms is impossible.")

        # The calculations are charge density calculations, so let's assure it
        self.calculate_charge_density = True
        calculator = initialize_calculator(self.program)
        calculator.set_general_parameters(self)
        
        number_of_relaxation = self.number_of_relaxation
        # reload original geometry optimization calculation that should be finished
        calculation = calculator.initialize_calculation(None, self.program, calculator, number, self.folder, number_of_relaxation, self.method, self.xc, None, self.basis_set, charge = self.charge, calculation_type = 'GEOMETRY_OPTIMIZATION', ice_structure_calculation = False, additional_atoms_calculation = False, debug = self.debug)
        # make sure that the calculation is finished
        if calculation.started() and calculation.finished():
            # do the all atoms charge density calculation
            atoms = calculation.get_final_structure()
            atoms_calculation = calculator.initialize_calculation(atoms, self.program, calculator, number, self.folder, number_of_relaxation, self.method, self.xc, None, self.basis_set, charge = self.charge, calculation_type = 'ENERGY', directory = calculation.directory, filename = calculation.filename+"_cd", debug = self.debug)
            atoms_calculation.start()            
            atoms_charge_density = atoms_calculation.read_charge_density()

            # do the additional atoms charge density calculation
            additional_atoms = self.remove_ice_structure(atoms)
            additional_atoms_calculation = calculator.initialize_calculation(additional_atoms, self.program, calculator, number, self.folder, number_of_relaxation, self.method, self.xc, None, self.basis_set, charge = self.charge, calculation_type = 'ENERGY', directory = calculation.directory, filename = calculation.filename+"_cda", debug = self.debug)
            additional_atoms_calculation.start()
            additional_atoms_density = additional_atoms_calculation.read_charge_density()

            # do the ice structure charge density calculation
            ice_structure = self.remove_additional_atoms(atoms)
            ice_structure_calculation = calculator.initialize_calculation(ice_structure, self.program, calculator, number, self.folder, number_of_relaxation, self.method, self.xc, None, self.basis_set, charge = self.charge, calculation_type = 'ENERGY', directory = calculation.directory, filename = calculation.filename+"_cdi", debug = self.debug)
            ice_structure_calculation.start()
            ice_structure_density = ice_structure_calculation.read_charge_density()
        
            # reduce ice structure density and additional atoms density from the total density
            atoms_charge_density.reduce_density(additional_atoms_density)
            atoms_charge_density.reduce_density(ice_structure_density)
            atoms_charge_density.write_to_file(calculation.directory + "/" + calculation.filename +"_ct.cube") 
        else:
            print "The original geometry optimization for structure number %i is not finished, please finish the original calculation before starting the charge transfer calculation." % number

    def calculate_hydrogen_bond_charge_transfer(self, number):
        """
            Calculates the charge transfer between the water molecules.
                The process contains N+1 steps for each molecule
                 1. Calculate the total density of the structure
                 2.-N+1. Calculate total density for each molecule and reduce it from the total
                    density of the structure

        """

        if self.additional_atoms_calculation:
            raise Exception("Calculating hydrogen bond charge transfer for additional atoms only is not possible.")
        elif self.read_additional_atoms() is not None:
            raise Exception("Calculating hydrogen bond charge transfer for structure that contains additional atoms is impossible.")

        # The calculations are charge density calculations, so let's assure it
        self.calculate_charge_density = True
        calculator = initialize_calculator(self.program)
        calculator.set_general_parameters(self)
        
        number_of_relaxation = self.number_of_relaxation
        # reload original geometry optimization calculation that should be finished
        calculation = calculator.initialize_calculation(None, self.program, calculator, number, self.folder, number_of_relaxation, self.method, self.xc, None, self.basis_set, charge = self.charge, calculation_type = 'GEOMETRY_OPTIMIZATION', ice_structure_calculation = False, additional_atoms_calculation = False, debug = self.debug)
        
        

        # make sure that the geometry optimization is started and finished
        if not calculation.started():
            print "The original geometry optimization for structure number %i is not started, please perform the geometry optimization before doing charge transfer calculations." % number
        elif not calculation.finished():
            print "The original geometry optimization for structure number %i is not finished, please finish the geometry optimization before starting the charge transfer calculation." % number
        else:
            # retrieve the final structure
            atoms = calculation.get_final_structure()
        
    
            # do the all atoms charge density calculation
            atoms_calculation = calculator.initialize_calculation(atoms, self.program, calculator, number, self.folder, number_of_relaxation, self.method, self.xc, None, self.basis_set, charge = self.charge, calculation_type = 'ENERGY', directory = calculation.directory, filename = calculation.filename+"_cd", debug = self.debug)
            atoms_calculation.start()            
            atoms_charge_density = atoms_calculation.read_charge_density()

            
            # during the geometry optimization the structure could have changed, so we need to determine
            # the structure of the molecules again 
            nearest_neighbors_nos, bond_variables, elongated_bonds, elongated_hydrogen_bonds, nearest_neighbors_hydrogens, oxygen_indeces = self.get_bond_variables_from_atoms(atoms)

            # do the N charge density calculations for each molecule
            for i, oxygen_index in enumerate(oxygen_indeces):
                indeces = [oxygen_index]
                # determine which hydrogens are bonded to the oxygen and add
                # those to the calculation
                for j in range(4):
                    if bond_variables[i, j] == 1:
                        indeces.append(nearest_neighbors_hydrogens[i, j])

                # remove all except the molecule atoms
                molecule_atoms = remove_all_except(atoms, indeces)

                # initialize molecule calculation
                molecule_atoms_calculation = calculator.initialize_calculation(molecule_atoms, self.program, calculator, number, self.folder, number_of_relaxation, self.method, self.xc, None, self.basis_set, charge = self.charge, calculation_type = 'ENERGY', directory = calculation.directory, filename = calculation.filename+"_%icd" % i, debug = self.debug)
                # start
                molecule_atoms_calculation.start()
    
                if molecule_atoms_calculation.finished():
                    # reduce from the total density
                    atoms_charge_density.reduce_density(molecule_atoms_calculation.read_charge_density())
                    # reduce molecule calculation files to avoid flooding of the directory
                    molecule_atoms_calculation.delete_files()
                else:
                    raise Exception("Single molecule charge density calculation failed to finish.")

            
        
            # write the result to file
            atoms_charge_density.write_to_file(calculation.directory + "/" + calculation.filename +"_hbct.cube") 
    
    def calculate_bonding_energies(self, number):
        """
            Calculates the bonding energies per molecule by removing water molecules one by one
            (starting from the molecule with the largest number) and calculating the energies.
                The process contains N steps (i.e., one for each molecule)

        """

        if self.additional_atoms_calculation:
            raise Exception("Calculating bonding energies for additional atoms only is not possible.")
        elif self.read_additional_atoms() is not None:
            raise Exception("Calculating bonding energies for structure that contains additional atoms is impossible.")

        # Initialize the correct calculator
        calculator = initialize_calculator(self.program)
        calculator.set_general_parameters(self)
        
        number_of_relaxation = self.number_of_relaxation
        # reload original geometry optimization calculation that should be finished
        calculation = calculator.initialize_calculation(None, self.program, calculator, number, self.folder, number_of_relaxation, self.method, self.xc, None, self.basis_set, charge = self.charge, calculation_type = 'GEOMETRY_OPTIMIZATION', ice_structure_calculation = False, additional_atoms_calculation = False, debug = self.debug)
        
        

        # make sure that the geometry optimization is started and finished
        if not calculation.started():
            print "The original geometry optimization for structure number %i is not started, please perform the geometry optimization before doing bonding energy calculations." % number
        elif not calculation.finished():
            print "The original geometry optimization for structure number %i is not finished, please finish the geometry optimization before doing bonding energy calculations" % number
        else:
            # retrieve the final structure of the geometry optimization
            atoms = calculation.get_final_structure()
        
    
            # get the initial energy
            energy = calculation.get_energy()

            
            # during the geometry optimization the structure could have changed, so we need to determine
            # the structure of the molecules again 
            nearest_neighbors_nos, bond_variables, elongated_bonds, elongated_hydrogen_bonds, nearest_neighbors_hydrogens, oxygen_indeces = self.get_bond_variables_from_atoms(atoms)
            
            # initialize the results array and add the geometry optimization energy as the initial energy
            results = np.zeros(len(oxygen_indeces) +1, dtype=np.float)
            results[0] = energy
            
            # do the N charge electronic energy calculation
            N = len(oxygen_indeces)
            for i in range(N):
                indeces = []
                # determine which atoms are removed from the i:th calculation
                # (i+1) last oxygens and corresponding hydrogens
                if self.debug:
                    print "range from %i to %i" % (N -1, N - (i+1))
                for j in range(N - 1, N - (i + 1), -1):
                    indeces.append(oxygen_indeces[j])
                    # determine which hydrogens are bonded to the oxygen and remove
                    # those from the calculation
                    for k in range(4):
                        if bond_variables[j, k] == 1:
                            indeces.append(nearest_neighbors_hydrogens[j, k])
 
                # remove the atoms specified in 'indeces'
                structure_atoms = remove_indeces(atoms, indeces)

                # initialize molecule calculation
                structure_atoms_calculation = calculator.initialize_calculation(structure_atoms, self.program, calculator, number, self.folder, number_of_relaxation, self.method, self.xc, None, self.basis_set, charge = self.charge, calculation_type = 'ENERGY', directory = calculation.directory, filename = calculation.filename+"_%i" % i, debug = self.debug)
                # start
                structure_atoms_calculation.start()
    
                if structure_atoms_calculation.finished():
                    # add the energy to the results
                    results[i+1] = structure_atoms_calculation.get_energy()
                    # delete calculation files to avoid flooding of the directory
                    structure_atoms_calculation.delete_files()
                else:
                    raise Exception("Step %i of bonding energy calculation failed to finish. %i" % i)
                    
            # write the result to file
            np.savetxt(calculation.directory + "/" + calculation.filename +"_benes.txt", results, fmt='%.6f')
        

    def calculate(self, number, lattice_constant = None, additional_atoms_calculation = False, ice_structure_calculation = False):
        converged = False
        altered = False
        locked = False
        if lattice_constant is not None:
            if additional_atoms_calculation:
                initial_lattice_constant = self.get_additional_atoms_initial_lattice_constant()
            else:
                initial_lattice_constant = self.O_O_distance
            scale_factor = (lattice_constant/initial_lattice_constant)
            atoms = self.read(number, O_O_distance = initial_lattice_constant, number_of_relaxation = self.number_of_relaxation, additional_atoms_calculation = additional_atoms_calculation, ice_structure_calculation = ice_structure_calculation)
            reference_cell = atoms.get_cell() * 1.1
        else:
            scale_factor = 1.0
            reference_cell = None
        try:
            calculator = initialize_calculator(self.program)
            calculator.set_general_parameters(self)
            initial_number_of_relaxation = self.number_of_relaxation
            number_of_relaxation = initial_number_of_relaxation
            while number_of_relaxation < initial_number_of_relaxation + 3 and not converged:
                atoms = self.read(number, O_O_distance = lattice_constant, number_of_relaxation = number_of_relaxation, basis_set = self.basis_set, additional_atoms_calculation = additional_atoms_calculation, ice_structure_calculation = ice_structure_calculation)
                
                if atoms is not None:
                    if self.program == 'GPAW':
                        finished = self.calculate_gpaw(atoms, number, number_of_relaxation = number_of_relaxation, lattice_constant = lattice_constant)
                    elif self.program == 'TurboMole':
                        finished = self.calculate_turbomole(atoms, number, number_of_relaxation = number_of_relaxation, lattice_constant = lattice_constant)
                    elif self.program == 'cp2k':
                        calculation = calculator.initialize_calculation(atoms, self.program, calculator, number, self.folder, number_of_relaxation, self.method, self.xc, lattice_constant, self.basis_set, charge = self.charge, calculation_type = self.calculation_type, ice_structure_calculation = ice_structure_calculation, additional_atoms_calculation = additional_atoms_calculation, debug = self.debug)
                        finished = calculation.start()
                        locked = calculation.locked()
                        #finished = self.calculate_cp2k(atoms, number, number_of_relaxation = number_of_relaxation, lattice_constant = lattice_constant, reference_cell = reference_cell, basis_set = self.basis_set, additional_atoms_calculation = additional_atoms_calculation, ice_structure_calculation = ice_structure_calculation)
                    if self.debug:
                        print "Calculation finished: %s" % finished
                    converged = finished and self.calculation_finished(number, number_of_relaxation = number_of_relaxation +1, lattice_constant = lattice_constant, basis_set = self.basis_set, additional_atoms_calculation = additional_atoms_calculation, ice_structure_calculation = ice_structure_calculation)

                    if converged and not additional_atoms_calculation:
                        alteredd, dissosiated, assembled, proton_migration, hydrogen_bonds_broken, elongated_bonds, elongated_hydrogen_bonds, changed = self.structure_altered_during_relaxation(number, number_of_relaxation = number_of_relaxation +1, lattice_constant = lattice_constant, O_H_distance =  scale_factor * self.O_H_distance)
                        # Check if structure changed. We ignore elongation of bonds or hydrogen bonds
                        if changed or proton_migration or hydrogen_bonds_broken or dissosiated or assembled:
                            altered = True
                            break
                     
                    if locked:
                        break
                    if not converged:
                        number_of_relaxation += 1
                        if self.debug:
                            print "Relaxation number %i did not converge." % number_of_relaxation 
                    
                    
                    
                else:
                    break 
        except:
            print "Failed to calculate structure number %i:" % number
            print "----------------------------------------"
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            print "----------------------------------------"
        
        if locked:
            print "Calculation for structure number %i is locked" % number
        elif not converged:
            print "Failed to converge structure number %i" % number
        return converged, altered

    def calculate_vibrations(self, atoms, number):
        if self.turbomole:
            raise Exception("Calculating vibrations with turbomole is not possible for now")
        else:
            return self.calculate_vibrations_gpaw(atoms, number)  
    
    def optimize_additional_atoms_lattice_constant(self):
        self.optimize_lattice_constant(additional_atoms_calculation = True)
        

    def optimize_lattice_constant(self, number = None, additional_atoms_calculation = False, ice_structure_calculation = False): 
        if additional_atoms_calculation:
            self.logfile = open(self.get_folder()+"optimize_aa_lattice_constant_log%i.txt" % rank, 'w+')
        else:
            logfile_name = self.get_folder()+"optimize_lattice_constant%i_log%i.txt" % (number, rank)
            print logfile_name
            self.logfile = open(logfile_name, 'w+')
        number_of_relaxation = self.number_of_relaxation
        lattice_constant_queue = Queue.Queue()
        
        
        if additional_atoms_calculation:
            initial_lattice_constant = self.get_additional_atoms_initial_lattice_constant()
        else:
            initial_lattice_constant = self.O_O_distance

        # get three lattice constants for which the calculation is performed
        #  -mid_point, low_point, high_point
        #  -mid_point is the current minimum energy lattice constant
        #  -low_point and high_point are offset below and above the mid_point
        # use the initial lattice constant as a starting mid point

        offset = 0.03 * initial_lattice_constant
        mid_point = initial_lattice_constant
        high_point = mid_point + offset
        low_point = mid_point - offset
        lattice_constant_queue.put(mid_point)
        lattice_constant_queue.put(high_point)
        lattice_constant_queue.put(low_point)
        results = {}
        success = True
        if self.debug:
            print "Starting to optimize lattice constant"

        while not lattice_constant_queue.empty():
            lattice_constant = lattice_constant_queue.get()
            converged, altered = self.calculate(number, lattice_constant = lattice_constant, additional_atoms_calculation = additional_atoms_calculation, ice_structure_calculation = ice_structure_calculation)
            if altered:
                #if self.debug:  
                print_parallel("Structure with lattice constant %.4f altered during relaxation. i.e., it is not stable." % lattice_constant, self.logfile)      
                break
            elif converged:
                energy = self.read_latest_energy(number, lattice_constant = lattice_constant, basis_set = self.basis_set, additional_atoms_calculation = additional_atoms_calculation, ice_structure_calculation = ice_structure_calculation)
                results[lattice_constant] = energy
                print_parallel("Energy for lattice constant %.4f: %.4f eV" % (lattice_constant, energy), self.logfile)  
            else:
                print_parallel("Failed to converge a result with lattice constant %f" % lattice_constant, self.logfile)  
                success = False
                break
            if lattice_constant_queue.empty():
                # check if high point is lower in energy than the previous minimum
                if high_point in results and results[mid_point] > results[high_point]:
                    # check if also low_point is lower in energy, THIS SHOULD NOT HAPPEN
                    if results[mid_point] > results[low_point]:
                        print_parallel("Double minima lattice constant detected!", self.logfile)
                        if results[high_point] < results[low_point]:
                            low_point = mid_point
                            mid_point = high_point
                            high_point = mid_point + offset
                            lattice_constant_queue.put(high_point)
                            print_parallel("New mid point %.4f" % mid_point, self.logfile) 
                        else:
                            high_point = mid_point
                            mid_point = low_point
                            low_point = mid_point - offset
                            lattice_constant_queue.put(low_point)
                            print_parallel("New mid point %.4f" % mid_point, self.logfile) 
                    else:
                        # set high_point as the new mid_point (i.e., minimum energy structure)
                        low_point = mid_point
                        mid_point = high_point
                        high_point = mid_point + offset
                        # add new high_point to queue
                        lattice_constant_queue.put(high_point)
                        print_parallel("New mid point %.4f" % mid_point, self.logfile)  
                # check if low point is lower in energy than the previous minimum
                elif low_point in results and results[mid_point] > results[low_point]:
                    # set low_point as the new mid_point (i.e., minimum energy structure)
                    high_point = mid_point
                    mid_point = low_point
                    low_point = mid_point - offset
                    # add new low_point to queue
                    lattice_constant_queue.put(low_point)
                    print_parallel("New mid point %.4f" % mid_point, self.logfile)  
                else:
                    # if the mid_point remains the same, the offset is divided by two, meaning that the accuracy of the search
                    # is increased 
                    offset *= 0.5
                    high_point = mid_point  + offset
                    low_point = mid_point - offset
                    lattice_constant_queue.put(high_point)
                    lattice_constant_queue.put(low_point)
                if offset < 0.001:
                    print_parallel("Lattice constant converged to %f" % mid_point, self.logfile)  
                    break

        if success:
            directory, filename = self.get_directory_and_filename(number, lattice_constant = mid_point, additional_atoms_calculation = additional_atoms_calculation)    
            filename = directory + "/" + self.get_identifier_for_result_file("lc_optimization_summary") + ".p"  
            if self.debug:
                print "Writing summary of the process to %s" % filename 
            output = open(filename, 'wb')
            pickle.dump(results, output)
            output.close()
            print_parallel("Minimum energy lattice constant for number %i is %.3f" % (number, mid_point), self.logfile) 
            print_parallel("Minimum energy is %f eV" % results[mid_point] , self.logfile) 
        return success
            

    def calculate_vibrations_gpaw(self, atoms, number):
        try:
            vib_filename = "vibrations_%i_%i" % (number, self.number_of_relaxation)
            dir = self.folder+"relaxed/relaxed_structure_%i" % number
            filename = "relaxed_structure_%i_%i" % (number, self.number_of_relaxation)

            xc = self.xc
            if xc == "vdW-DF2":
                from gpaw.xc.vdw import FFTVDWFunctional
                xc = FFTVDWFunctional('vdW-DF2',Nalpha=size)

            if self.xc != "PBE":
                filename = filename + "_" + self.xc
                if self.xc == 'TS09':
                    xc = "PBE"
            if rank == 0:
                print "Calculating vibrations for %i" % number
            atoms = self.read(number)
            if atoms == None:
                return False
            calc = self.initialize_gpaw_calculator(atoms, xc, dir, vib_filename)

            stop = False
            if rank == 0:
                if not os.path.exists(dir) or not os.path.exists(dir+"/"+filename+".traj"): #or os.path.exists(dir+"/"+vib_filename+".txt"): 
                    stop = True
            if size > 1:
                stop = comm.bcast(stop,  root=0)
            if stop:
                return False

            if rank == 0:
                print "Calculating vibrations for %i" % number
            if rank == 0 and not os.path.exists(dir+"/ir/"):
                os.makedirs(dir+"/ir/")
            vib = InfraRed(atoms, name=dir+"/ir/ir")
            vib.run()
            #vib.summary(log = dir+"/"+vib_filename+".txt")
            vib.summary()
            vib.write_spectra(out=dir+"/ir-spectra.dat")
            for mode in range(len(vib.get_frequencies(method='frederiksen'))):
                vib.write_mode(mode)
        except:
            print "Calculating vibrations with GPAW for %i failed" % number
            return False
            
        return True
        
        
    def calculate_turbomole(self, atoms, number, number_of_relaxation = None, lattice_constant = None):
        dir, filename = self.get_directory_and_filename(number, extension = None, number_of_relaxation = number_of_relaxation, lattice_constant = lattice_constant)
        
        if self.method == 'mp2':
            tasks=["dscf","ricc2"]
            if (self.basis_set=="tzvpd"):
              basis="def2-TZVPD"; auxbasis="def2-TZVPD"; jkauxbasis="aug-cc-pVTZ";
            elif (self.basis_set=="szvpd"):
              basis="def2-SVPD"; auxbasis="def2-SVP"; jkauxbasis="def2-SVP"; # no auxbasis for def2-SVPD ..! use def2-SVP instead? # "svpd"
            elif (self.basis_set=="tzvpd+"):
              basis="def2-TZVPD"; auxbasis="def2-TZVP"; jkauxbasis="def2-TZVP"; # "tzvpd+"
            elif (self.basis_set=="tzvpd++"):
              basis="def2-TZVPD"; auxbasis="def2-SVP"; jkauxbasis="def2-SVP"; # "tzvpd++"
            calc=mycalculators.turbomole(dirname=dir,tasks=tasks,basis=basis, auxbasis=auxbasis, jkauxbasis=jkauxbasis)
            if (rank==0):
                calc.create_control(atoms,sections=[2],pars={"charge":self.charge})
        elif self.method == 'dft':
            
            tasks=["dscf","grad"] # dscf, use section 0
            sections=[0]
            
            if (self.basis_set == "szvpd"):
              turboshit=False; basis="def2-SVPD"; auxbasis="def2-SVPD"; jkauxbasis="def2-SVPD";
            elif (self.basis_set == "tzvpd"):
              turboshit=True; basis="def2-TZVPD"; auxbasis="aug-cc-pVTZ"; jkauxbasis="aug-cc-pVTZ"; # should probably always use this.. # def2 not in murska .. now it is!
            # turboshit=False; basis="aug-cc-pVTZ"; auxbasis="aug-cc-pVTZ"; jkauxbasis="aug-cc-pVTZ"; # this is *very* slow
            # turboshit=False; basis="def2-SVP"; auxbasis="def2-SVP"; jkauxbasis="def2-SVP"; # only good for testing..? for nothing else!
            # ******************
            
            morepars={"turbofunc":turbofunc,"rimem":800,"charge":self.charge} # some turbomole dft parameters
            if (turboshit):
              morepars["shit"]=True
            # pars={"dirname":dirname,"tasks":tasks,"sections":sections,"morepars":morepars} # encapsulate parameteres for a neb run..
            calc = mycalculators.turbomole(dirname=dir,tasks=tasks, basis=basis, auxbasis=auxbasis, jkauxbasis=jkauxbasis)
            
            if (rank==0):
              calc.create_control(atoms,sections=sections,pars=morepars)
        else:
            raise Exception("No method specified for Turbomole calculations")
    
        if not self.debug:
            atoms.set_calculator(calc)
            relax = ase.optimize.QuasiNewton(atoms, logfile=dir+"/"+filename+'.log',  trajectory=dir+"/"+filename+'.traj')
            relax.run(fmax=self.fmax)
        else:
            print "DEBUG: Skipping actual calculation with TurboMole"
        return True

        
    def calculate_gpaw(self, atoms, number, number_of_relaxation = None, lattice_constant = None):
        #ase.visualize.view(atoms)
        dir, filename = self.get_directory_and_filename(number, number_of_relaxation = number_of_relaxation, lattice_constant = lattice_constant)
        xc = self.xc
        if self.xc == 'TS09':
            xc = "PBE"
        if xc == "vdW-DF2":
            from gpaw.xc.vdw import FFTVDWFunctional
            xc = FFTVDWFunctional('vdW-DF2',Nalpha=size)
        stop = False
        
            # calculation is already running?
        if rank == 0 and os.path.exists(dir+"/"+filename+".traj"):
            stop = True
        if size > 1:
            stop = comm.bcast(stop,  root=0)
        if stop:
            return False
            
        if not self.debug:
            calc = self.initialize_gpaw_calculator(atoms, xc, dir, filename)
            relax = ase.optimize.QuasiNewton(atoms, logfile=dir+"/"+filename+'.log',  trajectory=dir+"/"+filename+'.traj')
            relax.run(fmax=self.fmax)
            calc.write(dir+"/"+filename+".gpw")
        else:
            print "DEBUG: Skipping actual calculation with GPAW"
        return True
    
    def initialize_gpaw_calculator(self, atoms, xc, dir, filename):
        calc = GPAW(h=0.2,  nbands=-20, xc=xc, txt=dir+"/"+filename+'.out')

        # TS09 from https://wiki.fysik.dtu.dk/gpaw/documentation/xc/vdwcorrection.html
        if self.xc == 'TS09':
            from ase.calculators.vdwcorrection import vdWTkatchenko09prl
            from gpaw.analyse.hirshfeld import HirshfeldDensity, HirshfeldPartitioning
            from gpaw.analyse.vdwradii import vdWradii
            calc = vdWTkatchenko09prl(HirshfeldPartitioning(calc), vdWradii(atoms.get_chemical_symbols(), 'PBE'))
        atoms.set_calculator(calc)
        return calc

    def get_cp2k_basis_set(self, basis_set):
        if basis_set is None or basis_set == 'dzvp-sr':
            return "DZVP-MOLOPT-SR-GTH"
        elif basis_set == 'dzvp':
            return "DZVP-MOLOPT-GTH"
        elif basis_set == "tzvp-sr":
            return "TZVP-MOLOPT-SR-GTH" 
        elif basis_set == "tzvp":
            return "TZVP-MOLOPT-GTH" 
        elif basis_set == "tzv2p-sr":
            return "TZV2P-MOLOPT-SR-GTH"
        elif basis_set == "tzv2p":
            return "TZV2P-MOLOPT-GTH"

    def calculate_cp2k(self, atoms, number = None, number_of_relaxation = None, lattice_constant = None, basis_set = None, additional_atoms_calculation = False, reference_cell = None, ice_structure_calculation = False):
        dir, filename = self.get_directory_and_filename(number = number, loading = self.mulliken, number_of_relaxation = number_of_relaxation, lattice_constant = lattice_constant, basis_set = basis_set, additional_atoms_calculation = additional_atoms_calculation, ice_structure_calculation = ice_structure_calculation)
        return self._calculate_cp2k(atoms, dir, filename, reference_cell = reference_cell, basis_set = basis_set)

    def _calculate_cp2k(self, atoms, dir, filename,  reference_cell = None, basis_set = None):
        """
          variable "pars" is a hierarchical dictionary, i.e. "dictionaries inside dictionaries inside .. etc"
          subdictionaries correspond to CP2K subsections as described in the manual:
          
          http://manual.cp2k.org/trunk/CP2K_INPUT.html
          
          .. for example, setting 
          
          http://manual.cp2k.org/trunk/CP2K_INPUT/MOTION/MD/THERMOSTAT.html
          
          i.e.
          
          &MOTION
          &MD
           ENSEMBLE NVE
           STEPS 20001
           TIMESTEP 0.5
           &THERMOSTAT
            TYPE NOSE
            &NOSE
             LENGTH 3
             YOSHIDA 3
             TIMECON  300.0
             MTS 2
            &END NOSE
           &END THERMOSTAT
          &END MD
          &END MOTION
          
          Here section MOTION => MD can be accessed via:
          
          pars["MOTION"]["MD"]
          
          Setting variables in whole section is done as follows:
          
          cp2ktools.stuff(pars,["MOTION","MD"],"inp",[
          "ENSEMBLE NVE",
          "STEPS 20001",
          "TIMESTEP 0.5"
          ]
          
          and
          
          cp2ktools.stuff(pars,["MOTION","MD","THERMOSTAT"],"inp",[
          "TYPE NOSE"
          ]
          
          etc..
          
          Changing a single variable:
          
          cp2ktools.set_input(pars,["MOTION","MD","THERMOSTAT],"TYPE","NOSE")
          
          Deleting a variable (or a section):
          
          cp2ktools.del_input(pars,["MOTION","MD"],"ENSEMBLE")
          
          Many sections can be invoked simply by calling subroutine from the cp2ktools module, for example:
          
          cp2ktools.metals(pars)
          
          Inserts parameters related to SCF cycle in calculations of metallic systems (i.e. pulay mixing, etc.)
          
          
       """
        
        # So, first, let's get some initial DFT parameters:
        pars=copy.deepcopy(cp2ktools.def_kari)
          
        # Modify, "fix", parameters .. set XC functional and consistent basis sets..
        cp2ktools.fixpars(pars, self.wrkdir, pseudo=self.xc,basis=self.get_cp2k_basis_set(basis_set))
        # .. look for the fixpars subroutine for more info!
        #
        # wrk = working directory in the cluster, should have the following files:
        # wrk+'/cp2k/libs/QS/BASIS_MOLOPT'
        # wrk+'/cp2k/libs/QS/GTH_POTENTIALS'
        #
        # .. how to find these?  Look at the info about modules (see also (**))
        # module show cp2k-env/2.4 
        # in taito, the files can be found in 
        # /appl/chem/cp2k/tests/QS/
        #
        # .. so, copy them into wrk+'/cp2k/etc..'
          
        # Let's use Grimme D3 dispersion correction:
        cp2ktools.toggle_vdw(pars, self.wrkdir, xcf=self.xc)
        # working directory should have the following file:
        # wrk+'/cp2k/libs/QS/dftd3.dat
          
        # Let's imagine you decide to change the charge of the system..
        # .. and have a parameter "charge" in your python file, then
        if (self.charge!=0):
            cp2ktools.set_input(pars,["FORCE_EVAL","DFT"],"CHARGE",str(self.charge))
          
        # choose XC functional..
        cp2ktools.stuff(pars, ["FORCE_EVAL","DFT","XC","XC_FUNCTIONAL "+self.xc], "inp", [])

        # set reference cell
        if reference_cell is not None:
            cp2ktools.bigcellref(pars, reference_cell)
            if self.debug:
                print "Setting reference lattice constants", reference_cell
          
        # the type of calculation, if calculating dipole moment or charge positions, do only one scf loop
        if self.calculate_dipole_moment or self.calculate_charge_positions:
            cp2ktools.set_input(pars, ["GLOBAL"], "RUN_TYPE", "ENERGY")
        else:
            cp2ktools.set_input(pars, ["GLOBAL"], "RUN_TYPE", "GEO_OPT")

        # next, insert info about velocities and atomic coordinates to the parameter dictionary:
        psolver="mt"
        usevelocities=False # use velocities present in the traj file or not
        
        # .. if the system is only 2D periodic, then the poisson solver defined in "psolver"
        # is used.  "mt" stands for the Martyna-Tuckermann (MT) solver.
        # IMPORTANT: when using MT solver, vacuum in the non-periodic dimension should be
        # at least "2*L" if "L" is the extent of your electron density

        # ******* ok, now you can choose either a native CP2K run or an ASE "slave" ***********
        # typ="cp2krelax"
        if (self.metals):
            cp2ktools.metals(pars)
        if self.mulliken:
            cp2ktools.mulliken(pars, filename=filename + "_mulliken.dat")
            relaxation_type = "aserelax"
        else:
            relaxation_type = "cp2krelax"
        if self.debug:
            fn = self.wrkdir+"/"+dir+"/"+filename+'.log'
            fnt = self.wrkdir+"/"+dir+"/"+filename+'.traj'
            fno = self.wrkdir+"/"+dir+"/"+filename+'.out'
            cp2ktools.dipanalysis2(pars, self.wrkdir+"/"+dir+"/"+filename)
            lines=cp2ktools.getlines(pars)
            
            # write cp2k into an input file..
            cp2ktools.writefile(self.wrkdir+"/"+dir+"/"+filename+".inp",lines)
            print "Skipping actual calculation with CP2K. Writing a fake file with random energy to \"%s\"" % fn
            print atoms.constraints
            import random
            ase.io.write(fnt, atoms)
            energy = (0.1)**2
            self.write_fake_cp2k_out_file(fno)
            self.write_fake_ase_log_file(fn, energy)
            
        elif relaxation_type=="cp2krelax":
            stop = False
            in_filename = self.wrkdir+"/"+dir+"/"+filename+".inp"
            log_filename = self.wrkdir+"/"+dir+"/"+filename+".log"
            out_filename = self.wrkdir+"/"+dir+"/"+filename+".out"
            pos_filename = self.wrkdir+"/"+dir+"/"+"cp2k-pos-1.xyz"
            # Check if the calculation has already finished of is processed
            if rank == 0:
                if os.path.exists(in_filename) and os.path.exists(out_filename):
                    if self._cp2k_calculation_locked(out_filename) or self._cp2k_calculation_finished(out_filename, log_filename):
                        stop = True
                    
            if size != 1:
                stop = comm.bcast(stop,  root=0)
            if stop:
                if self._cp2k_calculation_finished(out_filename, log_filename):
                    return True
                else:
                    return False
        
            
            # Because we are this far we know that calculation has been initialized (.inp file exists), is not locked
            # but has not finished. However, we do not know if any steps of the calculation have been made.
            # This can be checked with _cp2k_calculation_started. If this is the case
            # we continue from the last executed step
            if self._cp2k_calculation_started(in_filename, pos_filename, log_filename):
                cont = True
                start_atoms = ase.io.read(self.wrkdir+"/"+dir+"/"+filename+"_start.traj")
                if rank == 0:
                    self.write_cp2k_traj_file(dir, filename, start_atoms)
                # cont is just a dummy parameter to assure that rank 0 is finished writing the traj file, before others try
                # to read it
                if size != 1:
                    cont = comm.bcast(cont, root = 0)
                
            traj_path = self.wrkdir+"/"+dir+"/"+filename+".traj"
            if os.path.exists(traj_path):
                atoms = ase.io.read(traj_path)
                
            cp2ktools.atoms2cpk(pars, atoms, velocities=usevelocities, poisson=psolver)
            
            ase.io.write(self.wrkdir+"/"+dir+"/"+filename+"_start.traj", atoms, format = "traj")
            
            # Uncomment following if a dipole moment analysis is needed:
            # cp2ktools.dipanalysis2(pars, self.wrkdir+"/"+dir+"/"+filename)
            
            
            # write cp2k input file
            lines=cp2ktools.getlines(pars)
            cp2ktools.writefile(in_filename, lines)
            
            # launch execution
            cp2ktools.runcp2k(self.wrkdir+"/"+dir, filename+".inp", filename+".out")
            
            # write traj from pos-xyz
            if rank == 0:
                self.write_cp2k_traj_file(dir, filename, atoms)
            
            #cp2ktools.atoms2cpk(pars, result_atoms, velocities=usevelocities, poisson=psolver)
            energy = mycalculators.read_cp2k_energy(self.wrkdir+"/"+dir+"/"+filename+".out")
            self.write_fake_ase_log_file(self.wrkdir+"/"+dir+"/"+filename+'.log', energy)
            
        return True

    

    def _cp2k_calculation_locked(self, out_filename):
        seconds_from_modification = time() - os.path.getmtime(out_filename)
        return seconds_from_modification < 60.0

    def write_cp2k_traj_file(self, dir, filename, atoms):
        pos_path = self.wrkdir+"/"+dir+"/"+"cp2k-pos-1.xyz"
        if os.path.exists(pos_path):
            traj_path = self.wrkdir+"/"+dir+"/"+filename+".traj"
            index = 0
            if not os.path.exists(traj_path):
                result_atoms = ase.io.read(pos_path, index) 
                result_atoms.set_cell(atoms.get_cell())
                result_atoms.set_pbc(atoms.get_pbc())
                if atoms.constraints is not None:
                    for constraint in atoms.constraints:
                        result_atoms.set_constraint(constraint)
                traj = PickleTrajectory(traj_path, mode="w", atoms = result_atoms)
                traj.write()
                index = 1
            traj = PickleTrajectory(self.wrkdir+"/"+dir+"/"+filename+".traj", mode="a")
            while True:
                try:
                    result_atoms = ase.io.read(pos_path, index) 
                    result_atoms.set_cell(atoms.get_cell())
                    result_atoms.set_pbc(atoms.get_pbc())
                    if atoms.constraints is not None:
                        for constraint in atoms.constraints:
                            result_atoms.set_constraint(constraint)
                    traj.write(result_atoms)
                    index += 1
                except:
                    break
            os.remove(pos_path)
        else:
            if self.debug:
                print "%s does not exist. Not writing anything." % pos_path

    def write_fake_cp2k_out_file(self, file_name):
        contents = " ***                    GEOMETRY OPTIMIZATION COMPLETED"
        fo = open(file_name, "w")
        fo.write(contents)
        fo.close()

    def write_fake_ase_log_file(self, file_name, energy):
        contents = "BFGSLineSearch:   0  00:00:00  %f      %f" % (float(energy), self.fmax - 0.001)
        fo = open(file_name, "w")
        fo.write(contents)
        fo.close()

def print_parallel(text, file = None):
    if rank == 0:
        if file is not None:
            print >> file, text
        print text
                    

def main(folder = ''):
    parser = OptionParser()
    #parser.add_option("-n", "--number_of_results",
    #                  action="store", dest="number_of_results", type="int", default=0,
    #                  help="Number of generated geometries")
    parser.add_option("-r", "--random",
                      action="store_true", dest="random", default=False,
                      help="When applied the relaxed geometries are selected randomly")
    parser.add_option("-z", "--no_zeros",
                      action="store_true", dest="no_zeros", default=False,
                      help="When applied the geometries are taken from 'nozeros'-folder instead of 'originals'")
    #parser.add_option("--run_single_geometry", type="int",
    #                  action="store", dest="run_single", default=None,
    #                  help="Run a single structure")
    
    parser.add_option("--calculated_numbers", type="string",
                      action="store", dest="calculated_numbers_string", default=None,
                      help="Run structure numbers, separate with ','")
    parser.add_option("--additional_atoms_calculation", 
                      action="store_true", dest="additional_atoms_calculation", default=False,
                      help="Calculate only additional atoms.")
    parser.add_option("--ice_structure_calculation", 
                      action="store_true", dest="ice_structure_calculation", default=False,
                      help="Calculate only ice structure.")
    parser.add_option("--vibrations", 
                      action="store_true", dest="vibrations", default=False,
                      help="Calculate vibrations")
    parser.add_option("--metals", 
                      action="store_true", dest="metals", 
                      help="Structure has metals (CP2K)")
    parser.add_option("--calculate_charge_transfer", 
                      action="store_true", dest="calculate_charge_transfer", 
                      help="Calculates charge transfer between additional atoms and ice structure.")
    parser.add_option("--calculate_hydrogen_bond_charge_transfer", 
                      action="store_true", dest="calculate_hydrogen_bond_charge_transfer", 
                      help="Calculates charge transfer in the formation of hydrogen bonds in the structure.")
    parser.add_option("--calculate_dipole_moment", 
                      action="store_true", dest="calculate_dipole_moment", 
                      help="Calculates dipole moment (CP2K)")
    parser.add_option("--calculate_molecular_dipole_moments", 
                      action="store_true", dest="calculate_molecular_dipole_moments", 
                      help="Calculates molecular dipole moments (CP2K)")
    parser.add_option("--calculate_charge_density", 
                      action="store_true", dest="calculate_charge_density", 
                      help="Calculates charge density (CP2K)")
    parser.add_option("--calculate_electron_density", 
                      action="store_true", dest="calculate_electron_density", 
                      help="Calculates charge density (CP2K)")
    parser.add_option("--calculate_wannier_centers", 
                      action="store_true", dest="calculate_wannier_centers", 
                      help="Calculates wannier centers (CP2K)")
    parser.add_option("--calculate_electrostatic_potential", 
                      action="store_true", dest="calculate_electrostatic_potential", 
                      help="Calculates electrostatic potential (CP2K)")
    parser.add_option("--calculate_bonding_energies", 
                      action="store_true", dest="calculate_bonding_energies", 
                      help="Calculates bonding energies for each molecule at the time")
    parser.add_option("--mulliken", 
                      action="store_true", dest="mulliken", 
                      help="Get Mulliken charges (CP2K)")
    parser.add_option("--fix",
                      action="store", dest="fix", type="string", default=None,
                      help="Fix atom positions in given directions. Options: xy, xz, yz.")
    parser.add_option("--calculate_lattice_constant",
                      action="store_true", dest="calculate_lattice_constant",
                      help="The maximum force tolerance eV per Angstrom")
    parser.add_option("--initial_lattice_constant", type="float",
                      action="store", dest="initial_lattice_constant", default=2.7,
                      help="The maximum force tolerance eV per Angstrom")
    parser.add_option("--wrkdir", type="string",
                      action="store", dest="wrkdir", default=None,
                      help="Used work directory that contains the cp2k definitions.")

    add_options(parser, folder)
    (options, args) = parser.parse_args()

    

    
    #if (options.number_of_results != 0):
    EnergyCalculations(**vars(options)).run()



if __name__ == "__main__":
    main()
