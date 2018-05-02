import os, ase, copy
from time import time
from ase.io.trajectory import PickleTrajectory

from calculator import Calculator
from calculation import Calculation
from ..system_commons import rank, size, comm

from ..stools import mycalculators, cp2ktools

class CP2K(Calculator):
    def __init__(self):
        self.parameters = None

    def __str__(self):
        return "CP2K"

    def set_general_parameters(self, energy_calculations):
        # So, first, let's get some initial DFT parameters:
        self.wrkdir = energy_calculations.wrkdir
        self.metals = energy_calculations.metals
        self.mulliken = energy_calculations.mulliken
        self.calculate_dipole_moment = energy_calculations.calculate_dipole_moment
        self.calculate_molecular_dipole_moments = energy_calculations.calculate_molecular_dipole_moments
        self.calculate_charge_density = energy_calculations.calculate_charge_density
        self.calculate_wannier_centers = energy_calculations.calculate_wannier_centers
        self.calculate_electrostatic_potential = energy_calculations.calculate_electrostatic_potential
        self.calculate_electron_density = energy_calculations.calculate_electron_density


    def geometry_optimization(self, calculation):
        pars = self.initialize_calculator_parameters(calculation)

        # the type of calculation, if calculating dipole moment or charge positions, do only one scf loop
        cp2ktools.set_input(pars, ["GLOBAL"], "RUN_TYPE", "GEO_OPT")

        if calculation.debug:
            self.debug_run(calculation, pars)
        else:
            self.relax(calculation, pars)
            
        return True

    def debug_run(self, calculation, pars):
        log_filename = self.wrkdir+"/"+calculation.directory+"/"+calculation.filename+'.log'
        traj_filename = self.wrkdir+"/"+calculation.directory+"/"+calculation.filename+'.traj'
        output_filename = self.wrkdir+"/"+calculation.directory+"/"+calculation.filename+'.out'
        input_filename = self.wrkdir+"/"+calculation.directory+"/"+calculation.filename+'.inp'
        
        cp2ktools.atoms2cpk(pars, calculation.start_atoms, velocities=False, poisson="mt")
        lines=cp2ktools.getlines(pars)
        # write cp2k into an input file..
        cp2ktools.writefile(input_filename, lines)
        print "DEBUG: Skipping actual calculation with CP2K. Writing a fake file with random energy to \"%s\"" % output_filename   
        calculation.write_fake_cp2k_out_file()
        if calculation.calculation_type == 'GEOMETRY_OPTIMIZATION':
            ase.io.write(traj_filename, calculation.get_initial_structure())
            calculation.energy = (0.1)**2
            calculation.write_fake_ase_log_file()
        if self.calculate_charge_density:
            from ..misc.cube import write_dummy_cube
            write_dummy_cube(calculation.get_initial_structure(), calculation.directory+"/"+calculation.filename+".cube")

    def relax(self, calculation, pars):
        """
            Call when calculation type specific parameters are set
        """
        # next, insert info about velocities and atomic coordinates to the parameter dictionary:
        psolver="mt"
        usevelocities=False # use velocities present in the traj file or not

        # get the file names
        filename = self.wrkdir+"/"+calculation.directory+"/"+calculation.filename

        log_filename = filename+'.log'
        output_filename = filename+'.out'
        input_filename = filename+'.inp'
        
        # do not add postfix to traj and pos path, as we want to get the geometry optimization's result as the starting point
        pos_filename = self.wrkdir+"/"+calculation.directory+"/cp2k-pos-1.xyz"
        traj_path = self.wrkdir+"/"+calculation.directory+"/"+calculation.filename+".traj"
        
            
        # Because we are this far we know that calculation has been initialized (.inp file exists), is not locked
        # but has not finished. However, we do not know if any steps of the calculation have been made.
        # This can be checked with 'calculation.started'. If this is the case
        # we continue from the last executed step
        if calculation.calculation_type == 'GEOMETRY_OPTIMIZATION' and calculation.started():
            cont = True
            if rank == 0:
                calculation.write_cp2k_traj_file(self.wrkdir)
            # cont is just a dummy parameter to assure that rank 0 is finished writing the traj file, before others try
            # to read it
            if size != 1:
                cont = comm.bcast(cont, root = 0)
                
        atoms = calculation.get_initial_structure()
        # if calculation is not finished, but has started let's continue with the previous structure
        if os.path.exists(traj_path):
            atoms = ase.io.read(traj_path)
                
        cp2ktools.atoms2cpk(pars, atoms, velocities=usevelocities, poisson=psolver)

        # write starting geometry to a file, this is required to have the constraints etc stored
        if calculation.calculation_type == 'GEOMETRY_OPTIMIZATION' and not os.path.exists(calculation.directory+"/"+calculation.filename+"_start.traj"):
            ase.io.write(self.wrkdir+"/"+calculation.directory+"/"+calculation.filename+"_start.traj", calculation.start_atoms, format = "traj")
            
        
        # write cp2k input file
        lines=cp2ktools.getlines(pars)
        cp2ktools.writefile(input_filename, lines)
            
        # launch execution
        cp2ktools.runcp2k(self.wrkdir+"/"+calculation.directory, calculation.filename+".inp", calculation.filename+".out")

        # read energy
        energy = mycalculators.read_cp2k_energy(output_filename)
        calculation.energy = energy
            
        # write traj from pos-xyz
        if calculation.calculation_type != 'ENERGY' and rank == 0:
            calculation.write_cp2k_traj_file(self.wrkdir)
        calculation.write_fake_ase_log_file()
            
        #cp2ktools.atoms2cpk(pars, result_atoms, velocities=usevelocities, poisson=psolver)
        
        

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

    def initialize_calculator_parameters(self, calculation):
        if calculation.method == 'dft':
            pars = copy.deepcopy(cp2ktools.def_kari)

            # Set XC functional and consistent basis sets:
            cp2ktools.fixpars(pars, self.wrkdir, pseudo=calculation.xc,basis=self.get_cp2k_basis_set(calculation.basis_set))

            # Let's use Grimme D3 dispersion correction:
            cp2ktools.toggle_vdw(pars, self.wrkdir, xcf=calculation.xc)
              
            # Let's set charge of the system
            if (calculation.charge != 0):
                cp2ktools.set_input(pars,["FORCE_EVAL","DFT"],"CHARGE", str(calculation.charge))
              
            # choose XC functional
            cp2ktools.stuff(pars, ["FORCE_EVAL","DFT","XC","XC_FUNCTIONAL "+calculation.xc], "inp", [])

            # set reference cell
            if calculation.reference_cell is not None:
                cp2ktools.bigcellref(pars, calculation.reference_cell)
                if calculation.debug:
                    print "Setting reference lattice constants", calculation.reference_cell

        if (self.metals):
            cp2ktools.metals(pars)
        if self.mulliken:
            cp2ktools.mulliken(pars, filename=filename + "_mulliken.dat")
            relaxation_type = "aserelax"
        return pars

    def energy_calculation(self, calculation):
        # So, first, let's get some initial DFT parameters:
        pars = self.initialize_calculator_parameters(calculation)
          
        # the type of calculation, when calculating dipole moment, do only one scf loop
        cp2ktools.set_input(pars, ["GLOBAL"], "RUN_TYPE", "ENERGY")

        # do dipole moment/charge or electron density/wannier center/electrostatic_potential analysis if required
        if self.calculate_dipole_moment:
            cp2ktools.total_dipole_moment(pars, self.wrkdir+"/"+calculation.directory+"/"+calculation.filename)
        if self.calculate_molecular_dipole_moments:
            cp2ktools.molecular_dipole_moments(pars, self.wrkdir+"/"+calculation.directory+"/"+calculation.filename)
        if self.calculate_charge_density:
            cp2ktools.total_density(pars, self.wrkdir+"/"+calculation.directory+"/"+calculation.filename)
        if self.calculate_electron_density:
            cp2ktools.electron_density(pars, self.wrkdir+"/"+calculation.directory+"/"+calculation.filename)
        if self.calculate_wannier_centers:
            cp2ktools.wannier_centers(pars, self.wrkdir+"/"+calculation.directory+"/"+calculation.filename)
        if self.calculate_electrostatic_potential:
            cp2ktools.electrostatic_potential(pars, self.wrkdir+"/"+calculation.directory+"/"+calculation.filename)

        # if debug, do a debug run, else do the real thing
        if calculation.debug:
            self.debug_run(calculation, pars)
        else:
            self.relax(calculation, pars)
            
        return True

    def initialize_calculation(self, start_atoms, program, calculator, number, folder, number_of_relaxation, method, xc, lattice_constant, basis_set, charge = 0, reference_cell = None, calculation_type = 'GEOMETRY OPTIMIZATION', ice_structure_calculation = True, additional_atoms_calculation = False, directory = None, filename = None, debug = True):
        return CP2KCalculation(start_atoms, program, self, number, folder, number_of_relaxation, method, xc, lattice_constant, basis_set, charge = charge, reference_cell = reference_cell, calculation_type = calculation_type, ice_structure_calculation = ice_structure_calculation, additional_atoms_calculation = additional_atoms_calculation, directory = directory, filename = filename, debug = debug)
        

class CP2KCalculation(Calculation):
    def __init__(self, start_atoms, program, calculator, number, folder, number_of_relaxation, method, xc, lattice_constant, basis_set, charge = 0, reference_cell = None, calculation_type = 'GEOMETRY_OPTIMIZATION', ice_structure_calculation = True, additional_atoms_calculation = False, directory = None, filename = None, debug = True):
        Calculation.__init__(self, start_atoms, program, calculator, number, folder, number_of_relaxation, method, xc, lattice_constant, basis_set, charge = charge, reference_cell = reference_cell, calculation_type = calculation_type, ice_structure_calculation = ice_structure_calculation, additional_atoms_calculation = additional_atoms_calculation, directory = directory, filename = filename, debug = debug)
        
    def write_cp2k_traj_file(self, wrkdir):
        pos_path = wrkdir+"/"+self.directory+"/"+"cp2k-pos-1.xyz"
        if os.path.exists(pos_path):
            traj_path = wrkdir+"/"+self.directory+"/"+self.filename+".traj"
            start_atoms = self.get_initial_structure()
            index = 0
            if not os.path.exists(traj_path):
                result_atoms = ase.io.read(pos_path, index) 
                result_atoms.set_cell(start_atoms.get_cell())
                result_atoms.set_pbc(start_atoms.get_pbc())
                if start_atoms.constraints is not None:
                    for constraint in start_atoms.constraints:
                        result_atoms.set_constraint(constraint)
                traj = PickleTrajectory(traj_path, mode="w", atoms = result_atoms)
                traj.write()
                index = 1
            traj = PickleTrajectory(traj_path, mode="a")
            while True:
                try:
                    result_atoms = ase.io.read(pos_path, index) 
                    result_atoms.set_cell(start_atoms.get_cell())
                    result_atoms.set_pbc(start_atoms.get_pbc())
                    if start_atoms.constraints is not None:
                        for constraint in start_atoms.constraints:
                            result_atoms.set_constraint(constraint)
                    traj.write(result_atoms)
                    index += 1
                except:
                    break
            os.remove(pos_path)
        else:
            if self.debug:
                print "%s does not exist. Not writing anything." % pos_path

    def finished(self):
        out_filename = self.directory + "/" + self.filename + ".out"
        log_filename = self.directory + "/" + self.filename + ".log"
        if self.debug:
            print "Checking if CP2K-calculation with filename \"%s\" finished" % out_filename
        if os.path.exists(out_filename):
            if self.debug:
                print "  -File exists"
            
            if self.calculation_type == 'GEOMETRY_OPTIMIZATION' and os.path.exists(log_filename):
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
            elif self.calculation_type == 'ENERGY':
                text = open(out_filename, 'r').read()
                lines = iter(text.split('\n'))
                for line in lines:
                    if line.startswith('  *** SCF run converged in'):
                        if self.debug:
                            print "  -%s was finished" % self
                        return True
                    else:
                        if self.debug:
                            print "  -%s was not finished" % self
        else:
            if self.debug:
                print "  -File does not exist"
            
        return False

    def initialized(self):
        in_filename = self.directory + "/" + self.filename + ".inp"
        return os.path.exists(in_filename)

    def started(self): 
        """
            Implementation of the 'Calculation.started()' method
              - determines if the cp2k calculation is started by checking if the
                'in_filename' and ('log_filename', or 'pos_filename') exist
        """
        in_filename = self.directory + "/" + self.filename + ".inp"
        log_filename = self.directory + "/" + self.filename + ".log"
        
        pos_filename = self.directory + "/cp2k-pos-1.xyz"  
        if self.debug:
            print "-----------------------------------"
            print "Checking if CP2K-calculation with filename \"%s\" started" % in_filename
        if os.path.exists(in_filename):
            if self.debug:
                print "  -File exists"
            
            if self.calculation_type == 'GEOMETRY_OPTIMIZATION' and os.path.exists(log_filename) or os.path.exists(pos_filename):
                if self.debug:
                    print "  -Calculation was started"
                    print "-----------------------------------"
                return True

            out_filename = self.directory + "/" + self.filename + ".out"
            if self.calculation_type == 'ENERGY' and os.path.exists(out_filename):
                if self.debug:
                    print "  -Calculation was started"
                    print "-----------------------------------"
                return True
        else:
            if self.debug:
                print "  -File does not exist"
                print "-----------------------------------"
            
        return False
    
    def get_energy(self):
        if self.energy is None:
            output_filename = self.directory + "/" + self.filename + ".out"
            mycalculators.read_cp2k_energy(output_filename)
        return self.energy
            
    
    def locked(self):
        out_filename = self.directory + "/" + self.filename + ".out"
        if os.path.exists(out_filename):
            seconds_from_modification = time() - os.path.getmtime(out_filename)
            return seconds_from_modification < 60.0
        else:
            in_filename = self.directory + "/" + self.filename + ".inp"
            if os.path.exists(in_filename):
                seconds_from_modification = time() - os.path.getmtime(in_filename)
                return seconds_from_modification < 60.0
            else:
                return False

    def write_fake_cp2k_out_file(self):
        file_name = self.directory + "/" + self.filename + ".out"
        if self.calculation_type == 'ENERGY':
            contents = "  *** SCF run converged in"
        else:
            contents = " ***                    GEOMETRY OPTIMIZATION COMPLETED"
        fo = open(file_name, "w")
        fo.write(contents)
        fo.close()

    def write_fake_ase_log_file(self):
        file_name = self.directory + "/" + self.filename + ".log"
        contents = "BFGSLineSearch:   0  00:00:00  %f      %f" % (float(self.energy), self.fmax - 0.001)
        fo = open(file_name, "w")
        fo.write(contents)
        fo.close()
