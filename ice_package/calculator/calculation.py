from calculator import initialize_calculator
from ..energy_commons import EnergyCommons
import ase, os
from ..misc import cube as cube

class Calculation(EnergyCommons):
    def __init__(self, start_atoms, program, calculator, number, folder, number_of_relaxation, method, xc, lattice_constant, basis_set, charge = 0, reference_cell = None, calculation_type = 'GEOMETRY_OPTIMIZATION', ice_structure_calculation = True, additional_atoms_calculation = False, directory = None, filename = None, debug = False):
        EnergyCommons.__init__(self, folder = folder, number_of_relaxation = number_of_relaxation,  program = program, method = method, xc = xc, basis_set = basis_set, fmax = 0.05, file_type = "traj", original_program = None, debug = debug) 
        self.reference_cell = reference_cell
        self.charge = charge
        self.calculator = calculator
        self.number = number
        self.start_atoms = start_atoms
        self.lattice_constant = lattice_constant
        self.ice_structure_calculation = ice_structure_calculation
        self.additional_atoms_calculation = additional_atoms_calculation

        self.directory = directory
        self.filename = filename
        if self.directory is None or self.filename is None:
            directory, filename = self.get_directory_and_filename(number = number, number_of_relaxation = number_of_relaxation, loading = False, xc = xc, program = program, method = method, folder = self.folder, lattice_constant = lattice_constant, basis_set = basis_set, additional_atoms_calculation = additional_atoms_calculation, ice_structure_calculation = ice_structure_calculation) 
            if self.directory is None:
                self.directory = directory
            if self.filename is None:
                self.filename = filename
    
        self.energy = None
        self.calculation_type = calculation_type

    def finished(self):
        raise NotImplemented
        
    def locked(self):
        return False
    
    def initialized(self):
        raise NotImplemented        
    
    def started(self):
        raise NotImplemented

    def get_energy(self):
        return self.energy

    def start(self):
        if self.initialized():
            if self.locked():
                print "'%s' is locked, cannot restart" % self
                return False
            elif self.finished():
                print self.directory, self.filename
                print "'%s' is already finished" % self
                return True
            elif self.started():
                print "Restarting '%s'" % self
            else:
                print "Starting '%s'" % self
        else:
            print "Starting '%s'" % self
        return self.calculator.calculate(self)

    def get_initial_structure(self):
        if self.start_atoms is not None:
            return self.start_atoms
        else:
            traj_path = directory + "/" + filename + "_start.traj"
            if os.path.exists(traj_path):
                self.start_atoms = ase.io.read(traj_path)
                return self.start_atoms
            else:
                return None

    def get_final_structure(self):
        traj_path = self.directory + "/" + self.filename + ".traj"
        if os.path.exists(traj_path):
            return ase.io.read(traj_path)
        return None

    def read_charge_density(self):
        cube_path = self.directory + "/" + self.filename + ".cube"
        if os.path.exists(cube_path):
            return cube.read_cube(cube_path)
        else:
            return None
    
    def delete_files(self):
        """
            Deletes all the files related to the calculation
        """ 
        if os.path.exists(self.directory):
            filenames = os.listdir(self.directory)
            for filename in filenames:
                if filename.startswith(self.filename + "."):
                    os.remove(self.directory + "/" + filename)

    def __str__(self):
        result = "%s, %s, %s calculation for" % (self.calculator, self.method, self.calculation_type)
    
        if self.number is None and self.additional_atoms_calculation:
            result += " additional atoms"
        elif self.ice_structure_calculation:
            result += " bare ice structure %i" % self.number
        else:
            result += " structure %i" % self.number
        if self.number_of_relaxation > 0:
            result += ", restart %i" %  self.number_of_relaxation
        return result

   
