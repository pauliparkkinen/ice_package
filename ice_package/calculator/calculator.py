
class Calculator:
    def initialize_calculation(self, start_atoms, program, calculator, number, folder, number_of_relaxation, method, xc, lattice_constant, basis_set, charge = 0, reference_cell = None, calculation_type = 'GEOMETRY OPTIMIZATION', ice_structure_calculation = True, additional_atoms_calculation = False, debug = True):
        pass

    def calculate(self, calculation):
        if calculation.calculation_type == 'GEOMETRY_OPTIMIZATION':
            return self.geometry_optimization(calculation)
        elif calculation.calculation_type == 'ENERGY':
            return self.energy_calculation(calculation)
        else:
            raise NotImplemented

    def geometry_optimization(self, calculation):
        pass
    
    def energy_calculation(self, calculation):
        pass


def initialize_calculator(program):
    if program == 'cp2k':
        from cp2k import CP2K
        return CP2K()
    elif program == 'GPAW':
        from gpaw_calculator import GPAW
        return GPAW()
    elif program == 'TurboMole':
        from turbomole import TurboMole
        return TurboMole()





