
def calculate_core_electrostatic_potential_energy_for_atoms(atoms):
    return calculate_core_electrostatic_potential_energy(atoms.get_atomic_numbers(), atoms.get_positions())

cdef np.float_t calculate_core_electrostatic_potential_energy(np.int_t[::1] atomic_numbers, np.float_t[:, ::1] atom_positions):
    """
        calculates the electrostatic potential energy between atomic cores, 
        assuming that the cores are point charges
        
        returns the energy in Joules
    """
    cdef int i, j, N = atomic_numbers.shape[0]
    cdef np.float_t result = 0.0, distance, coulombs_constant = 8.987551787e9, elementary_charge = 1.602176565e-19, one_per_angstrom = 1e10
    for i in range(N):
        for j in range(N):
            if i < j: # calculate only once
                distance = euler_distance(atom_positions[i], atom_positions[j])
                result += atomic_numbers[i] * atomic_numbers[j] / distance
            
    
    # convert from current units e^2 / Ang to SI units 
    result *= coulombs_constant * elementary_charge * elementary_charge * one_per_angstrom
    return result
    

cdef inline np.float_t euler_distance(np.float_t[::1] a, np.float_t[::1] b):
    cdef np.float_t result = sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2) + pow(a[2] - b[2], 2))
    return result
