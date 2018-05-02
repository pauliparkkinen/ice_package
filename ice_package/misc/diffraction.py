import numpy as np
import ase, math

def powder_diffraction(atom_positions, atomic_numbers, cell, reciprocal_cell = None, wavelength = 1.476):
    """
        "Constructive interference will occur provided that the change in wave vector K = k' - k 
        is a vector of the reciprocal lattice."
        
        Thus, there will only be peaks with k-values that are multiples of reciprocal lattice vectors.

        In powder reflection each reciprocal lattice vector of length less than 2k generates a cone of
        scattered radiation at an angle phi to the forward direction where the length of the change in wave
        vector |K| = 2 |k| sin (1/2 phi)

        The relationship between |k| and the wavelength lambda is 
        |k| = 2 pi / lambda

        thus we can derive the peak positions with respect to phi and to K
              |K| = 4 pi / lambda (sin 1/2 phi)
           => phi = 2 asin( (lambda |K|) / (4 pi) )

        'wavelength' and 'atom_positions' are represented in angstroms
    """
    # initialize reciprocal cell if needed
    if reciprocal_cell is None:
        reciprocal_cell = calculate_reciprocal_cell(atom_positions, atomic_numbers, cell)
    test_reciprocal_cell(cell, reciprocal_cell)    

    peak_angles = {}
    peak_intensities = {}
    # get the incident vector length
    k_length = 2.0 * math.pi / wavelength

    # go through all reciprocal lattice vectors K
    # (i.e., vectors that have independent i, j, and k multipliers) and
    # get corresponding phi-positions  
    for i in range(0, 6):
        for j in range(0, 6):
            for k in range(0, 6):
                length_not_zero = i != 0 or j != 0 or k != 0
                if length_not_zero:
                    # get the actual vector
                    K = reciprocal_cell[0] * i + reciprocal_cell[1] * j + reciprocal_cell[2] * k
                    # get the vector length
                    K_length = math.sqrt(K[0]*K[0] + K[1]*K[1] + K[2]*K[2])
                    
                    if K_length <  2* k_length:  
                        phi = 2.0 * math.asin( (wavelength * K_length) / (4.0 * math.pi) )
                        phi = round(phi, 4)
                        # get the intensities 
                        if i not in peak_intensities:
                            peak_intensities[i] = {}
                            peak_angles[i] = {}
                        if j not in peak_intensities[i]:
                            peak_intensities[i][j] = {}
                            peak_angles[i][j] = {}
                        
                        peak_intensities[i][j][k] = structure_factor_square(atom_positions, atomic_numbers, K)
                        peak_angles[i][j][k] = phi
    return peak_intensities, peak_angles

def print_peak_information(peak_intensities, peak_angles):
    print "Phi [deg]   (Miller Index)   Intensity"
    for i in peak_angles:
        for j in peak_angles[i]:
            for k in peak_angles[i][j]:
                phi = (180.0 / math.pi) * peak_angles[i][j][k]
                if peak_intensities[i][j][k] > 0.00002 or ((i + j) % 2 != 0 and peak_intensities[i][j][k] > 0.000005) or phi > 40.0 and phi < 42.0:
                    print "%-14.10g (%2i, %2i, %2i)     %-20.10g " % (phi, i, j, k, peak_intensities[i][j][k])

def plot_diffraction_spectrum(peak_intensities, peak_angles):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    x = np.arange(0, 180, 0.1)
    y = np.zeros(x.shape[0])
    ticks = []
    tick_labels = []
    for i in peak_angles:
        for j in peak_angles[i]:
            for k in peak_angles[i][j]:
                phi = round((180.0 / math.pi) * peak_angles[i][j][k], 1)
                index = phi / 0.1 
                y[index] += peak_intensities[i][j][k]
                if peak_intensities[i][j][k] > 0.00001 or ((i + j) % 2 != 0) :
                    ticks.append(phi)
                    tick_labels.append("(%i, %i, %i)" % (i, j, k))
    ax1.plot(x, y, '-')
    ax1.set_xlabel("$\Theta$ [Deg]")
    
    ax2 = ax1.twiny()
    ax2.set_xlabel("Miller Plane")
    ax2.set_xlim(0, 180)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(tick_labels, rotation=90, fontsize=10)
    plt.show()
    
    
def get_atomic_form_factor(atomic_number, K):
    """
        Returns the bound coherent neutron scattering lengths of different atoms. 
        The units are scaled to angstroms
    """
    if atomic_number == 1:
        return 3.7409 * 0.0001
    elif atomic_number == 8:
        return 5.805 * 0.0001
    return 1.0

def structure_factor_square(atom_positions, atomic_numbers, K):
    """
        atom_positions : basis
    """
    n = len(atom_positions)
    result = complex(0.0, 0.0)
    for i in range(n):
        atomic_form_factor = get_atomic_form_factor(atomic_numbers[i], K)
        angle = np.dot(K, atom_positions[i])
        result += atomic_form_factor * np.exp(complex(0, angle))
    
    return np.power(np.abs(result), 2)
        
       
    

def powder_diffraction_for_atoms(atoms):
    return powder_diffraction(atoms.get_positions(), atoms.get_atomic_numbers(), atoms.get_cell(), 2*math.pi * atoms.get_reciprocal_cell())

def calculate_reciprocal_cell(atom_positions, atomic_numbers, cell):
    atoms = ase.Atoms(positions = atom_positions, numbers = atomic_numbers, cell = cell, pbc = True)
    return 2*math.pi * atoms.get_reciprocal_cell()



def test_reciprocal_cell(cell, reciprocal_cell):
    result = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            result[i, j] = np.dot(cell[i], reciprocal_cell[j])
    return True
