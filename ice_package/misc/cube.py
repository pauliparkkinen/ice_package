import numpy as np

def read_cube(filename):
    cube_file = open(filename, "r")
    origin_position = np.empty((3))
    matrix_vectors = np.empty((3, 3))
    matrix_shape = np.empty(3, dtype=int)
    flat_matrix = None
    labels = ["", ""]
    z = 0
    x = 0
    y = 0
    for i, line in enumerate(cube_file):
        if i < 2:
            labels[i] = line[:-1]
        elif i == 2:
            words = line.split()
            number_of_atoms = int(words[0])
            origin_position[0] = float(words[1]) 
            origin_position[1] = float(words[2])
            origin_position[2] = float(words[3])
            atom_positions = np.empty((number_of_atoms, 3))
            atomic_numbers = np.empty(number_of_atoms, dtype=int)
        elif i < 6:
            words = line.split()
            matrix_shape[i-3] = int(words[0])
            matrix_vectors[i-3, 0] = float(words[1])
            matrix_vectors[i-3, 1] = float(words[2])
            matrix_vectors[i-3, 2] = float(words[3])
            if i == 5:
                density_matrix = np.empty(matrix_shape)
        elif i < number_of_atoms + 6:
            words = line.split()
            atomic_numbers[i-6] = int(words[0])
            atom_positions[i-6, 0] = float(words[2]) 
            atom_positions[i-6, 1] = float(words[3]) 
            atom_positions[i-6, 2] = float(words[4]) 
        else:
            words = line.split()
            for word in words:
                density_matrix[x, y, z] = float(word)
                if z < matrix_shape[2] -1:
                    z += 1
                else:
                    z = 0
                    if y < matrix_shape[1] -1:
                        y += 1
                    else:
                        y = 0
                        if x < matrix_shape[0] -1:
                            x += 1
                
    cube_file.close()
    return Cube(origin_position, density_matrix, matrix_vectors, atom_positions, atomic_numbers, labels, filename)

def write_dummy_cube(atoms, filename):
    density_matrix = np.zeros((95, 83, 462))
    matrix_vectors = np.zeros((3,3))
    matrix_vectors[0, 0] = 0.369623
    matrix_vectors[1, 1] = 0.366664
    matrix_vectors[2, 2] = 0.373319
    
    cu = Cube(np.zeros(3), density_matrix, matrix_vectors, atoms.get_positions(), atoms.get_atomic_numbers(), ['DUMMY', 'CUBE FILE'], filename)
    cu.write_to_file()

class Cube:
    def __init__(self, origin_position, density_matrix, matrix_vectors, atom_positions, atomic_numbers, labels, filename):
        self.origin_position = origin_position        
        self.density_matrix = density_matrix
        if sum(matrix_vectors[0]) < 0:
            self.matrix_vectors = matrix_vectors * (-1)
            self.atom_positions = atom_positions
        else:
            self.matrix_vectors = matrix_vectors * 0.529 # Bohr radius to Angstrom
            self.atom_positions = atom_positions * 0.529
        
        self.atomic_numbers = atomic_numbers
        self.labels = labels
        self.filename = filename
    
    def reduce_density(self, cube):
        if cube.density_matrix.shape == self.density_matrix.shape:
            self.density_matrix -= cube.density_matrix
        else:
            raise Exception("Cube file densities must have similar shape")

    def write_to_file(self, filename = None):
        if filename is None:
            filename = self.filename
        lines = []
        # add labels to rows 1 and 2
        lines.extend(self.labels)
        # add number of atoms and origin position to row 3
        lines.append("%i %.6f %.6f %.6f" % ( self.atomic_numbers.shape[0], self.origin_position[0], self.origin_position[1], self.origin_position[2] ) )
        # change from Angstrom to Bohr
        matrix_vectors = self.matrix_vectors / 0.529
        density_matrix = self.density_matrix
        # add density matrix shape and the vectors defining the voctels
        lines.append("%i %.6f %.6f %.6f" % (density_matrix.shape[0], matrix_vectors[0, 0], matrix_vectors[0, 1], matrix_vectors[0, 2] ) )
        lines.append("%i %.6f %.6f %.6f" % (density_matrix.shape[1], matrix_vectors[1, 0], matrix_vectors[1, 1], matrix_vectors[1, 2] ) )
        lines.append("%i %.6f %.6f %.6f" % (density_matrix.shape[2], matrix_vectors[2, 0], matrix_vectors[2, 1], matrix_vectors[2, 2] ) )
        # add atomic numbers and positions of atoms change from Angstrom to Bohr
        atom_positions = self.atom_positions / 0.529
        for i, atomic_number in enumerate(self.atomic_numbers):
            lines.append("%i 0.000000 %.6f %.6f %.6f" % (atomic_number, atom_positions[i, 0], atom_positions[i, 1], atom_positions[i, 2] ) )
        
        cube_file = open(filename, 'w')
        # write initial lines
        for item in lines:
            cube_file.write("%s\n" % item)

        # write charge density to file
        counter = 0
        line = ""
        density_matrix = self.density_matrix
        shape = self.density_matrix.shape
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    line += "%.7g " % density_matrix[x, y, z]
                    if counter == 5:
                        cube_file.write("%s\n" % line)
                        line = ""
                        counter = 0
                    else:
                        counter += 1

        # finally close the file
        cube_file.close()

    def get_dipole_moment(self):
        density_matrix = self.density_matrix
        shape = self.density_matrix.shape
        x_vector = self.matrix_vectors[0]
        y_vector = self.matrix_vectors[1] 
        z_vector = self.matrix_vectors[2] 
        total = np.zeros(3)
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    position = x*x_vector + y*y_vector + z*z_vector
                    total += position * density_matrix[x, y, z]
        volume = x_vector[0] * shape[0] * y_vector[1] * shape[1] * z_vector[2] * shape[2]
        return total / (volume * 0.20819434) # conversion from e*Ang to D

    

def __main__():
    import tkFileDialog
    filename = tkFileDialog.askopenfilename()
    return read_cube(filename)
        
        
        
