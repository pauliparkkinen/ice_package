
cdef class SelfSymmetryGroup:
    def __init__(self, list symmetry_operations, np.ndarray[DTYPE2_t, ndim=1] water_orientation):
        self.symmetry_operations = symmetry_operations
        self.water_orientations = np.empty((1, water_orientation.shape[0]))
        self.water_orientations[0] = water_orientation
