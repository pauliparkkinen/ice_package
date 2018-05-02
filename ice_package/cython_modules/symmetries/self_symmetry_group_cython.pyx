cimport numpy as np
import numpy as np
from cpython cimport bool

cdef class SelfSymmetryGroup:
    def __init__(self, list symmetry_operations = [], np.int8_t[::1] water_orientation = None, SelfSymmetryGroup parent_self_symmetry_group = None, bool finalized = False, list child_self_symmetry_groups = None, np.ndarray[np.int8_t, ndim=2] water_orientations = None):
        self.symmetry_operations = symmetry_operations
        if water_orientation is None: 
            if water_orientations is None:
                self.water_orientations = None
            else:
                self.water_orientations = water_orientations
        else:  
            if water_orientations is None:
                self.water_orientations = np.empty((1, water_orientation.shape[0]), dtype=np.int8)
                self.water_orientations[0] = water_orientation
            else:
                self.water_orientations = water_orientations
            
        if child_self_symmetry_groups is None or len(child_self_symmetry_groups) == 0:
            self.child_self_symmetry_groups = []
        else:
            self.child_self_symmetry_groups = child_self_symmetry_groups
        self.finalized = finalized
        if parent_self_symmetry_group is None:
            self.parent_self_symmetry_group = None
        else:
            self.parent_self_symmetry_group = parent_self_symmetry_group
        
    """def __reduce__(self):
        state = {}
        state['symmetry_operations'] = self.symmetry_operations
        cdef SelfSymmetryGroup ssg = self.parent_self_symmetry_group
        state['parent_self_symmetry_group'] = ssg
        state['finalized'] = self.finalized
        state['child_self_symmetry_groups'] = self.child_self_symmetry_groups
        state['water_orientations'] = self.water_orientations
        return (SelfSymmetryGroup, # constructor to be called
        (self.symmetry_operations, None, self.parent_self_symmetry_group,  self.finalized, self.child_self_symmetry_groups, self.water_orientations), # arguments to constructor 
         state, None , None)"""
    
    def __setstate__(self, state):
        self.symmetry_operations = state['symmetry_operations']
        self.parent_self_symmetry_group = state['parent_self_symmetry_group']
        self.finalized = state['finalized']
        self.child_self_symmetry_groups = state['child_self_symmetry_groups']
        self.water_orientations = state['water_orientations']

    def __getstate__(self):
        state = {}
        state['symmetry_operations'] = self.symmetry_operations
        state['parent_self_symmetry_group'] = self.parent_self_symmetry_group
        state['finalized'] = self.finalized
        state['child_self_symmetry_groups'] = self.child_self_symmetry_groups
        state['water_orientations'] = self.water_orientations
        return state
    

    cdef np.ndarray[np.int8_t, ndim=1] get_all_water_orientations(self):
        cdef np.ndarray[np.int8_t, ndim=2] result = np.empty((1, self.water_orientations.shape[1]))
        cdef list child_self_symmetry_groups = self.child_self_symmetry_groups
        cdef int i, N = len(child_self_symmetry_groups)
        for i from 0 <= i < N:
            child_self_symmetry_group = child_self_symmetry_groups[i]
            result = np.vstack((result, child_self_symmetry_group.get_all_water_orientations()))
        result = np.vstack((result, self.water_orientations))
        return result
        
    cdef list get_all_active_symmetry_operations(self):
        cdef SelfSymmetryGroup parent = self.get_active_parent_self_symmetry_group()
        cdef list result = []
        while parent is not None:
            result.extend(parent.symmetry_operations)
            parent = parent.get_active_parent_self_symmetry_group()
        print "The group itself has %i symmetry operations" % len(self.symmetry_operations)
        result.extend(self.symmetry_operations)
        print "Finally %i active symmetry operations" % len(result)
        raw_input()
        return result
        
    cdef list get_active_parent_symmetry_operations(self, list possible_symmetry_operations):
        cdef SelfSymmetryGroup parent = self.get_active_parent_self_symmetry_group()
        cdef list result = []
        cdef list tested, leftover
        while parent is not None:
            tested, leftover = parent.get_tested_and_leftover_symmetry_operations(possible_symmetry_operations)
            result.extend(tested)
            parent = parent.get_active_parent_self_symmetry_group()
        
        return result
        
    cdef SelfSymmetryGroup get_active_parent_self_symmetry_group(self):
        cdef SelfSymmetryGroup parent = self.parent_self_symmetry_group
        while parent is not None and parent.finalized:
            parent = parent.parent_self_symmetry_group
        return parent
        
    cdef tuple get_tested_and_leftover_symmetry_operations(self, list possible_symmetry_operations):
        """
            Leftover symmetry operations contains the operations that are not
            tested at the current point of execution, the tested symmetry_operations will
            contain the ones that are tested.
        """
        cdef list tested_symmetry_operations  = []
        cdef list leftover_symmetry_operations = []
        cdef list active_symmetry_operations = self.symmetry_operations
        cdef int i, M = len(active_symmetry_operations)
        for j from 0 <= j < M:
            symmetry_operation = active_symmetry_operations[j]
            #print symmetry_operation
            if symmetry_operation in possible_symmetry_operations:
                tested_symmetry_operations.append(symmetry_operation)
                #print symmetry_operation
            else:
                leftover_symmetry_operations.append(symmetry_operation)
        return tested_symmetry_operations, leftover_symmetry_operations

    cdef list get_all_child_self_symmetry_groups(self):
        cdef list result = []
        cdef list child_self_symmetry_groups = self.child_self_symmetry_groups
        cdef int i, N = len(child_self_symmetry_groups)
        cdef SelfSymmetryGroup child_self_symmetry_group
        for i from 0 <= i < N:
            child_self_symmetry_group = child_self_symmetry_groups[i]
            result.extend(child_self_symmetry_group.get_all_child_self_symmetry_groups())
            if not child_self_symmetry_group.finalized:
                result.append(child_self_symmetry_group)
        return result
        
