#cython: boundscheck=False
#cython: wraparound=False
# cython: infer_types=True
#cython: none_check=False
#cython: c_line_in_traceback=False
from libc cimport math
import numpy as np
cimport numpy as np
np.import_array()
from cpython cimport bool
from .graph_invariants_cython cimport Invariant
from .water_algorithm_cython cimport size, rank
  
cdef class ResultGroup:
    """
        Class that represents a single group that contains either other groups or proton configurations.
        Can have infinitely many child groups, i.e., 'subgroups'. Each group contains results with the
        same 'value' for the 'invariant' (Invariant object).
        
        Grouping with result groups forms a tree that helps to find the results that have to be compared against
        each other.
    """
    def __cinit__(self, list invariant_list, signed char value, np.ndarray[np.int8_t, ndim=2] wos = None, subgroups = None, subgroup_count = 0):
        """
            list invariant_list:  contains Invariants that are used for sorting, the first one used in 
            this group to sort graphs
            float value -  value of the previous invariant that makes the group unique 
        """
        self.value = value
        self.invariant_list = invariant_list
        self.invariant = self.invariant_list[0]
        self.invariant_list_length = len(invariant_list)
        self.subgroup_count = subgroup_count
        self.subgroups = subgroups
        self.results = []
        self.bvs = None
        self.wos = wos
        if subgroups is None:
            self.subgroups = []
        else:
            self.subgroups = subgroups
        

    #def __cinit__(self):
    #    self.value = 0
    #    self.invariant_list = []
    #    self.invariant = None
    #    self.subgroups = []
    #    self.results = []
    #    self.bvs = None
    #    self.wos = []"""

    def __reduce__(self):
        """
            Needed for pickle because of mpi4py
        """
        return (ResultGroup, # constructor to be called
        (self.invariant_list, self.value, self.wos, self.subgroups, self.subgroup_count), # arguments to constructor 
         None , None , None)
   
         
    def __richcmp__(self, ResultGroup other_group, i):
        """ Works only for first level invariant, use with caution """
        if i == 2:
            self.is_equal_with(other_group)
    

    cdef bool is_equal_with(self, ResultGroup other_group):
        return other_group is not None and other_group.wos is not None and self.wos is not None and len(self.invariant_list)  == len(other_group.invariant_list) and self.value == other_group.value

    cdef void merge(self, ResultGroup other_group):
        self.wos = np.vstack((self.wos, other_group.wos))
        #self.wos.extend(other_group.wos)
        other_group.wos = None 
            
    
    cdef inline bool belongs_to_group(self, signed char value):
        return self.value == value

    cdef public ResultGroup try_subgroups(self, np.ndarray[np.int8_t, ndim=2] bond_variables, signed char destination_level):
        return self._try_subgroups(bond_variables, destination_level)
     
    cdef public ResultGroup _try_subgroups(self, np.int8_t[:, ::1] bond_variables, signed char destination_level):
        """
            Try to which subgroup a structure with 'bond_variables' belongs to.
            The tree is browsed downwards until 'destination_level' is reached, on which the 'self' group is returned
        """
        cdef np.int8_t value = self.invariant._get_value(bond_variables)
        cdef ResultGroup subgroup 
        cdef list subgroups = self.subgroups
        cdef np.int8_t N = self.subgroup_count
        cdef list new_invariant_list
        cdef list invariant_list = self.invariant_list
        cdef np.int8_t subgroup_value
   
        # reduce the destination level if the level is not reached,
        # otherwise return 'self'
        if destination_level > 0:
            destination_level -= 1
        elif destination_level == -1: 
            pass
        else:
            return self
        cdef unsigned int i = 0 
        #print "Result group value on destination level %i is %i and has %i subgroups" % (destination_level, value, N)
        for i from 0 <= i < N:           
            subgroup = <ResultGroup>subgroups[i]
            subgroup_value = subgroup.value
            if value == subgroup_value:
                return subgroup._try_subgroups(bond_variables,  destination_level)

        # subgroup was not_found, create new group or return self
        cdef np.int8_t length = self.invariant_list_length
        if length == 1:
            return self
        else:
            new_invariant_list = invariant_list[1:]
            self.subgroup_count += 1
            subgroup = ResultGroup(new_invariant_list,  value)
            subgroups.append(subgroup)
            return subgroup._try_subgroups(bond_variables, destination_level)

    #def get_subgroups_from_level(self, unsigned char level):
    #    return self._get_subgroups_from_level(level)
            
    cdef public list get_subgroups_from_level(self, unsigned char level):
        """
            return all subgroups that are on level 'level'
        """
        cdef list result = []
        cdef list subgroups = self.subgroups, subsubgroups
        cdef ResultGroup subgroup
        cdef unsigned int N = self.subgroup_count
        cdef unsigned int i = 0 
        #print "Current level %i, result count %i" % (level,  len(self.results))
        if level > 0:
            level -= 1
            for i from 0 <= i < N:
                subgroup = <ResultGroup>subgroups[i]
                subsubgroups = subgroup.get_subgroups_from_level(level)
                result.extend(subsubgroups)
                i += 1
            return result
        else:
            return subgroups

    cdef void clear_wos_from_level(self, unsigned char level):
        """
            remove all water orientations from the subgroup level 'level'
        """
        cdef ResultGroup subgroup
        cdef list subgroups = self.subgroups
        cdef unsigned int N = self.subgroup_count
        cdef unsigned int i = 0  
        if level > 0:
            level -= 1
            for i from 0 <= i < N:
                subgroup = <ResultGroup>subgroups[i]
                subgroup.clear_wos_from_level(level)
        else:
            self.wos = None
            
    cdef int get_total_number_of_grouped_wos(self, signed char destination_level):
        """
            return the total number of wos that reside in the subgroups and subsubgroup, ... of this group
        """
        cdef ResultGroup subgroup 
        cdef list subgroups = self.subgroups
        cdef np.int8_t N = self.subgroup_count
        cdef list new_invariant_list
        cdef list invariant_list = self.invariant_list
        cdef np.int8_t subgroup_value
   
        if destination_level > 0:
            destination_level -= 1
        elif destination_level == -1: 
            pass
        cdef unsigned int i = 0 
        cdef int result = 0
        
        for i from 0 <= i < N:
            subgroup = <ResultGroup>subgroups[i]
            if destination_level == 0:     
                if subgroup.wos is not None:
                    result += subgroup.wos.shape[0]
            else:
                result += subgroup.get_total_number_of_grouped_wos(destination_level)
        return result
        
    #def add_result(self, np.ndarray[np.int8_t, ndim=1] water_orientations,  np.ndarray[np.int8_t, ndim=2] bond_variables=None):
    #    self._add_result(water_orientations, bond_variables)

    
        
    cdef void add_result(self, np.int8_t[::1] water_orientations, np.int8_t[:, ::1] bond_variables):
        #if len(self.wos) == 0:
        #    self.wos = np.zeros((0, len(water_orientations)), dtype=np.int8)
        cdef np.int8_t[:, ::1] wos = self.wos

        if wos is None:
            wos = np.empty((0, water_orientations.shape[0]), dtype=np.int8)
        
        self.wos = append(wos, water_orientations)

        # = np.vstack((self.wos, water_orientations))
        #if bond_variables != None:
        #    self.bvs.append(bond_variables)
    
    #def get_wos(self):
    #    return self._get_wos()

    
    cdef np.int8_t[:, ::1] get_wos(self):
        return self.wos 

cdef public list merge_groups(list groups): 
    cdef list result = []
    cdef int i
    cdef bint merged
    cdef ResultGroup group, group_2
    if rank == 0:
        for i, group in enumerate(groups):
            merged = False
            for group_2 in result:
                if group_2.is_equal_with(group):
                    group_2.merge(group)
                    merged = True
                    break
            if not merged:
                result.append(group)
    return result

cdef inline np.int8_t[:, ::1] append(np.int8_t[:, ::1] a, np.int8_t[::1] b):
    cdef int a_length = a.shape[0], new_length = a.shape[0] +1, N = a.shape[1]
    cdef np.int8_t[:, ::1] result
    result = np.empty((new_length, N), dtype=np.int8)
    result[:a_length] = a
    result[a_length] = b
    return result
