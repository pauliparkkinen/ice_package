cimport numpy as np
from cpython cimport bool
cimport stdlib

from .water_algorithm_cython cimport WaterAlgorithm
from .symmetries.symmetry_operation_cython cimport SymmetryOperation

cdef list get_dangling_bond_invariants(list invariants)
cdef get_invariants(list symmetry_operations, np.ndarray[np.int_t, ndim=3]  nearest_neighbors_nos, bint second_order, bint third_order, folder, bint reload, WaterAlgorithm water_algorithm, int maximum_number, bint dangling_bond_invariants)
cdef void initialize_new_indexing(dict conversion_table, list invariants) except *
cdef void print_invariants(list invariant_list)
cdef Invariant get_dangling_bond_invariant(np.ndarray[np.int_t, ndim=3]  nearest_neighbors_nos)

cdef class InvariantTerm: 
    cdef np.uint8_t length
    cdef public np.uint8_t[:, ::1] indeces, new_indeces
    cdef public bint is_constant
    cdef public np.int8_t constant_value
    cdef public np.int8_t multiplier
    cdef public bint minus
    cdef public signed char get_value(self, np.int8_t[:, ::1] bond_variables)
    cdef public bint is_opposite_of(self, InvariantTerm other_term)
    cdef public void initialize_new_indexing(self, dict index_conversion_table)
    cdef bint is_reducable_with(self, InvariantTerm other_term)
    cdef void reduce(self, InvariantTerm other_term)
    cdef public InvariantTerm apply_symmetry_operation(self, SymmetryOperation operation)
    
    #cdef bool eq(self, InvariantTerm other)
    cdef void order(self)
    cdef public bool ne(self, InvariantTerm other)

cdef class Invariant:
    cdef float normalizing_constant
    cdef list remove_constant_terms(self, list invariant_terms)
    cdef np.int8_t constant
    cdef np.ndarray original_indeces
    cdef str name 
    cdef np.uint8_t length
    cdef public void initialize_new_indexing(self, dict index_conversion_table)
    cdef public char _get_value(self, np.int8_t[:, ::1] bond_variables)
    cdef void debug_value(self, np.int8_t[:, ::1] bond_variables)  except *
    cdef public list invariant_terms
    cdef public bool is_opposite_of(self, Invariant other)
    cdef bint is_invariant(self, list symmetry_operations)
    cdef void order(self)
    cdef void reduce(self)
    

