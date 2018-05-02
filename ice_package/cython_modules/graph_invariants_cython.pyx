#cython: boundscheck=False
#cython: wraparound=False
# cython: infer_types=True
#cython: none_check=False
#cython: c_line_in_traceback=False
__package__ = "ice_package"
import numpy as np
from libc cimport math
cimport numpy as np
import os
from .symmetries.symmetry_operation_cython cimport SymmetryOperation, get_opposite_periodicity_axis_number
from cpython cimport bool
from .water_algorithm_cython cimport WaterAlgorithm

cdef void print_invariants(list invariant_list):
    cdef int i
    cdef int N = len(invariant_list)
    cdef Invariant invariant
    print "Printing %i invariants" % N
    for i in range(N):
        print i
        invariant = invariant_list[i]
        print "#%i: %s" % (i, invariant)
        print "-----------------"
        #if invariant.invariant_terms[0].indeces[0] == 0:
        #    raw_input()

cdef list get_dangling_bond_invariants(list invariants):
    cdef Invariant invariant
    cdef int i, j, N = len(invariants), M
    cdef list result = []
    cdef bint is_dbi
    cdef np.ndarray[np.uint8_t, ndim=2] original_indeces
    for i in range(N):
        invariant = <Invariant> invariants[i]
        original_indeces = invariant.original_indeces
        is_dbi = True
        if original_indeces is not None:
            M = original_indeces.shape[0]
            for j in range(M):
                if original_indeces[j, 0] != original_indeces[j, 1] or original_indeces[j, 2] != 13: 
                    is_dbi = False
                    break
            if is_dbi:
                result.append(invariant)
    return result

cdef list generate_invariants(list symmetry_operations, np.ndarray[np.int_t, ndim=3]  nearest_neighbors_nos, bint second_order, bint third_order, WaterAlgorithm water_algorithm, int maximum_number, bint dangling_bond_invariants):
    # invariants for each first order bond
    cdef list result = []
    result = generate_first_order_invariants(result, nearest_neighbors_nos,  symmetry_operations, water_algorithm, maximum_number, dangling_bond_invariants)
    if second_order:
        result = generate_second_order_invariants(result, nearest_neighbors_nos,  symmetry_operations, water_algorithm, maximum_number, dangling_bond_invariants)
    #if third_order:
    #    generate_third_order_invariants(result, nearest_neighbors_nos,  symmetry_operations)
    #print "Initially %i invariants." % len(invariants)
    #result = remove_empty_invariants(result)
    #result = remove_equal_invariants(result)
    result = order_invariants_by_length(result)
    
    #_print_invariants(result)
    #print "Finally %i invariants @ generate_invariants" % len(result)
    return result
    
cdef list generate_first_order_invariants(list result, np.ndarray[np.int_t, ndim=3]  nearest_neighbors_nos, list symmetry_operations, WaterAlgorithm water_algorithm, int maximum_number, bint dangling_bond_invariants):
    cdef int i, j, k, nn
    cdef int N = nearest_neighbors_nos.shape[1], M = nearest_neighbors_nos.shape[2], O = len(symmetry_operations)
    cdef np.int_t[:, :, ::1] nnn = nearest_neighbors_nos
    cdef np.uint8_t periodicity_axis
    cdef SymmetryOperation symmetry_operation
    cdef Invariant invariant
    cdef list invariant_terms
    cdef dict sbvm
    cdef np.ndarray[np.uint8_t, ndim=2] original_indeces
    cdef InvariantTerm invariant_term
    for i from 0 <= i < N:
        for j from 0 <= j < M:
            nn = nearest_neighbors_nos[0, i, j]
            periodicity_axis = nearest_neighbors_nos[2, i, j]
            if (not dangling_bond_invariants and i < nn) or (i == nn and periodicity_axis == 13):
                invariant_terms = []
                ps = ""
                original_indeces = np.array([[i, nnn[0, i, j], nnn[2, i, j]]], dtype = np.uint8)
                if periodicity_axis != 13: ps = "_%i" % periodicity_axis 
                for k from 0 <= k < O:
                    symmetry_operation = symmetry_operations[k]
                    sbvm = symmetry_operation.get_symbolic_bond_variable_matrix(nearest_neighbors_nos, water_algorithm, False)
                    invariant_term = <InvariantTerm> InvariantTerm(np.array([sbvm[i][nnn[0, i, j]][nnn[2, i, j]]], dtype=np.uint8))
                    invariant_terms.append(invariant_term)
                    a = np.array([sbvm[i][nnn[0, i, j]][nnn[2, i, j]]], dtype=np.uint8)
                    
                
                invariant = Invariant(invariant_terms, name='I_{%i, %i%s}' % (i, nnn[0, i, j], ps), original_indeces = original_indeces)
                #print invariant
                
                invariant.reduce()
                invariant.order()
                if invariant.is_invariant(symmetry_operations) and not equal_or_opposite_invariant_found(invariant, result) and len(invariant.invariant_terms) != 0:
                    result.append(invariant)
                    #print "%i/%i invariants found" % (len(result), maximum_number)
                    if len(result) >= maximum_number:
                        return result
                    
    return result
    


cdef list generate_second_order_invariants(list result, np.ndarray[np.int_t, ndim=3]  nearest_neighbors_nos, list symmetry_operations, WaterAlgorithm water_algorithm, int maximum_number, bint dangling_bond_invariants):
    #invariants for each second order bond
    cdef SymmetryOperation symmetry_operation
    cdef Invariant invariant
    cdef int i, j, k, l, m
    cdef np.uint8_t periodicity_axis, periodicity_axis2, nn, nn2, N = nearest_neighbors_nos.shape[1], M = nearest_neighbors_nos.shape[2], O = len(symmetry_operations)
    cdef np.int_t[:, :, ::1] nnn = nearest_neighbors_nos
    cdef list invariant_terms
    cdef np.ndarray[np.uint8_t, ndim=2] original_indeces
    cdef InvariantTerm invariant_term
    cdef dict sbvm
    counter = 0
    for i from 0 <= i < N:
        nns = nnn[0, i]
        #if i != 0:
        #    continue
        for j from 0 <= j < M:
            nn = nns[j]
            #if nn != 1 and nn != 2:
            #    continue
            periodicity_axis = nnn[2, i, j]
            if (not dangling_bond_invariants and i < nn) or (i == nn and periodicity_axis == 13):
                for k from 0 <= k < N:
                    nns2 = nnn[0, k]
                    for l from 0 <= l < M:
                        nn2 = nns2[l]
                        periodicity_axis2 = nnn[2, k, l]
                        
                        counter += 1
                        if (not dangling_bond_invariants and k < nn2) or (k == nn2 and periodicity_axis2 == 13):
                            invariant_terms = []
                            for m from 0 <= m < O:
                                symmetry_operation = <SymmetryOperation>symmetry_operations[m]
                                sbvm = symmetry_operation.get_symbolic_bond_variable_matrix(nearest_neighbors_nos, water_algorithm, False)
                                #new_no1 = symmetry_operation.molecule_change_matrix[i]
                                #new_no2 = symmetry_operation.molecule_change_matrix[k]
                                #k1 = symmetry_operation.nn_change_matrix[i][j]
                                #k2 = symmetry_operation.nn_change_matrix[k][l]
                                #invariant_terms.append(InvariantTerm(np.array([[new_no1, nearest_neighbors_nos[0][new_no1][k1], nearest_neighbors_nos[2][new_no1][k1]],  [new_no2, nearest_neighbors_nos[0][new_no2][k2], nearest_neighbors_nos[2][new_no2][k2]]], dtype=np.uint8)))
                                invariant_term = InvariantTerm(np.array([sbvm[i][nnn[0, i, j]][nnn[2, i, j]],  sbvm[k][nnn[0, k, l]][nnn[2, k, l]]], dtype=np.uint8))
                                invariant_terms.append(invariant_term)
                                
                                #invariant_terms.append(InvariantTerm(np.array([sbvm[i][nnn[0, i, j]][nnn[2, i, j]],  sbvm[k][nnn[0, k, l]][nnn[2, k, l]]], dtype=np.uint8)))
                                #if i == 0 and nn == 5 and periodicity_axis == 10 and k == 4 and nn2 == 6 and periodicity_axis2 == 13:
                                #    print symmetry_operation.molecule_change_matrix
                                #    print invariant_terms[m]
                                #    raw_input()
                            p1s = ""
                            p2s = ""                            
                            if periodicity_axis != 13: p1s = "_%i" % periodicity_axis 
                            if periodicity_axis2 != 13: p2s = "_%i" % periodicity_axis2
                            original_indeces = np.array([[i, nn, periodicity_axis], [k, nn2, periodicity_axis2]], dtype = np.uint8)
                            invariant = Invariant(invariant_terms, name='I_{%i, %i%s; %i, %i%s}' % (i, nnn[0, i, j], p1s, k, nn2, p2s), original_indeces = original_indeces)
                            invariant.reduce()
                            invariant.order()
                            if invariant.is_invariant(symmetry_operations) and not equal_or_opposite_invariant_found(invariant, result) and len(invariant.invariant_terms) != 0:
                                result.append(invariant)
                                #print "%i/%i invariants found" % (len(result), maximum_number)
                                if len(result) >= maximum_number:
                                    return result
    
        
    return result
"""                            
def generate_third_order_invariants(result,  nearest_neighbors_nos,  symmetry_operations):
    #invariants for each second order bond
    count = 0
    print "Generating 3rd order invariants"
    for i in range(len(nearest_neighbors_nos[0])):
        print "  -Molecule %i" %i
        for n,  nn in enumerate(nearest_neighbors_nos[0][i]):
            if i <= nn:
                periodicity_axis = nearest_neighbors_nos[2][i][n]
                for i2 in range(len(nearest_neighbors_nos[0])):
                    for n2,  nn2 in enumerate(nearest_neighbors_nos[0][i2]):
                        
                        if i2 <= nn2:
                            periodicity_axis2 = nearest_neighbors_nos[2][i2][n2]
                            for i3 in range(len(nearest_neighbors_nos[0])):
                                for n3,  nn3 in enumerate(nearest_neighbors_nos[0][i3]):
                                    if i3 <= nn3:
                                        invariant_terms = []
                                        periodicity_axis3 = nearest_neighbors_nos[2][i3][n3]
                                        
                                        for symmetry_operation in symmetry_operations:
                                            invariant_terms.append(InvariantTerm([symmetry_operation.symbolic_bond_variable_matrix[periodicity_axis][i][nn],  symmetry_operation.symbolic_bond_variable_matrix[i2][nn2][periodicity_axis2], symmetry_operation.symbolic_bond_variable_matrix[i3][nn3][periocity_axis3]]))

                                        invariant = Invariant(invariant_terms)
                                        invariant.reduce()
                                        result.append(invariant)
                                        count += 1
                                        if count > 100:
                                            return
                            
"""

cdef bint equal_or_opposite_invariant_found(Invariant invariant, list invariants):
    """
        Also removes opposite invariants
    """
    cdef list result = []
    cdef int i, j, equal, opposite
    cdef int N = len(invariants)
    cdef Invariant invariant_2
    cdef bool equa
    cdef bool add 
    for i from 0 <= i < N:
        invariant_2 = invariants[i]
        add = True
        equa = invariant.is_equal(invariant_2)
        if equa or invariant.is_opposite_of(invariant_2):
            return True
    return False

cdef list remove_equal_invariants(list invariants):
    """
        Also removes opposite invariants
    """
    cdef list result = []
    cdef int i, j, equal, opposite
    cdef int N = len(invariants), M
    cdef Invariant invariant, invariant_2
    cdef bool equa
    cdef bool add 
    for i from 0 <= i < N:
        invariant = invariants[i]
        add = True
        M = len(result)
        for j from 0 <= j < M:
            invariant_2 = result[j]
            equa = invariant.is_equal(invariant_2)
            if equa or invariant.is_opposite_of(invariant_2):
                if equa:
                    equal += 1
                else:
                    opposite += 1
                add = False
                break
        if add:
            result.append(invariant)
    #print "Removing %i invariants because they are equal with something else" % equal
    #print "Removing %i invariants because they are opposite of other invariant" % opposite
    #_print_invariants(result)
    return result
    
cdef list remove_empty_invariants(list invariants):
    cdef list result = []
    cdef Invariant invariant
    cdef int N = len(invariants)
    for i from 0 <= i < N:
        invariant = invariants[i]   
        if len(invariant.invariant_terms) != 0:
            result.append(invariant)
        else:
            if invariant.constant != 0:
                print "Removing constant invariant"
                print invariant
    return result

cdef void save_invariants(char* folder, list invariants):
    cdef Invariant invariant
    for i, invariant in enumerate(invariants):
        if not os.path.exists(folder+"invariants"):
            os.makedirs(folder+"invariants")
        a, m, c, mu, oi = invariant.as_array()
        np.save(folder+"invariants/invariant_%i" % (i), a)
        np.save(folder+"invariants/invariant_%i_m" % (i), m)
        np.save(folder+"invariants/invariant_%i_c" % (i), c)
        np.save(folder+"invariants/invariant_%i_mu" % (i), mu)
        np.save(folder+"invariants/invariant_%i_oi" % (i), oi)

#def get_invariants(list symmetry_operations,  nearest_neighbors_nos, second_order=True, third_order=False, folder="", reload=False):
#    return _get_invariants(symmetry_operations,  nearest_neighbors_nos, second_order, third_order, folder, reload)
    

cdef list load_invariants(char* folder):
    cdef int i = 0
    cdef int l, N
    cdef list result = []
    i = 0
    while True:
        try:
            array = np.load(folder+"invariants/invariant_%i.npy" % (i))
            multipliers = np.load(folder+"invariants/invariant_%i_mu.npy" % (i))
            minuses = np.load(folder+"invariants/invariant_%i_m.npy" % (i))
            constant = np.load(folder+"invariants/invariant_%i_c.npy" % (i))
            original_indeces = np.load(folder+"invariants/invariant_%i_oi.npy" % (i))
        except IOError:
            if i == 0:
                print "Loading prestored invariants failed. (Re)generating invariants."
            else:
                print "Loaded %i invariants" %i
            break
        # invariant terms are in array
        array = np.array(array, dtype=np.uint8)
        minuses = np.array(minuses, dtype=bool)
        invariant_terms = []
        N = len(array)
        for l in range(N):
            row = array[l]
            invariant_terms.append(InvariantTerm(row, minus=bool(minuses[l]), multiplier = <np.int8_t> multipliers[l]))
        invariant = Invariant(invariant_terms, original_indeces = original_indeces)
        invariant.constant = constant[0]
        result.append(invariant)
        i += 1
    return result

cdef get_invariants(list symmetry_operations, np.ndarray[np.int_t, ndim=3]  nearest_neighbors_nos, bint second_order, bint third_order, folder, bint reload, WaterAlgorithm water_algorithm, int maximum_number, bint dangling_bond_invariants):
    if folder[len(folder) - 1] != "/":
        folder = folder + "/"
    cdef list result = []
    if not reload:
        result = load_invariants(folder)
    if len(result) == 0 or len(result) < maximum_number or dangling_bond_invariants:
        if len(result) != 0 and len(result) < maximum_number and not dangling_bond_invariants:
            print "The number of invariants is not sufficient. Regenerating invariants."
        result = generate_invariants(symmetry_operations,  nearest_neighbors_nos, second_order, third_order, water_algorithm, maximum_number, dangling_bond_invariants)
        if not dangling_bond_invariants:
            save_invariants(folder, result)
            result = load_invariants(folder)
    #print_invariants(result)
    return result


cdef list order_invariants_by_length(list invariants):
    cdef list result = []
    cdef int i, N = len(invariants)
    cdef Invariant invariant
    cdef list lengths = []
    cdef list ones = []
    for i from 0 <= i < N:
        invariant = invariants[i]
        lengths.append(len(invariant.invariant_terms))
        ones.append(1)
    ind = np.lexsort((lengths, ones))
    return [invariants[i] for i in ind]
    
        


cdef Invariant get_dangling_bond_invariant(np.ndarray[np.int_t, ndim=3]  nearest_neighbors_nos):
    cdef np.int_t[:, :, :] nnn = nearest_neighbors_nos
    cdef int i, j, N = nnn.shape[1], M = nnn.shape[2]
    cdef list invariant_terms = []
    for i from 0 <= i < N:
        for j from 0 <= j < M:
            if nnn[0, i, j] == i and nnn[2, i, j] == 13:
               invariant_terms.append(InvariantTerm(np.array([i, i, 13], dtype=np.uint8))) 
    cdef Invariant invariant = Invariant(invariant_terms, name='Dangling bond invariant' , original_indeces = None)
    return invariant



cdef void initialize_new_indexing(dict conversion_table, list invariants) except *:
    cdef Invariant invariant
    cdef int N = len(invariants)
    cdef unsigned char i
    
    for i from 0 <= i < N:
        invariant = invariants[i]
        invariant.initialize_new_indexing(conversion_table)
   
      
  
cdef class InvariantTerm:
    def __cinit__(self, np.ndarray[np.uint8_t, ndim=2] indeces, bint minus = False, np.ndarray[np.uint8_t, ndim=2] new_indeces = None, bint is_constant = False, np.int8_t constant_value = 0, np.int8_t multiplier = 1):
        """
            Index[0] Molecule that bonds
            Index[1] Molecule which to the bond is
            Index[2] The number of periodic axis (13 == nonperiodic)
        """
        # order the indeces so that the smaller index is before and change the sign similarly    
        cdef np.ndarray[np.uint8_t, ndim=2] indec
        cdef np.ndarray[np.uint8_t, ndim=2] new_indece
        cdef int i
        cdef unsigned char new_index_2
        cdef np.ndarray[np.uint8_t, ndim=1] index, previous_index = None
        cdef np.int8_t N 
        
        if new_indeces is None:
            indec = np.array(indeces, dtype=np.uint8)
            N = indec.shape[0]
            new_indece = np.zeros_like(indeces)
            for i from 0 <= i < N:
                index = indec[i]
                if index[0] > index[1]:
                    new_index_2 = get_opposite_periodicity_axis_number(index[2])
                    new_indece[i] = np.array([index[1], index[0], new_index_2], dtype=np.uint8)
                    minus = not minus
                else:
                    new_indece[i] = index
            self.indeces = np.array(new_indece, dtype=np.uint8)
            
            is_constant = True
            self.order()
            if N % 2 == 0: 
                for i from 0 <= i < N:
                    index = indec[i]
                    if previous_index is not None and np.any(np.not_equal(index, previous_index)):
                        is_constant = False
                        break
                    previous_index = index
                if is_constant:
                    constant_value = 1
                    if minus:
                        constant_value *= -1
                    
            else:
                is_constant = False
        else:
            self.indeces = indeces
            self.new_indeces = new_indeces
        self.multiplier = multiplier
        self.length = self.indeces.shape[0]
        self.minus = minus   
        self.is_constant = is_constant
        self.constant_value = constant_value

    def __reduce__(self):
        return (InvariantTerm, # constructor to be called
        (self.indeces.base, self.minus, self.new_indeces.base, self.is_constant, self.constant_value, self.multiplier), # arguments to constructor 
         None , None , None)
        
    cdef void order(self):
        ind = np.lexsort((self.indeces[:, 2], self.indeces[:,  1],  self.indeces[:,  0]))
        self.indeces = np.array([self.indeces[i] for i in ind], dtype=np.uint8)

    def __str__(self):
        if self.multiplier == 0:
            return ""
        result = None
        previous_index = None
        cdef signed char mul = 0
        cdef int i
        cdef np.uint8_t[:, ::1] indeces = self.indeces
        cdef int N = indeces.shape[0]
        if self.minus:
            result = " -"
        else:
            result = " +"
        result += " %i" % self.multiplier
        for i from 0 <= i < N:
            if previous_index is None or (previous_index == indeces[i]).all():
                mul += 1
            else: 
                if result is not None:
                    result += " * "
                else:
                    result = ""
                s = ""
                if previous_index[2] != 13:
                    s = "_[%i]" % previous_index[2]
                if mul > 1:
                    result += "b_{%i, %i%s}^%i" % (previous_index[0],  previous_index[1],  s, mul)
                else:
                    result += "b_{%i, %i%s}" % (previous_index[0],  previous_index[1],  s)
                mul = 1
            previous_index = indeces.base[i]

        if result != None:
            result += " * "
        s = ""
        if previous_index[2] != 13:
            s = "_[%i]" % previous_index[2]
        if mul > 1:
            result += "b_{%i, %i%s}^%i" % (previous_index[0],  previous_index[1],  s, mul)
        else:
            result += "b_{%i, %i%s}" % (previous_index[0],  previous_index[1],  s)
        return result
    
    cdef bint is_reducable_with(self, InvariantTerm other_term):
        if other_term is None:
            return False
        if self.is_constant != other_term.is_constant:
            return False
        #elif self.is_constant:
        #    return False
        if other_term.indeces.shape[0] != self.indeces.shape[0]:
            return False
        if np.all(np.equal(other_term.indeces, self.indeces).flatten()):
            return True
        else:
            return False

    cdef public InvariantTerm apply_symmetry_operation(self, SymmetryOperation operation):
        sbvm = operation.symbolic_bond_variable_matrix
        cdef list indeces = []
        cdef int i, N = len(self.indeces)
        for i in range(N):
            indeces.append(sbvm[self.indeces[i, 0]][self.indeces[i, 1]][self.indeces[i, 2]])
        cdef InvariantTerm invariant_term = <InvariantTerm> InvariantTerm(np.array(indeces, dtype=np.uint8), minus = self.minus)
        return invariant_term
            
    cdef void reduce(self, InvariantTerm other_term):
        mul = self.multiplier
        other_mul = other_term.multiplier
        if self.minus:
            mul *= -1
        if other_term.minus:
            other_mul *= -1 
        self.multiplier = mul + other_mul
        self.minus = False
        
        
    cdef public bint is_opposite_of(self, InvariantTerm other_term): 
        if other_term is None:
            return False
        #if self.is_constant != other_term.is_constant: 
        #    return False
        #elif self.is_constant and self.constant_value == -other_term.constant_value:
        #    return True
 
        if len(other_term.indeces) != len(self.indeces):
            return False
        if np.all(np.equal(other_term.indeces, self.indeces).flatten()) and self.minus != other_term.minus and self.multiplier == other_term.multiplier:
            return True
        if np.all(np.equal(other_term.indeces, self.indeces).flatten()) and self.minus == other_term.minus and self.multiplier == -other_term.multiplier:
            return True   
        
        for i,  index in enumerate(self.indeces):
            if (index[0] == index[1] or index[2] != other_term.indeces[i][2] or index[0] != other_term.indeces[i][1] or index[1] != other_term.indeces[i][0]):
                return False
        return True


    cdef public  np.int8_t get_value(self, np.int8_t[:, ::1] bond_variables) except *: 
        cdef np.int8_t result = 1 
        cdef int i
        cdef np.uint8_t[:, ::1] indeces = self.new_indeces
        cdef np.int8_t bv
        cdef np.uint8_t N = indeces.shape[0]
        
        for i from 0 <= i < N:
            #index = indeces[i]
            bv = bond_variables[indeces[i, 0], indeces[i, 1]]
            result *= bv
            #mul(result, bond_variables, index[0], index[1])
        if self.minus:
            result *= -1
        result *= self.multiplier
        return result


    
    def __richcmp__(self, other, i):
        if i == 0:
            return self.lt(other)
        elif i == 2:
            return self.eq(other)
        elif i == 4:
            return self.gt(other)
        elif i == 3:
            return self.ne(other)
    
    
    def lt(self, other):
        """Indeces have to be ordered if this is used"""
        cdef int i
        cdef np.ndarray[np.uint8_t, ndim=1] index
        cdef int N = len(self.indeces)
        for i from 0 <= i < N:
            index = self.indeces[i]
            if other.indeces[i][0] != self.indeces[i][0]:
                return self.indeces[i][0] < other.indeces[i][0] 
            elif other.indeces[i][1] != self.indeces[i][1]:
                return self.indeces[i][1] < other.indeces[i][1]
        # They are equal
        return False

    def  gt(self, other):
        """Indeces have to be ordered if this is used"""
        cdef int i
        cdef np.ndarray[np.uint8_t, ndim=1] index
        cdef int N = len(self.indeces)
        for i from 0 <= i < N:
            index = self.indeces[i]
            if other.indeces[i][0] != self.indeces[i][0]:
                return self.indeces[i][0] > other.indeces[i][0] 
            elif other.indeces[i][1] != self.indeces[i][1]:
                return self.indeces[i][1] > other.indeces[i][1]
        # They are equal
        return False
        
    def eq(self, InvariantTerm other):
        if other == None:
            return False
        return self.is_constant == other.is_constant and self.constant_value == other.constant_value and len(self.indeces) == len(other.indeces) and np.all(np.equal(self.indeces, other.indeces).flatten()) and self.minus == other.minus
        
    cdef bool ne(self, InvariantTerm other):
        cdef bool d = self.is_constant != other.is_constant
        if d:
            return self.constant_value != other.constant_value
        cdef bool a = any(np.not_equal(self.indeces, other.indeces).flatten())
        cdef bool b = len(self.indeces) != len(other.indeces) 
        cdef bool c =  self.minus != other.minus
        return a or b or c
     
    def as_array(self):
        ind = self.indeces.copy()
        return ind
            

    def get_indeces(self):
        return self.indeces 


    cdef public void initialize_new_indexing(self, dict index_conversion_table):
        cdef np.uint8_t[:, ::1] result = np.zeros((self.indeces.shape[0], 2), dtype=np.uint8)
        cdef np.uint8_t[::1] index_conversion
        cdef np.uint8_t i 
        for i, index in enumerate(self.indeces):
            index_conversion = index_conversion_table[index[0]][index[1]][index[2]]
            
            for j in range(2):
                result[i, j] = index_conversion[j] 
        self.new_indeces = result

cdef class Invariant:
    
    def __cinit__(self, list invariant_terms, name="", bool constants_removed = False, np.int8_t constant_value = 0, np.ndarray[np.uint8_t, ndim=2] original_indeces = None):
        #result = []    
        if not constants_removed:
            #self.invariant_terms = self.remove_constant_terms(invariant_terms)
            self.invariant_terms = invariant_terms
        else:
            self.invariant_terms = invariant_terms
            self.constant = constant_value
        
        self.length = len(self.invariant_terms)
        #self.invariant_terms = np.sort(self.invariant_terms)
        #self.normalizing_constant  = 1. / float(len(invariant_terms))
        self.original_indeces = original_indeces 
        self.name = name

    def __reduce__(self):
        
        return (Invariant, # constructor to be called
        (self.invariant_terms, self.name, True, self.constant, self.original_indeces), # arguments to constructor 
         None , None , None)


    cdef list remove_constant_terms(self, list invariant_terms):
        cdef np.int8_t constant = 0
        cdef int i, N = len(invariant_terms)
        cdef list result = []
        cdef InvariantTerm term
        for i from 0 <= i < N:
            term = invariant_terms[i]
            if term.is_constant:
                constant += term.constant_value
            else:
                result.append(term)
        self.constant = constant
        return result

    cdef void debug_value(self, np.int8_t[:, ::1] bond_variables)  except *:
        cdef  np.int8_t result = 0
        #res = "" 
        cdef list invariant_terms = self.invariant_terms
        cdef InvariantTerm term
        cdef int i
        cdef np.int8_t N = self.length
        cdef np.int8_t term_value
        res = ""
        if self.original_indeces is not None:
            for i, index in enumerate(self.original_indeces):
                if i != 0:
                    res += " * "
                res += "I_{%i, %i}_%i" % (index[0], index[1], index[2])
            res += ":"
        try:
            for i from 0 <= i < N:
                term = <InvariantTerm>invariant_terms[i]
                
                term_value = term.get_value(bond_variables)
                result += term_value
                res += "%s: %i" % (term, term_value) 
        except:
            import traceback
            print traceback.format_exc()
        
        res += " + %i = %i" % (self.constant, result)
        print res
        

    cdef public char _get_value(self, np.int8_t[:, ::1] bond_variables)  except *:
        cdef  np.int8_t result = 0
        #res = "" 
        cdef list invariant_terms = self.invariant_terms
        cdef InvariantTerm term
        cdef int i
        cdef np.int8_t N = self.length
        cdef np.int8_t term_value
        #try:
        for i from 0 <= i < N:
            term = <InvariantTerm>invariant_terms[i]
            #res += " + %i" % term.get_value(bond_variables)
            term_value = term.get_value(bond_variables)
            result += term_value
        #except:
        #    import traceback
        #    print traceback.format_exc()

        #result = self.normalizing_constant * result
        result += self.constant
        return result
    
    cdef void reduce(self):
        cdef list removed_indeces = []
        cdef list new_invariant_terms = []
        cdef InvariantTerm invariant_term, invariant_term_2
        cdef list invariant_terms = self.invariant_terms
        cdef int i, j
        cdef bint removed
        cdef int M, N = len(invariant_terms)
        for i from 0 <= i < N:
            invariant_term = <InvariantTerm>invariant_terms[i]
            M = len(new_invariant_terms)
            removed = False
            for j from 0 <= j < M:
                invariant_term_2 = <InvariantTerm>new_invariant_terms[j]
                if invariant_term_2.is_reducable_with(invariant_term):
                    invariant_term_2.reduce(invariant_term)
                    removed = True
                    break
            if not removed:
                new_invariant_terms.append(invariant_term)
        cdef list result = []
        M = len(new_invariant_terms)
        for i from 0 <= i < M:
            invariant_term = new_invariant_terms[i]
            if invariant_term.multiplier != 0:
                result.append(invariant_term)
        self.length = len(result)
        self.invariant_terms = result

    cdef void  order(self):
        
        cdef np.ndarray[np.int_t, ndim=1] ind
        cdef int i, l, counter
        
        cdef np.ndarray[np.uint8_t, ndim=2] np_order_list, lists
        cdef list invariant_terms = self.invariant_terms
        cdef int N = len(invariant_terms)
        cdef int order_of_invariant # first, second or third order
        cdef InvariantTerm term
        if N != 0:
            order_of_invariant = invariant_terms[0].indeces.shape[0]
            np_order_list = np.empty((N, order_of_invariant*3), dtype=np.uint8)
            lists = np.empty((order_of_invariant*3, N), dtype=np.uint8)
            for i from 0 <= i < N:
                term = <InvariantTerm>invariant_terms[i]
                np_order_list[i] = term.get_indeces().base.flatten()
            l = order_of_invariant -1;
            counter = 0
            while l >= 0:
                lists[l*3 + 2] = np_order_list[:, counter*3+2]
                lists[l*3 + 1] = np_order_list[:, counter*3+1]
                lists[l*3 + 0] = np_order_list[:, counter*3+0]
                
                l -= 1
                counter += 1
            ind = np.lexsort(lists)
            self.invariant_terms = [invariant_terms[i] for i in ind] 

    def __richcmp__(self, other, i):
        if i == 0:
            return self.lt(other)
        elif i == 2:
            return self.is_equal(other)
        elif i == 4:
            return self.gt(other)
        elif i == 3:
            return self.ne(other)
        
    def is_equal(self, Invariant other):
        cdef InvariantTerm invariant_term, invariant_term_2
        cdef int i
        cdef int N = len(self.invariant_terms)
        """Terms have to be ordered if this is used"""
        if (N != len(other.invariant_terms) or self.constant != other.constant):
            return False
        for i from 0 <= i < N:
            invariant_term = self.invariant_terms[i]
            invariant_term_2 = other.invariant_terms[i]
            if invariant_term.ne(invariant_term_2):
                #print "Returning false because of ne"
                #print invariant_term
                #print invariant_term_2
                return False
        return True

    # Opposite invariants are different but do not help the calculation at all
    cdef public bool is_opposite_of(self, Invariant other):
        cdef InvariantTerm invariant_term
        if (len(self.invariant_terms) != len(other.invariant_terms)):
            return False
        for i, invariant_term in enumerate(self.invariant_terms):
            if not invariant_term.is_opposite_of(other.invariant_terms[i]):
                return False
        return True      

    cdef bint is_invariant(self, list symmetry_operations):
        return True
        """ Checks that with any symmetry operation, the "invariant" remains the same, i.e., generates the same function"""
        cdef SymmetryOperation operation
        cdef int i, N = len(symmetry_operations), M = len(self.invariant_terms)       
        cdef list invariant_terms
        cdef Invariant invariant
        cdef InvariantTerm invariant_term, new_invariant_term
        self.order()
        for i in range(N):
            invariant_terms = []
            operation = <SymmetryOperation> symmetry_operations[i]
            for j in range(M):
                invariant_term = <InvariantTerm> self.invariant_terms[j]
                new_invariant_term = invariant_term.apply_symmetry_operation(operation)
                invariant_terms.append(new_invariant_term)
            invariant = Invariant(invariant_terms, original_indeces = self.original_indeces)
            invariant.order()
            if operation.additional_requirements is None and not self.is_equal(invariant):
                print i, operation.molecule_change_matrix
                print self
                print invariant
                print "Invariant is not invariant!!"
                return False
        return True
                

    def __str__(self):
        if self.name != None and self.name != "":
            result = self.name +": " 
        else:
            result = ""
            if self.original_indeces is not None:
                for i, index in enumerate(self.original_indeces):
                    if i != 0:
                        result += " * "
                    result += "I_{%i, %i}_%i" % (index[0], index[1], index[2])
                result += ":"
            
        previous_term = None
        pt_mul = 0
        if len(self.invariant_terms) > 0:
            for i, invariant_term in enumerate(self.invariant_terms):
                result += str(invariant_term)
        if self.constant >= 0: 
            result += " + %i" % self.constant
        elif self.constant < 0:
            result += " %i" % self.constant
        return result
 
    def as_array(self):
        cdef list result = []
        cdef list minuses = []
        cdef list multipliers = []
        cdef int i
        cdef int N = len(self.invariant_terms)
        for i in range(N):
            invariant_term = self.invariant_terms[i]
            minuses.append(invariant_term.minus)
            result.append(invariant_term.as_array())
            multipliers.append(invariant_term.multiplier)
        cdef list constant = [self.constant]
        return np.array(result, dtype=int), np.array(minuses, dtype=int),  np.array(constant, dtype=int),  np.array(multipliers, dtype=int), self.original_indeces

    cdef public void initialize_new_indexing(self, dict conversion_table):
        cdef InvariantTerm invariant_term
        cdef int N = len(self.invariant_terms)
        for i from 0 <= i < N:
            invariant_term = self.invariant_terms[i]
            invariant_term.initialize_new_indexing(conversion_table)
        
        
        
        
