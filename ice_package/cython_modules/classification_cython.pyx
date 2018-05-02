#cython: boundscheck=True
#cython: wraparound=False
# cython: nonecheck=True
#cython: infer_types=True

cimport cython
from classification import close_files
from graph_invariants_cython cimport Invariant
#import numpy as np
cimport numpy as np

np.import_array()
from help_methods import *
from time import time

def get_coordination_numbers(nearest_neighbors_nos):
    return _get_coordination_numbers(nearest_neighbors_nos).base

cdef np.int_t[:] _get_coordination_numbers(np.int_t[:, :, :] nearest_neighbors_nos):
    cdef int N = nearest_neighbors_nos.shape[1], M = 4
    cdef np.int_t[:] result = np.empty(N, dtype=np.int)
    cdef int molecule_type, molecule_no, j, coordination_number
    for molecule_no from 0 <= molecule_no < N:
        coordination_number = 0
        for bond_no from 0 <= bond_no < M:
            if nearest_neighbors_nos[0, molecule_no, bond_no] != molecule_no or nearest_neighbors_nos[2, molecule_no, bond_no] != 13:
                coordination_number += 1
        result[molecule_no] = coordination_number
    return result

cdef inline np.int_t get_water_molecule_type(np.int_t molecule_no, np.int8_t[:] water_orientations, np.int_t[:, :, :] nearest_neighbors_nos):
    """
      
        Return the type of the molecule.
        -1 : not set
        In case of neutral molecule:
        
         0 : AADD
         1 : AAD
         2 : ADD
         3 : AD
         4 : AA
         5 : DD
         6 : A
         7 : D
        In case of H3O+ (8 + index):
         0 : ADDD
         1 : ADD
         2 : DDD
         3 : AD
         4 : DD
         5 : A
         6 : D
        In case of OH- (15 + index ):
         0 : AAAD
         1 : AAA
         2 : AAD
         3 : AD
         4 : AA
         5 : A
         6 : D
    """
    # TODO: Does not handle 0, 1 and two coordinated water molecules
    cdef np.int8_t[:] bvv
    cdef list indeces
    cdef np.int_t index 
    cdef np.int8_t orientation
    cdef np.int8_t amount
    cdef int result
   

    
    if water_orientations[molecule_no] == -1:
        return -1;
    else:
        result = 0
        orientation = water_orientations[molecule_no]
        if orientation > 9:
            result = 8
        elif orientation > 6:
            result = 15

        # Find the indeces of dangling bonds
        indeces = get_dangling_bond_indeces(molecule_no, nearest_neighbors_nos, 4)
        if len(indeces) == 0:
            return result  # No dangling bonds, pick the first
        elif len(indeces) == 1:
            bvv = get_bond_variable_values_from_water_orientation(orientation)
            index = indeces[0]
            if bvv[index] == -1: # One dangling oxygen
                return result + 2
            else: # One dangling hydrogen
                return result + 1
        elif len(indeces) == 2: # two fold coordinated
            bvv = get_bond_variable_values_from_water_orientation(orientation)
            amount = 0
            for index in indeces:
                amount += bvv[index]
            if result == 0: # neutral
                if amount == 0: # one dO and one dH
                    return result + 3
                elif amount == 2: # two dH's
                    return result + 4
                elif amount == -2: # two dO's
                    return result + 5
            elif result == 8: # H3O+
                if amount == 0: # one dO and one dH: DD
                    return result + 4
                elif amount == 2: # two dH's : AD
                    return result + 3
            else: # OH-
                if amount == 0: # one dO and one dH: AA
                    return result + 4
                elif amount == -2: # two dO's : AD
                    return result + 3
                
        elif len(indeces) == 3: # one fold coordinated
            bvv = get_bond_variable_values_from_water_orientation(orientation)
            amount = 0
            for index in indeces:
                amount += bvv[index]
            if result == 0: # neutral
                if amount == 1:
                    return result + 6 # A
                elif amount == -1:
                    return result + 7 # D
            elif result == 8:  # H3O+
                if amount == 3:
                    return result + 5
                elif amount == 1:
                    return result + 6
            else:  # OH-
                if amount == -3:
                    return result + 6
                elif amount == -1:
                    return result + 5

def get_molecule_type_counts(np.int8_t[:, :] results, np.int_t[:, :, :] nearest_neighbors_nos, str folder):
    _get_molecule_type_counts(results, nearest_neighbors_nos, folder)

cdef void _get_molecule_type_counts(np.int8_t[:, :] results, np.int_t[:, :, :] nearest_neighbors_nos, str folder):
    cdef np.int_t[:] water_molecule_types 
    cdef np.int_t[:, :] result = np.zeros((22, results.shape[0]), dtype=np.int)
    cdef np.int_t[:, :] types = np.zeros((results.shape[1], results.shape[0]), dtype=np.int)
    cdef np.int_t i, j, N = results.shape[0], M = results.shape[1], wm_type
    for i from 0 <= i < N:
        water_molecule_types = get_water_molecule_types(results[i], nearest_neighbors_nos, M)
        for j from 0 <= j < M:
            wm_type = water_molecule_types[j]
            result[wm_type, i] += 1
            types[j, i] = wm_type

    cdef dict files = {}
    cdef str write_string, file_name, bl = '\n'
    for i from 0 <= i < 9:
        joinlist = [`num` for num in result[i]]
        write_string = bl.join(joinlist)
        f = get_result_file(molecule_types[i], files, folder, True)                            
        f.write(write_string)
        f.close()
    for i from 0 <= i < M:
        joinlist = [`num` for num in types[i]]
        write_string = bl.join(joinlist)
        f = get_result_file("molecule_%i_type" % i, files, folder, True)                            
        f.write(write_string)
        f.close()
    

cdef inline list get_dangling_bond_indeces(np.int_t molecule_no, np.int_t[:, :, :] nearest_neighbors_nos, np.int_t neighbor_count):
    """
        Gets the indeces of dangling bonds.
    """
    cdef np.int_t i    
    cdef list result = [] 
    for i from 0 <= i < neighbor_count:
        if nearest_neighbors_nos[0, molecule_no, i] == molecule_no:
            result.append(i)
    return result

def get_molecule_types(water_orientations, nearest_neighbors_nos):
    """
        Returns the molecule type numbers in a memoryview 
    """
    return get_water_molecule_types(water_orientations, nearest_neighbors_nos, water_orientations.shape[0]).base

def get_molecule_type_strings(water_orientations, nearest_neighbors_nos):
    """
        Returns the string correspondents of the molecule type numbers in a 
        python list
    """
    cdef list result = []
    cdef np.int_t[:] molecule_type_numbers = get_water_molecule_types(water_orientations, nearest_neighbors_nos, water_orientations.shape[0])
    cdef i, N = molecule_type_numbers.shape[0]
    for i in range(N):
        if molecule_type_numbers[i] == -1:
            result.append("unset")
        else:
            result.append(molecule_types[molecule_type_numbers[i]])
    return result


cdef inline np.int_t[:] get_water_molecule_types(np.int8_t[:] water_orientations, np.int_t[:, :, :] nearest_neighbors_nos, unsigned int N):
    """
        Determines the water molecule types  for each molecul in the given structure 
    """    
    cdef np.int_t[:] result = np.empty(N, dtype=np.int)
    cdef int molecule_type, molecule_no
    for molecule_no from 0 <= molecule_no < N:
        molecule_type = get_water_molecule_type(molecule_no, water_orientations, nearest_neighbors_nos)
        result[molecule_no] = molecule_type
    return result

def get_all_bond_types(np.int8_t[:, :] results, np.ndarray[np.int_t, ndim = 3] nearest_neighbors_nos, str folder):
    _get_all_bond_types(results, nearest_neighbors_nos, folder)


cdef np.int8_t[:] get_allowed_types(np.int8_t[:] water_orientations, np.int_t[:, :, :] nearest_neighbors_nos):
    """
        Determines the allowed molecule types by going through all types and
        returning the ones that occur in the structure
    """
    cdef int i, N = water_orientations.shape[0]
    cdef np.int8_t molecule_type
    cdef list result = []
    for i in range(N):
        molecule_type = get_water_molecule_type(i, water_orientations, nearest_neighbors_nos)
        if molecule_type not in result:
            result.append(molecule_type)
    
    cdef np.int8_t[:] result_np = np.array(result, dtype=np.int8)
    return result_np

cdef np.int_t[:] get_allowed_bond_types(np.int8_t[:] water_orientations, np.int_t[:, :, :] nearest_neighbors_nos):
    """
        Determines the allowed bond types
    """
    cdef np.int8_t[:] allowed_types  = get_allowed_types(water_orientations, nearest_neighbors_nos)
    cdef int i, j, N = allowed_types.shape[0], count = 0 
    cdef np.int_t[:] result = np.empty(N*N+1, dtype=np.int_)
    for i in range(N):
        for j in range(N):
            result[count] = allowed_types[i] * 22 + allowed_types[j]
            count += 1
    result[N*N] = 22*22 # dangling bond index
    return result

@cython.boundscheck(True)
cdef void _get_all_bond_types(np.int8_t[:, :] results, np.int_t[:, :, :] nearest_neighbors_nos, str folder) except * :
    cdef dict files = {}
    cdef dict strings = {}
    cdef np.int8_t[:] wo
    cdef int i, j, k, l
    cdef np.int_t totalp2,  bi = 0, bi2 = 0, m, N = results.shape[0], total, M, O = results.shape[1], P, key, num, chunk_size = 100, Q, type_j, type_k
    cdef np.int_t[:, :, :] result
    cdef np.int_t[:, :, :] counts_2 
    
    cdef np.int_t[:, :] counts
    cdef BufferItem b
    M = 22*22 + 1
    P = 4
    cdef str write_string, file_name, bl = '\n'
    cdef list joinlist
    #cdef list result_buffer = []
    cdef np.int_t[:] allowed_types
    cdef np.ndarray[np.int_t, ndim=2] first_order = np.zeros((M, chunk_size), dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=4] second_order = np.zeros((M, M, P+1, chunk_size), dtype=np.int)
    cdef bint init
    #cdef np.ndarray[BufferItem] result_buffer = np.empty(1000*(M*M*P+1),
    #    dtype=np.dtype([('order', np.int), ('i', np.int),('j', np.int), ('k', np.int), ('value', np.int)]))
    s = time()
    load_time = 0.0
    cdef list all_allowed_types = []
    for i in xrange(N):
        if i < N:
            wo = results[i]
            allowed_types = get_allowed_bond_types(wo, nearest_neighbors_nos)
            Q = allowed_types.shape[0]
            if wo is not None:
                counts =  np.zeros([M, 2], dtype=np.int)
                result = get_bond_types(wo, nearest_neighbors_nos, O, counts)
                counts_2 = get_second_order_bond_types(wo, nearest_neighbors_nos, O, result)
                for key in range(Q):
                    first_order[allowed_types[key], bi] = counts[allowed_types[key], 0]   
                for j in xrange(Q):
                    type_j = allowed_types[j]
                    if type_j not in all_allowed_types:
                        all_allowed_types.append(type_j)
                    for k in xrange(Q):
                        type_k = allowed_types[k]
                        total = 0
                        for l in xrange(P):
                
                            second_order[type_j, type_k, l, bi] = counts_2[type_j, type_k, l]
                            total += counts_2[type_j, type_k, l]
                        totalp2 = total / 2
                        second_order[type_j, type_k, 4, bi] = totalp2
                bi += 1

        Q = len(all_allowed_types)
        # Write results to files if i is a multiple of 'chunk_size' or it is the last item
        if (i + 1) % chunk_size == 0 or i == N -1:
            print "%i/%i" % (i, N)
            if i == N -1:
                first_order = first_order[:, :i%chunk_size]
            if i != 0:
                init = (i + 1)  == chunk_size
                for j in xrange(Q):
                    type_j = all_allowed_types[j]
                    file_name = get_bond_type_string(type_j)
                    #print "HERE0-" 
                    if file_name is not None:
                        joinlist = [`num` for num in first_order[type_j]]
                        
                        write_string = bl.join(joinlist)
                        write_string += bl
        
                        #print "HERE2" 
                        f = get_result_file(file_name, files, folder, init)
                        f.write(write_string)
                        f.close()
                    
                    
                    for k in xrange(Q):
                        type_k = all_allowed_types[k]
                        if is_a_valid_second_order_bond(type_j, type_k):
                            file_name = get_second_order_bond_type_string(type_j, type_k)
 
                            if file_name is not None:
                                for l in xrange(P+1):                              
                                    if i == N:
                                        joinlist = [`num` for num in second_order[type_j, type_k, l, :i%chunk_size]]
                                        second_order[type_j, type_k, l, :i%chunk_size] = 0
                                    else:
                                        joinlist = [`num` for num in second_order[type_j, type_k, l]]
                                    write_string = bl.join(joinlist)
                                    write_string += bl
                                   
                                                           
                                    if l != 4:
                                        f = get_result_file(file_name+"_%i" % l, files, folder, init)
                                    else:
                                        f = get_result_file(file_name, files, folder, init)
                                    
                                    f.write(write_string)
                                    f.close()
                bi = 0
                all_allowed_types = []   
            print "Time used %f s" % (time() - s)
            s = time()
                
                        
    close_files(files)

cdef np.int_t  btm[16]
#cdef np.int_t[:, :] bond_type_matrix = np.array([[0, 1, 2, 14], [3, 4, 5, 15], [6, 7, 8, 16], [10, 11, 12, 13]], dtype=np.int)

def get_first_order_bond_types(water_orientations, nearest_neighbors_nos):
    counts = np.zeros([22*22+1, 2], dtype=int)
    result = get_bond_types(water_orientations, nearest_neighbors_nos, len(water_orientations), counts)
    return result, counts

def get_first_order_bond_type_strings(water_orientations, nearest_neighbors_nos):
    counts = np.zeros([22*22+1, 2], dtype=int)
    bond_types = get_bond_types(water_orientations, nearest_neighbors_nos, len(water_orientations), counts)
    cdef i, j, N = water_orientations.shape[0]
    result = np.empty(N,dtype=object)
    for i in range(N):
        string_list = []
        for j in range(4):
            string_list.append(get_bond_type_string(bond_types[i, j, 0]))
        result[i] = string_list
    return result

cdef np.int_t[:, :, :] get_bond_types(np.int8_t[:] water_orientations, np.int_t[:, :, :] nearest_neighbors_nos, int N, np.int_t[:, :] counts):
    """
            Bond types:
                 See get_water_molecule_types for molecule type numbers
                 bond type numbers are formed as 9 * (bond_type of first molecule) + bond_type of second_molecule
                 the bond type of dangling bond is 22*22
    """
    
    cdef np.int_t type_a, type_b, molecule_no, M, bt
    cdef np.uint8_t nn
    cdef int type_count = 22
    cdef int db_index = type_count * type_count
    cdef bint periodic
    cdef np.int8_t[:] bvv
    cdef np.int_t[:] water_molecule_types = get_water_molecule_types(water_orientations, nearest_neighbors_nos, N)
    cdef np.int_t[:, :, :] result = np.zeros((N, nearest_neighbors_nos.shape[2], 2), dtype=np.int)
    # counts contains the number of bond types
    #  1:st slot contains the number of donor types ie AAD*-AAD where the one marked with star is doning
    #      the hydrogen bond        
    #  2:nd slot contains the number of acceptor types ie AAD-AAD*
    M = 4 # nearest_neighbors_nos[0, molecule_no].shape[0]
    for molecule_no in xrange(N):
        type_a = water_molecule_types[molecule_no]
        if type_a != -1 :
            bvv = get_bond_variable_values_from_water_orientation(water_orientations[molecule_no])
            for i in xrange(M):
                nn = nearest_neighbors_nos[0, molecule_no, i]
                periodic = nearest_neighbors_nos[2, molecule_no, i] != 13
                # IF a dangling bond
                if nn == molecule_no and not periodic:
                    result[molecule_no, i, 0] = db_index
                    if bvv[i] == 1:
                        counts[db_index, 1] += 1
                    else:
                        counts[db_index, 0] += 1
                else:
                    type_b = water_molecule_types[nn]
                    if type_b != -1:
                        bt = type_a * type_count + type_b
                        result[molecule_no, i, 0] = bt
                        if bvv[i] == 1:
                            counts[bt, 0] += 1
                            result[molecule_no, i, 1] = 0
                        elif bvv[i] == -1:
                            counts[bt, 1] += 1 
                            result[molecule_no, i, 1] = 1
       
        
    return result

cdef np.int_t[:, :] combinations = np.array([[0, 1], [2, 3]], dtype=int)

cdef np.int_t[:, :, :] get_second_order_bond_types(np.int8_t[:] water_orientations, np.int_t[:, :, :] nearest_neighbors_nos, np.int_t N, np.int_t[:, :, :] bond_types):
    """
        In the following 
             A: AADD - AADD
             B: AADD - AAD
             C: AADD - ADD
              
             D: 1 - 0 : AAD - AADD 
             E: 1 - 1 : AAD - AAD
             F: 1 - 2 : AAD - ADD
             
             G: 2 - 0 : ADD - AADD
             H: 2 - 1 : ADD - AAD
             I: 2 - 2 : ADD - ADD
        Result is a matrix that has the counts of different kind of bond combinations
       from a starting molecule

        Bond combination types (counts second slot)
            0: 0 - 0 (both molecule_no and nn are donors)
            1: 0 - 1 (nn is double acceptor)
            2: 1 - 0 (nn is double donor)
            3: 1 - 1 (both are acceptors)
    """
    cdef np.int_t[:, :, :] counts
    cdef int type_count = 22
    cdef int db_index = type_count * type_count
    counts = np.zeros((db_index+1, db_index+1, 4), dtype=np.int)
    cdef int molecule_no, i, j, acceptor_a, acceptor_b, type_a, type_b, M
    cdef int nn, nn_2, periodic, periodic_2, axis, axis_2

    #result = np.zeros((len(water_orientations), len(self.water_algorithm.nearest_neighbors_nos[0])))
    M = 4 # nearest_neighbors_nos.base[0, 0].shape[0]
    for molecule_no in xrange(N):
        for i in xrange(M):
            nn = nearest_neighbors_nos[0, molecule_no, i]
            periodic = nearest_neighbors_nos[1, molecule_no, i]
            axis = nearest_neighbors_nos[2, molecule_no, i]
            type_a = bond_types[molecule_no, i, 0]
            # if molecule_no is acceptor
            acceptor_a = bond_types[molecule_no, i, 1]
            for j in xrange(M):
                #if is_a_valid_second_order_bond(i, j):
                nn_2 = nearest_neighbors_nos[0, nn, j]
                axis_2 = nearest_neighbors_nos[2, nn, j]
                periodic_2 = nearest_neighbors_nos[1, nn, j]
                type_b = bond_types[nn, j, 0]
                    
                #raw_input()
                # if nn is acceptor
                acceptor_b = bond_types[nn, j, 1]
                # Don't count circular bond combinations
                if (molecule_no != nn or periodic) and (molecule_no != nn_2 or axis != get_opposite_periodicity_axis_number(axis_2)) and (nn != nn_2 or periodic):
                    #print get_second_order_bond_type_string(type_a, type_b), sum(counts[type_a, type_b]) + 1
                    #raw_input()
                    counts[type_a, type_b, combinations[acceptor_a, acceptor_b]] += 1
                    #print counts[type_a, type_b, combinations[acceptor_a, acceptor_b]]
                    #raw_input()
                        

    #assert counts[3] == counts[1]
    #assert counts[5] == counts[7]
    #assert counts[2] == counts[6]  
    #print counts.base
    #print bond_types.base
    #raw_input()
    return counts

cdef int get_dangling_bond_maximum(self, Invariant dangling_bond_invariant, np.int8_t[:, ::1] bond_variables, np.int_t handled_dangling_bonds):
    """
        Returns the count of the bigger of dangling oxygens and dangling hydrogens
            if the amount of dangling oxygens is bigger then the result value is negative
        handled_three_coordinated: how many three coordinated molecules have been handled up to this point
    """
    cdef np.int_t value = dangling_bond_invariant._get_value(<np.ndarray[np.int8_t, ndim=2]>bond_variables.base)
    cdef np.int_t multiplier = 1    
    if value < 1:
        return value - (handled_three_coordinated + value) / 2
    else:
        return value + (handled_three_coordinated - value) / 2


#from libc.stdlib cimport malloc, free
#cdef char **string_buf = <char**>malloc(18 * sizeof(char*))
cdef list strings = ["AADD_AADD", "AADD_AAD", "AADD_ADD", "AAD_AADD", "AAD_AAD", "AAD_ADD", "ADD_AADD", "ADD_AAD", "ADD_ADD", "DB", "DDD_AADD", "DDD_AAD", "DDD_ADD", "DDD_DDD", "AADD_DDD", "AAD_DDD", "ADD_DDD", "DDD_DDD"] 
cdef list fp_strings = ["AADD", "AADD", "AADD", "AAD", "AAD", "AAD", "ADD", "ADD", "ADD", "DB", "DDD", "DDD", "DDD", "DDD", "AADD", "AAD", "ADD", "DDD"] 
#cdef list bond_types = ["AADD_AADD", "AAD_AAD", "AADD_ADD", "AAD_AADD", "AAD_AAD", "AAD_ADD", "ADD_AADD", "ADD_AAD", "ADD_ADD", "DB", "DDD_AADD", "DDD_AAD", "DDD_ADD", "DDD_DDD", "AADD_DDD", "AAD_DDD", "ADD_DDD", "DDD_DDD"] 

cdef list molecule_types = ["AADD", "AAD", "ADD", "AD", "AA", "DD", "A", "D", "ADDD", "ADD", "DDD", "AD", "DD", "A", "D", "AAAD", "AAA", "AAD", "AD", "AA", "A", "D"]


cdef class BufferItem(object):
    cdef char* name
    cdef np.int_t value

    def __cinit__(self, char* name, np.int_t value):
        self.name = name
        self.value = value
   
cdef inline str get_bond_type_string(np.int_t type_no):
    if type_no == 22*22:
        return "DB"
    return molecule_types[int(type_no / 22)]+"_"+molecule_types[type_no % 22]


cdef inline bint is_a_valid_second_order_bond(np.int_t j, np.int_t k):
    """
        Checks if the two bond values have the same end and start values.
            For example if bond type j is AAD-AAD the bond type k must start with AAD 
    """
    if j % 9 > 8 or k / 9 > 8:
        return False
    cdef str end_first = molecule_types[j % 22], start_second = molecule_types[k / 22]
    return end_first == start_second

cdef inline str get_second_order_bond_type_string(np.int_t j, np.int_t k):
    cdef str second_bond = get_bond_type_string(k)
    cdef str fp_string
    if j != 22*22:
        fp_string = molecule_types[int(j / 22)]
    else:
        fp_string = "DB"
    if fp_string == 'DB' or second_bond == 'DB':
        return None
    return fp_string + "_" + second_bond 

cdef void save_results(np.ndarray[object] buffer_items, dict files, str folder, np.int_t count):
    cdef BufferItem b
    cdef np.int_t i
    cdef dict contents = {}
    cdef list content
    cdef bytes name
    cdef str value, bl = "\n"
    s = time()
    for i from 0 <= i < count:
        b = <BufferItem> buffer_items[i]
        name = b.name
        value = str(b.value)
        if name not in contents:
            contents[name] = []
        content = <list>contents[name]
        content.append(value)
   
    print "Time used in concatenation %f s" % (time()-s)
    cdef str content_string
    for key in contents:
        content = <list>contents[key]
        content_string = '\n'.join(content)
        file_f = get_result_file(key, files, folder, True)
        file_f.write(content_string)
    print "Time used in writing and concatenation %f s" % (time()-s)

cdef void save_result(str key, dict files, np.int_t result, str folder, bint init):
    file_f = get_result_file(key, files, folder, int)
    file_f.write("%i\n" % result)
   

cdef get_result_file(str key, dict files, str folder, bint init):
    if False and key in files:
        return files[key]
    else:
        if init:
            mode = "wb"
        else:
            mode = "ab+"
        f = open(folder+key+".data", mode)
        files[key] = f
        return f 


include "help_methods_cython.pyx"

        
        
                

