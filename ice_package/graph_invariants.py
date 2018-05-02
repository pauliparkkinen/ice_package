import numpy as np
import os
from symmetries.symmetry_operation import get_opposite_periodicity_axis_number

def find_graph_invariants(test_set, water_algorithm):
	symmetry_lists = []
	for example in examples:
		symmetries = water_algorithm.get_symmetries()
		symmetry_lists.append(symmetries)

def print_invariants(invariant_list):
    for i, invariant in enumerate(invariant_list):
        print "#"+str(i)+": "+str(invariant)

def generate_invariants(symmetry_operations,  nearest_neighbors_nos, second_order=True, third_order=False):
    
    
    # invariants for each first order bond
    result = []
    generate_first_order_invariants(result, nearest_neighbors_nos,  symmetry_operations)
    if second_order:
        generate_second_order_invariants(result, nearest_neighbors_nos,  symmetry_operations)
    if third_order:
        generate_third_order_invariants(result, nearest_neighbors_nos,  symmetry_operations)
    #print "Initially %i invariants." % len(invariants)
    
    print "Result count er %i" % len(result)
    result = remove_empty_invariants(result)
    print "Result count ei %i" % len(result)
    result = remove_equal_invariants(result)
    print "Finally %i invariants" % len(result)
    return result
    
def generate_first_order_invariants(result,  nearest_neighbors_nos,  symmetry_operations):
    for i in range(len(nearest_neighbors_nos[0])):
        for n,  nn in enumerate(nearest_neighbors_nos[0][i]):
            periodicity_axis = nearest_neighbors_nos[2][i][n]
            if i < nn or (i == nn and periodicity_axis <= 13):
                invariant_terms = []
                
                for symmetry_operation in symmetry_operations:
                    invariant_terms.append(InvariantTerm([symmetry_operation.symbolic_bond_variable_matrix[i][nn][periodicity_axis]]))
                ps = ""
                if periodicity_axis != 13: ps = "_%i" % periodicity_axis 
                invariant = Invariant(invariant_terms, name='I_{%i, %i%s}' % (i, nn, ps))
                invariant.reduce()
                invariant.order()
                result.append(invariant)

def generate_second_order_invariants(result,  nearest_neighbors_nos,  symmetry_operations):
    #invariants for each second order bond
    for i in range(len(nearest_neighbors_nos[0])):
        for n,  nn in enumerate(nearest_neighbors_nos[0][i]):
            periodicity_axis = nearest_neighbors_nos[2][i][n]
            if i < nn or (i == nn and periodicity_axis <= 13):
                
                
                for i2 in range(len(nearest_neighbors_nos[0])):
                    for n2,  nn2 in enumerate(nearest_neighbors_nos[0][i2]):
                        periodicity_axis2 = nearest_neighbors_nos[2][i2][n2]
                        if i2 < nn2 or (i2 == nn2 and periodicity_axis2 <= 13):
                            invariant_terms = []
                            for symmetry_operation in symmetry_operations:
                                
                                invariant_terms.append(InvariantTerm([symmetry_operation.symbolic_bond_variable_matrix[i][nn][periodicity_axis],  symmetry_operation.symbolic_bond_variable_matrix[i2][nn2][periodicity_axis2]]))
                            p1s = ""
                            p2s = ""                            
                            if periodicity_axis != 13: p1s = "_%i" % periodicity_axis 
                            if periodicity_axis2 != 13: p2s = "_%i" % periodicity_axis2
                            
                            invariant = Invariant(invariant_terms, name='I_{%i, %i%s; %i, %i%s}' % (i, nn, p1s, i2, nn2, p2s))
                            invariant.reduce()
                            invariant.order()
                            result.append(invariant)
                            
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
                            

def remove_equal_invariants(invariants):
    """
        Also removes opposite invariants
    """
    result = []
    equal = 0
    opposite = 0
    for i,  invariant in enumerate(invariants):
        add = True
        for invariant_2 in result:
            if invariant == invariant_2 or invariant.is_opposite_of(invariant_2):
                add = False
                if invariant == invariant_2:
                    equal += 1
                else:
                    opposite += 1
                break
        if add:
            result.append(invariant)
    print "Removing %i invariants because they are equal with something else" % equal
    print "Removing %i invariants because they are opposite of other invariant" % opposite
    print_invariants(result);
    raw_input()
    return result
    
def remove_empty_invariants(invariants):
    result = []
    for invariant in invariants:
        if len(invariant.invariant_terms) != 0:
            result.append(invariant)
    return result

def save_invariants(folder, invariants):
    for i, invariant in enumerate(invariants):
        if not os.path.exists(folder+"invariants"):
            os.makedirs(folder+"invariants")
        a, m = invariant.as_array()
        np.save(folder+"invariants/invariant_%i" % (i), a)
        np.save(folder+"invariants/invariant_%i_m" % (i), m)


def load_invariants(folder):
    result = []
    i = 0
    while True:
        try:
            array = np.load(folder+"invariants/invariant_%i.npy" % (i))
            minuses = np.load(folder+"invariants/invariant_%i_m.npy" % (i))
        except IOError:
            if i == 0:
                print "Loading prestored invariants failed. (Re)generating invariants."
            else:
                print "Loaded %i invariants" %i
            break
        # invariant terms are in array
        invariant_terms = []
        for l, row in enumerate(array):
            invariant_terms.append(InvariantTerm(row, minus=minuses[l]))
        result.append(Invariant(invariant_terms))
        i += 1
    return result

def get_invariants(symmetry_operations,  nearest_neighbors_nos, second_order=True, third_order=False, folder="", reload=False):
    if folder[-1] != "/":
        folder = folder + "/"
    result = []
    if not reload:
        result = load_invariants(folder)
    if len(result) == 0:
        result = generate_invariants(symmetry_operations,  nearest_neighbors_nos, second_order=second_order, third_order=third_order)
        save_invariants(folder, result)
    return result

def order_invariants(invariants):
    for invariant in invariants:
        invariant.order()
    

        

    
            
        



class InvariantTerm:
    def __init__(self, indeces, minus = False):
        """
            Index[0] Molecule that bonds
            Index[1] Molecule which to the bond is
            Index[2] The number of periodic axis (13 == nonperiodic)
        """
        # order the indeces so that the smaller index is before and change the sign similarly    
        new_indeces = []
        for index in indeces:
            if index[0] > index[1]:
                new_index_2 = get_opposite_periodicity_axis_number(index[2])
                new_indeces.append([index[1], index[0], new_index_2])
                minus = not minus
                
            else:
                new_indeces.append(index)
        self.indeces = np.array(new_indeces, dtype=int)
        self.minus = minus    

        
        self.order()
        
    def order(self):
        ind = np.lexsort((self.indeces[:, 2], self.indeces[:,  1],  self.indeces[:,  0]))
        self.indeces = np.array([self.indeces[i] for i in ind])

    def __str__(self):
        result = None
        previous_index = None
        mul = 0
        for i, index in enumerate(self.indeces):
            if previous_index == None or (previous_index == index).all():
                mul += 1
            else: 
                if result != None:
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
            previous_index = index

        if result != None:
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
        return result
        
    def is_opposite_of(self,  other_term):
        if len(other_term.indeces) != len(self.indeces):
            return False
        if all(np.equal(other_term.indeces, self.indeces).flatten()) and self.minus != other_term.minus:
            return True
        
        for i,  index in enumerate(self.indeces):
            if (index[0] == index[1] or index[2] != other_term.indeces[i][2] or index[0] != other_term.indeces[i][1] or index[1] != other_term.indeces[i][0]):
                return False
        return True
        
    def get_value(self, bond_variables):
        result = 1 
        for index in self.indeces:
            result = result * bond_variables[index[0]][index[1]][index[2]] 
        if self.minus:
            result *= -1
        return result
    
    
    def __lt__(self, other):
        """Indeces have to be ordered if this is used"""
        for i,  index in enumerate(self.indeces):
            if other.indeces[i][0] != self.indeces[i][0]:
                return self.indeces[i][0] < other.indeces[i][0] 
            elif other.indeces[i][1] != self.indeces[i][1]:
                return self.indeces[i][1] < other.indeces[i][1]
        # They are equal
        return False

    def __gt__(self, other):
        """Indeces have to be ordered if this is used"""
        for i,  index in enumerate(self.indeces):
            if other.indeces[i][0] != self.indeces[i][0]:
                return self.indeces[i][0] > other.indeces[i][0] 
            elif other.indeces[i][1] != self.indeces[i][1]:
                return self.indeces[i][1] > other.indeces[i][1]
        # They are equal
        return False
        
    def __eq__(self,  other):
        if other == None:
            return False
        return len(self.indeces) == len(other.indeces) and all(np.equal(self.indeces, other.indeces).flatten()) and self.minus == other.minus
        
    def __ne__(self,  other):
        return len(self.indeces) != len(other.indeces) or all(np.not_equal(self.indeces, other.indeces).flatten()) or self.minus != other.minus
     
    def as_array(self):
        ind = self.indeces.copy()
        return ind
            

class Invariant:
    def __init__(self, invariant_terms, name=None):
        self.invariant_terms = np.array(invariant_terms)	
        self.invariant_terms = np.sort(self.invariant_terms)
        self.normalizing_constant  = 1. / float(len(invariant_terms))
        self.name = name
        
    def get_value(self, bond_variables):
        result = 0
        #res = ""
        for term in self.invariant_terms:
            #res += " + %i" % term.get_value(bond_variables)
            result = result + term.get_value(bond_variables)
        #result = self.normalizing_constant * result
        #print res
        return result
    
    """def is_invariant(self, test_set, symmetry_lists):
        for i, geometry in enumerate(test_set):
            value = self.get_value(geometry)
            for symmetry in symmetry_lists[i]:
    """
    
    def reduce(self):
        removed_terms = []
        for invariant_term in self.invariant_terms:
            if invariant_term not in removed_terms:
                for invariant_term_2 in self.invariant_terms:
                    if invariant_term_2 not in removed_terms and invariant_term.is_opposite_of(invariant_term_2):
                        removed_terms.append(invariant_term)
                        removed_terms.append(invariant_term_2)
                        break
        new_invariant_terms = []
        for invariant_term in self.invariant_terms:
            if invariant_term not in removed_terms:
                new_invariant_terms.append(invariant_term)
        self.invariant_terms = new_invariant_terms

    def order(self):
        if len(self.invariant_terms) != 0:
            order_list = []
            for term in self.invariant_terms:
                order_list.append(term.indeces.flatten())
            order_list = np.array(order_list)
            ind = np.lexsort((order_list[:,  1],  order_list[:,  0], order_list[:, 2]))
            self.invariant_terms = np.array([self.invariant_terms[i] for i in ind])
        
    def __eq__(self,  other):
        """Terms have to be ordered if this is used"""
        if not hasattr(other, "invariant_terms") or (len(self.invariant_terms) != len(other.invariant_terms)):
            return False
        for i,  invariant_term in enumerate(self.invariant_terms):
            if invariant_term != other.invariant_terms[i]:
                return False 
        return True

    # Opposite invariants are different but do not help the calculation at all
    def is_opposite_of(self, other):
        if not hasattr(other, "invariant_terms") or (len(self.invariant_terms) != len(other.invariant_terms)):
            return False
        for i, invariant_term in enumerate(self.invariant_terms):
            if not invariant_term.is_opposite_of(other.invariant_terms[i]):
                return False
        return True     

    def __str__(self):
        if self.name != None:
            result = self.name +": " 
        else:
            result = ""
        previous_term = None
        pt_mul = 0
        for i, invariant_term in enumerate(self.invariant_terms):
            if previous_term == None or previous_term == invariant_term:
                pt_mul += 1
            else:
                if invariant_term.minus:
                    result += " - "
                elif i != 0:
                    result += " + "
                if pt_mul > 1:
                    result += "%i %s" % (pt_mul, previous_term) 
                else:
                    result += "%s" % (previous_term) 
                pt_mul = 1
            previous_term = invariant_term
        if previous_term.minus:
            result += " - "
        elif i != 0:
            result += " + "
        if pt_mul > 1:
            result += "%i %s" % (pt_mul, previous_term) 
        else:
            result += "%s" % (previous_term) 
        return result

    def as_array(self):
        result = []
        minuses = []
        for invariant_term in self.invariant_terms:
            minuses.append(invariant_term.minus)
            result.append(invariant_term.as_array())
        return np.array(result, dtype=int), np.array(minuses, dtype=int)
        
