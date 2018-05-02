from symmetries.symmetry_operation import get_bond_variable_values_from_water_orientation, get_bond_variable_values_from_water_orientations
import numpy, scipy.linalg, math, os, ase
from help_methods import *
from optparse import OptionParser
import symmetries.interface as sym
from structure_commons import StructureCommons

class Classification(StructureCommons):
    def __init__(self, options = None, result_path=None):
        StructureCommons.__init__(self, **vars(options))
        self.result_path = result_path
        self.dangling_bond_invariant = None

    def get_water_molecule_type(self, molecule_no, water_orientations):
        return get_water_molecule_type(molecule_no, water_orientations, self.nearest_neighbors_nos)

    def get_coordination_numbers(self):
        return get_coordination_numbers(self.nearest_neighbors_nos)

    def get_water_molecule_types(self, water_orientations):
        return get_water_molecule_types(water_orientations, self.nearest_neighbors_nos)

    def get_water_molecule_type_strings(self, water_orientations):
        return get_water_molecule_type_strings(water_orientations, self.nearest_neighbors_nos)
    
    def get_molecule_type_counts(self, results):
        try:
            from classification_cython import get_molecule_type_counts
            folder = self.get_data_folder()
            get_molecule_type_counts(results, self.nearest_neighbors_nos, folder)
        except:
            import traceback
            import sys
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            sys.exit("Failed to initialize molecule type counts, exiting.")
            
        

    def get_all_bond_types(self, results):
        try:
            from classification_cython import get_all_bond_types as gab
            folder = self.get_data_folder()
            gab(np.array(results, dtype=np.int8), self.nearest_neighbors_nos, folder)
            return None, None
        except:
            import traceback
            import sys
            print traceback.format_exc()
            print "Falling back to python"
            first_order = {}
            second_order = {}
            for i, wo in enumerate(results):
                if wo != None:
                    counts_2 = self.get_second_order_bond_types(wo)
                    result, counts = self.get_bond_types(wo)
                    for key in range(len(counts)):
                        if key not in first_order:
                            first_order[key] = {}
                        first_order[key][i] = counts[key][0]
                    for j in range(len(counts_2)):
                        if j not in second_order:
                            second_order[j] = {}
                        for k in range(len(counts_2[j])):
                            if k not in second_order[j]:
                                second_order[j][k] = {}
                            total = 0
                            for l in range(len(counts_2[j][k])):
                                if l not in second_order[j][k]:
                                    second_order[j][k][l] = {}
                                second_order[j][k][l][i] = counts_2[j][k][l]
                                total += counts_2[j][k][l]
                            if len(counts_2[j][k]) not in second_order[j][k]:
                                second_order[j][k][len(counts_2[j][k])] = {}
                            second_order[j][k][len(counts_2[j][k])][i] = total / 2
            for key in first_order:
                self.write_dict_to_file(first_order[key], get_bond_type_string(key))
            for j in second_order:
                for k in second_order[j]:
                    for l in second_order[j][k]:
                        bt = get_second_order_bond_type_string(j, k)
                        if bt == None:
                            continue
                        if l == 4:
                            self.write_dict_to_file(second_order[j][k][l], bt)
                        else:
                            self.write_dict_to_file(second_order[j][k][l], bt+"_"+str(l))
        return first_order, second_order

    
        
    def get_bond_type_strings(self, water_orientations):
        return get_bond_type_strings(water_orientations, self.nearest_neighbors_nos)

    def get_bond_types(self, water_orientations):
        try:
            return get_bond_types(water_orientations, self.nearest_neighbors_nos)
        except:
            """
                Bond types:
                     0: AADD - AADD or a dangling bond
                     1: AADD - AAD
                     2: AADD - ADD
                     
                     3: 1 - 0 : AAD - AADD 
                     4: 1 - 1 : AAD - AAD
                     5: 1 - 2 : AAD - ADD
                     
                     6: 2 - 0 : ADD - AADD
                     7: 2 - 1 : ADD - AAD
                     8: 2 - 2 : ADD - ADD
                    
                     9: Dangling bond

                    10: 3 - 0 : DDD - AADD
                    11: 3 - 1 : DDD - AAD
                    12: 3 - 2 : DDD - ADD
                    13: 3 - 3 : DDD - DDD

                    14 - 16   : DDD:s other way around
            """
            bond_type_matrix = [[0, 1, 2, 14], [3, 4, 5, 15], [6, 7, 8, 16], [10, 11, 12, 13]]
            water_molecule_types = self.get_water_molecule_types(water_orientations)
            result = numpy.zeros((len(water_orientations), len(self.nearest_neighbors_nos[0]), 2), dtype=int)
            # counts contains the number of bond types
            #  1:st slot contains the number of donor types ie AAD*-AAD where the one marked with star is doning
            #      the hydrogen bond        
            #  2:nd slot contains the number of acceptor types ie AAD-AAD*
            counts = numpy.zeros((17, 2), dtype=int)
            for molecule_no in range(len(water_orientations)):
                type_a = water_molecule_types[molecule_no]
                if type_a != -1 :
                    bvv = get_bond_variable_values_from_water_orientation(water_orientations[molecule_no])
                    for i, nn in enumerate(self.nearest_neighbors_nos[0][molecule_no]):
                         # IF a dangling bond
                         if nn == molecule_no:
                            result[molecule_no][i][0] = 9
                            if bvv[i] == 1:
                                counts[9][1] += 1
                            else:
                                counts[9][0] += 1
                         else:
                            type_b = water_molecule_types[nn]
                            if type_b != -1:
                                bt = bond_type_matrix[type_a][type_b]
                                result[molecule_no][i][0] = bt
                                if bvv[i] == 1:
                                    counts[bt][0] += 1
                                    result[molecule_no][i][1] = 0
                                elif bvv[i] == -1:
                                    counts[bt][1] += 1 
                                    result[molecule_no][i][1] = 1
           
            
            return result, counts

    def get_second_order_bond_types(self, water_orientations):
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
        bond_types, counts = self.get_bond_types(water_orientations)
        combinations = [[0, 1], [2, 3]]
        #result = numpy.zeros((len(water_orientations), len(self.nearest_neighbors_nos[0])))
        counts = numpy.zeros((17, 17, 4), dtype=int)
        for molecule_no in range(len(water_orientations)):
            for i, nn in enumerate(self.nearest_neighbors_nos[0][molecule_no]):
                type_a = bond_types[molecule_no][i][0]
                # if molecule_no is acceptor
                acceptor_a = bond_types[molecule_no][i][1]
                for j, nn_2 in enumerate(self.nearest_neighbors_nos[0][nn]):
                    type_b = bond_types[nn][j][0]
                    # if nn is acceptor
                    acceptor_b = bond_types[nn][j][1]
                    # Don't count circular bond combinations
                    if molecule_no != nn and molecule_no != nn_2 and (nn != nn_2 or type_b == 9):
                        counts[type_a][type_b][combinations[acceptor_a][acceptor_b]] += 1

        #assert counts[3] == counts[1]
        #assert counts[5] == counts[7]
        #assert counts[2] == counts[6]

        return counts

    def group_by_aad(self, nozeros=False):
        return self.group_results_by_values([], [8, 0], self.load_results())

    def group_by_FF_and_LL(self, nozeros=False):
        return self.collapse_groups(self.group_results_by_values([[4, 0]], [8, 0], self.load_results(nozeros=nozeros)))
    
    def group_by_FF_and_FC(self, nozeros=False):
        """
            group by ADD-ADD bonds and then by ADD-AADD bonds
        """
        groups = self.group_results_by_values([2], 8, self.load_results(nozeros=nozeros))
        return self.collapse_groups(groups)

    def do_default_grouping(self, nozeros=False):
        self.nozeros = nozeros
        # [5, 7, 2], 
        groups = self.group_results_by_values([[2, 0], [4, 4], [8, 8]], [8, 0],  self.load_results(nozeros=nozeros), level=0, levels=[0, 1, 1])
        #groups = self.group_results_by_values([], 0,  self.load_results(nozeros=nozeros), level=2, levels=[])
        groups = self.sum_groups(groups, level=2)
        return self.collapse_groups(groups)

    def do_default_grouping_for_all(self):
        self.nozeros = False
        # [5, 7, 2], 
        groups = self.group_results_by_values([0, [4, 4], [8, 8]], [8, 0],  self.load_results(nozeros=False), level=0, levels=[5, 1, 1])
        groups = self.sum_groups(groups, level=2)
        #groups = self.group_results_by_values([], 0,  self.load_results(nozeros=nozeros), level=2, levels=[])
        #groups = self.sum_groups(groups, level=2)
        return self.collapse_groups(groups)

    def sum_groups(self, groups, level=0):
        """
            Sum groups so that the level and level+1's key's
            are summed together and a new group is formed
        """
        new_groups = {}
        for group_key in groups:
            
            if type(groups[group_key]) == dict:
                subgroups = self.sum_groups(groups[group_key], level = level-1)
                if level == 0:
                    for subgroup_key in subgroups:
                        new_key = group_key + subgroup_key
                        if new_key in new_groups:
                            if type(subgroups[subgroup_key]) == dict:
                                # Does not work
                                new_groups[new_key] = dict(new_groups[new_key].items() + subgroups[subgroup_key].items())
                            else:
                                new_groups[new_key].extend(subgroups[subgroup_key])
                        else:
                            new_groups[new_key] = subgroups[subgroup_key]
                else:
                    new_groups[group_key] = subgroups
            else:
                new_groups[group_key] = groups[group_key]
        return new_groups

    def collapse_groups(self, groups):
        new_groups = {}
        for group_key in groups:
            if type(groups[group_key]) == dict:
                subgroups = self.collapse_groups(groups[group_key])
                for subgroup_key in subgroups:
                    new_key = str(group_key) +", "+ str(subgroup_key)
                    new_groups[new_key] = subgroups[subgroup_key]
            else:
                new_groups[group_key] = groups[group_key]
            
        return new_groups

    def group_by(self, indeces, column_names, data):
        """
            Group results with different column values
                - Recursive method which groups in the order of column_names
                - Creates tree of lists with depth the number of column_names
            
        """
        result = {}
        column_name = column_names[0]
        for i in indeces:
            if type(data[column_name]) == dict:
                # if data is incomplete for the column,
                # we just leave it out
                if i in data[column_name]:
                    # use the value of the field as the 'key' for the dict
                    key = data[column_name][i]
                    # if one with the same value has not been found, use
                    if key not in result:
                        result[key] = []
                    result[key].append(i)
            else:
                # if data is incomplete for the column,
                # we just leave it out
                if len(data[column_name]) > i:
                    key = data[column_name][i]
                    if key not in result:
                        result[key] = []
                    result[key].append(i)
            

        if len(column_names) > 1: 
            for key in result.keys():
                result[key] = self.group_by(result[key], column_names[1:], data)

        return result

    def group_results_by_values(self, values, value, results, levels = None, level = 0, numbers = None, data = None):
        """
            values: values grouped with in the next iterations
            value: an array or int containing value(s) grouped with in current iteration (value meaning bond type value)
            results: water orientations grouped

            level: at which result the grouping is done
                0: bond type value (SEE: get_bond_types(self, wo))
                1: second order bond type value (get_second_order_bond_types(self, wo))
                2: cis_factor values
                3: side dipole moment
                4: number of impossible bonds
            levels: the following levels
        """
        result = {}
        new_numbers = {}
        if data == None:
            data = {}
        if (level == 0 and 0 not in data) or (level == 1 and 1 not in data):
            bt1, bt2 = self.get_all_bond_types(results)
            data[0] = bt1
            data[1] = bt2
        if level == 2 and 2 not in data:
            data[2] = self.get_cis_factor_values(self.nozeros)
        if level == 3  and 3 not in data:
            data[3] = self.get_oxygen_atoms_from_coordinates(self.oxygen_coordinates)
        if level == 4  and 4 not in data:
            data[4] = self.solve_impossible_angles()
        if level == 5  and 5 not in data:
            data[5] = self.get_structures_altered()
        for i, wo in enumerate(results):
            if numbers != None:
                no = numbers[i]
            else:
                no = i 
            if level == 0:
                assert len(value) == 2
                bt, counts = self.get_bond_types(wo)        
                key = counts[value[0]][value[1]]
            elif level == 1:
                counts = self.get_second_order_bond_types(wo)
                if len(value) == 2:
                    key = sum(counts[value[0]][value[1]]) / 2
                elif len(value) == 3:
                    key = counts[value[0]][value[1]][value[2]]
            elif level == 2 or level == 3 or level == 4 or level == 5:
                pass
            else:
                raise Exception("No such level %i" % level)
            if level == 2:
                key = numpy.around(data[2][no], decimals=-1)
            elif level == 3:
                dm = self.read_dipole_moment(no)
                if dm == None: 
                    continue
                side_dm = self.get_side_dipole_moment(oxygen_atoms, dm)
                key = numpy.around(numpy.linalg.norm(side_dm), decimals=0)
            elif level == 4:
                key = self.get_impossible_angle_count(wo, data[4])
            elif level == 5:
                key = data[5][no]
            if key not in result:
                result[key] = []
                new_numbers[key] = []
            result[key].append(wo)
            new_numbers[key].append(no)

        if len(values) > 0: 
            for key in result.keys():
                # Possibly error prone
                if levels == None:
                    new_levels = None
                    new_level = 0
                else:
                    new_level = levels[0]
                    new_levels = levels[1:]
                new_numbers[key] = self.group_results_by_values(values[1:], values[0], result[key], numbers = new_numbers[key], level=new_level, levels=new_levels, data=data)

        return new_numbers
     
    def get_minimum_index(self, items):
        minimum = None
        index = -1
        for i, item in enumerate(items):
            if minimum == None or item < minimum:
                minimum = item 
                index = i
        return index

    def get_maximum_index(self, items):
        maximum = None
        index = -1
        for i, item in enumerate(items):
            if maximum == None or item > maximum:
                maximum = item 
                index = i
        return index

    def get_side_dipole_moment(self, oxygen_atoms, dipole_moment):
        """ 
            Gets the dipole moment to the other than the main axis 
            direction
            -oxygen_atoms : Atoms object containing oxygen atoms
            -dipole_moment : array that contains the dipole moment in x, y and z directions
        """
        evals, evecs = sym.get_moments_of_inertia(oxygen_atoms)
        # The dipole moment is probably biggest along the smallest moment of inertia
        main_axis = evecs[self.get_minimum_index(evals)]
        axis = numpy.identity(3)
        main_dm = []
        for i, ax in enumerate(axis):
            dm = numpy.dot(main_axis, ax) * dipole_moment[i]
            main_dm.append(dm)
        if numpy.linalg.norm(dipole_moment - numpy.array(main_dm)) > numpy.linalg.norm(dipole_moment):
            return dipole_moment + numpy.array(main_dm)
        return dipole_moment - numpy.array(main_dm)
        

    def get_cis_factor_values(self):
        wos = self.load_results()
        result = {}
        cis_factors = self.get_cis_factors()
        for i, wo in enumerate(wos):
            bvv = get_bond_variable_values_from_water_orientations(wo)
            total_value = 0.0
            for f in cis_factors:
                molecule_1 = int(f[0])
                molecule_2 = int(f[2])
                bond_1 = int(f[1]) 
                bond_2 = int(f[3])
                value = f[4]
                # Don't add the values twice
                if molecule_1 < molecule_2:
                    if bvv[molecule_1][bond_1] == bvv[molecule_2][bond_2]:
                        if  abs(value) > 0.9:
                            total_value += value
                    else:
                        if  abs(value) > 0.9:
                            total_value -= value
            result[i] = total_value
        self.write_dict_to_file(result, 'cis_factor_value', universal = True)
        return result
                    
                    
                

    def get_cis_factors(self, atoms = None):
        """
            Gets a float factor of how much the bonds are parallel
            for a molecule's bond and its neighbors bonds

            Result array is of form N x 5, where N is the number of bond pairs
            and the 5 columns have the following information
               0: the number of the first molecule
               1: the number of the bond of the first molecule (0-3)
               2: the number of the second molecule
               3: the number of the bond of the second molecule (0-3)
               4: the cis-factor of this bond: 1-np.dot(vector1, vector2), of if the bonds are equivalent or are from the same molecule then 0
        """
        result = numpy.zeros((0, 5))
        if atoms != None:
            oxygen_coordinates = remove_hydrogens(atoms).get_positions()
        else:
            oxygen_coordinates = self.oxygen_coordinates
        bond_vectors = self.get_bond_vectors(oxygen_coordinates)
        for molecule_no in range(len(oxygen_coordinates)):
            for i, nn in enumerate(self.nearest_neighbors_nos[0][molecule_no]):
                bond_vector_1 = bond_vectors[molecule_no][i]
                for j, nn_2 in enumerate(self.nearest_neighbors_nos[0][molecule_no]):
                    for k, nn_3 in enumerate(self.nearest_neighbors_nos[0][int(nn_2)]):
                        bond_vector_2 = bond_vectors[nn_2][k]
                        if nn_3 == molecule_no or molecule_no == nn_2:
                            result = numpy.vstack((result, [molecule_no, i, nn_2, k, 0.0]))
                        else:
                            #print "%s dot %s = %f" % (bond_vector_1, bond_vector_2, numpy.dot(bond_vector_1, bond_vector_2))
                            #if 1-numpy.abs(numpy.dot(bond_vector_1, bond_vector_2)) 
                            result = numpy.vstack((result, [molecule_no, i, nn_2, k, numpy.dot(bond_vector_1, bond_vector_2)]))
                            factor = (numpy.dot(bond_vector_1, bond_vector_2) + 1) / 2
        
                            
        return result

    def get_ring_types(self, size):
        wos = self.get_results()
        rings = self.find_rings(size)
        
        result = {}
        g_result = {}
        non_trans = {}
        for i, wo in enumerate(wos):
            ring_types, g_types = self.get_single_wo_ring_types(wo, size, rings)
            for ring_type in ring_types:
                if ring_type not in result:
                    result[ring_type] = {}
                result[ring_type][i] = ring_types[ring_type]
            for g_type in g_types:
                if g_type not in g_result:
                    g_result[g_type] = {}
                g_result[g_type][i] = g_types[g_type]
            if size == 4:
                non_trans[i] = self.get_non_trans(ring_types)
        for ring_type in result:
            self.write_dict_to_file(result[ring_type], 'ring_type_'+str(size)+"_"+str(ring_type), universal = True)
        for g_ring_type in g_result:
            self.write_dict_to_file(g_result[g_ring_type], 'g_ring_type_'+str(size)+"_"+str(g_ring_type), universal = True)
        if size == 4:
            self.write_dict_to_file(non_trans, 'non_trans', universal = True)
        return result

    def get_non_trans(self, ring_types):
        return ring_types[1] * 2 + ring_types[2]*1

    def get_single_wo_ring_types(self, wo, size, rings):
        
        bvv = water_algorithm.get_bond_variables_2(wo, self.nearest_neighbors_nos)
        water_molecule_types = self.get_water_molecule_types(wo)
        result = {}
        g_result = {}
        for i in range(size+1):
            result[i] = 0
            g_result[i] = 0
        for ring in rings:
            ring_value = 0
            g_ring_value = 0
            aad_count = 0
            add_count = 0
            ring_string = ""
            for i in range(len(ring)):
                if i == len(ring) - 1:
                    ring_value += bvv[ring[i]][ring[0]][13]
                else:
                    ring_value += bvv[ring[i]][ring[i+1]][13]
                if water_molecule_types[ring[i]] == 1:
                    aad_count += 1
                    ring_string += "-AAD"
                elif water_molecule_types[ring[i]] == 2:
                    add_count += 1
                    ring_string += "-ADD"
                else:
                    ring_string += "-AADD"
            ring_value = abs(ring_value)
            g_ring_value = max([aad_count, add_count])
            if (ring_value == 0 or ring_value == 1):
                if bvv[ring[0]][ring[1]][13] == bvv[ring[1]][ring[2]][13] or bvv[ring[1]][ring[2]][13] == bvv[ring[2]][ring[3]][13]:
                    ring_value += 1
            #if ring_value not in result:
            #    result[ring_value] = 0
            g_result[g_ring_value] += 1
            result[ring_value] += 1
        return result, g_result

    def find_rings(self, size):
        """
            Finds rings in the oxygen structure with a circle size of 'size'
                returns the circles as a (N, size) list where N is the number 
                of  found circles and the size contains numbers of oxygen atoms
                forming the ring
        """
        assert size >= 3
        rings = []
        result = []
        # initialize rings
        for n in range(self.nearest_neighbors_nos.shape[1]):
            rings.append([n])
        result = self.do_ring_loop(rings, size)
        result = self.remove_equal_rings(result)
        return result

    
    def remove_equal_rings(self, rings):
        """ 
            Ring that contains the same numbers is the same ring
        """
        results = []
        for ring in rings:
            equivalent_found = False
            for result in results:
                all_found = True
                for no in ring:
                    if no not in result:
                        all_found = False
                        break
                if all_found:
                    equivalent_found = True
            if not equivalent_found:
                results.append(ring)
        return results
                    

    def do_ring_loop(self, rings, size):
        new_rings = []
        ring_size = len(rings[0])
        for ring in rings:
            for nn in self.nearest_neighbors_nos[0][ring[-1]]:
                if len(ring) < size:
                    if nn not in ring:
                        new_ring = list(ring)
                        new_ring.append(nn)
                        new_rings.append(new_ring)
                else:
                    if nn == ring[0]:
                        new_rings.append(ring)
        if ring_size == size:
            return new_rings
        else:
            return self.do_ring_loop(new_rings, size)
    
            
            

    

    def solve_impossible_angles(self, minimum=85, maximum=130):
        """
            Finds the orientation numbers that make the water molecule to have 
            too large bond angles 
        """
        result = {}
        oxygen_coordinates = self.oxygen_coordinates
        bond_vectors = self.get_bond_vectors(oxygen_coordinates)
        for molecule_no in range(len(oxygen_coordinates)):
            orientation_number = 0
            for i, nn in enumerate(self.nearest_neighbors_nos[0][molecule_no]):
                for j, nn2 in enumerate(self.nearest_neighbors_nos[0][molecule_no]):
                    if i < j:
                        angle_rad = numpy.arccos(numpy.dot(bond_vectors[molecule_no][i], bond_vectors[molecule_no][j]))
                        angle = (angle_rad/(2.0*math.pi))*360.0
                        if angle < minimum or angle > maximum:
                            if molecule_no not in result:
                                result[molecule_no] = []
                            result[molecule_no].append(orientation_number)
                        orientation_number += 1
        self.write_dict_to_file(result, 'impossible_angle', universal = True)
        return result

    def get_impossible_angle_count(self, wo, impossible_angles):
        count = 0
        for no in impossible_angles.keys():
            if wo[no] in impossible_angles[no]:
                count += 1
        return count

    def get_impossible_angle_counts(self, wos, minimum=85, maximum=130):
        result = {}
        impossible = self.solve_impossible_angles(minimum=minimum, maximum=maximum)
        for i, wo in enumerate(wos):
            result[i] = self.get_impossible_angle_count(wo, impossible)
        self.write_dict_to_file(result, 'impossible_angle_count', universal = True)
        return result

    def remove_impossible_angles(self, wos, minimum=85, maximum=130):
        impossible = self.solve_impossible_angles(minimum=minimum, maximum=maximum)
        result = {}
        for i, wo in enumerate(wos):
            valid = True
            for no in impossible.keys():
                if wo[no] in impossible[no]:
                    valid = False
                    break
            if valid:
                result[i] = wo
        return result

    def get_has_impossible_angles(self, wos, minimum=85, maximum=130):
        impossible = self.solve_impossible_angles(minimum=minimum, maximum=maximum)
        result = {}
        for i, wo in enumerate(wos):
            valid = True
            for no in impossible.keys():
                if wo[no] in impossible[no]:
                    valid = False
                    break
            if valid:
                result[i] = wo
        return result

    def remove_possible_angles(self, wos, minimum=85, maximum=130):
        impossible = self.solve_impossible_angles(minimum=minimum, maximum=maximum)
        result = {}
        for i, wo in enumerate(wos):
            valid = True
            for no in impossible.keys():
                if wo[no] in impossible[no]:
                    valid = False
                    break
            if not valid:
                result[i] = wo
        return result

                
                     
    def get_oxygen_coordinates_from_atoms(self, atoms):
        coordinates = atoms.get_positions()
        result = []
        for i, atom in enumerate(atoms):
            if atom.get_symbol() == 'O':
                result.append(coordinates[i])
        return result

    def get_oxygen_atoms_from_coordinates(self, oxygen_coordinates):
        N = len(oxygen_coordinates)
        atoms = ase.Atoms("O%i" % N, positions=oxygen_coordinates)
        return atoms

    def get_bond_vectors(self, oxygen_coordinates):
        result = numpy.zeros((len(oxygen_coordinates), len(self.nearest_neighbors_nos[0][0]), 3))
        
        for molecule_no in range(len(oxygen_coordinates)):
            for i, nn in enumerate(self.nearest_neighbors_nos[0][molecule_no]):
                if molecule_no != nn:
                    vector = numpy.array(oxygen_coordinates[molecule_no]-oxygen_coordinates[nn])
                    N = scipy.linalg.norm(vector)
                    vector /= N
                    result[molecule_no][i] = vector
        return result


    def get_oxygen_distances(self, atoms):
        oxygen_coordinates = self.get_oxygen_coordinates_from_atoms(atoms)
        distances = numpy.zeros((len(oxygen_coordinates), len(self.nearest_neighbors_nos[0][0])))
        total_distance = 0.0 
        minimum = 1000.0
        maximum = 0.0
        for i in range(len(oxygen_coordinates)):
            for l, nn in enumerate(self.nearest_neighbors_nos[0][i]):
                distance = numpy.abs(numpy.linalg.norm(oxygen_coordinates[i] - oxygen_coordinates[l]))
                distances[i][l] = distance
                if i < nn:
                    total_distance += distance
                if distance < minimum:
                    minimum = distance
                elif distance > maximum:
                    maximum = distance
        return distances, total_distance, minimum, maximum


    def get_total_oxygen_distance(self, atoms):
        distances, total_distance, minimum, maximum = self.get_oxygen_distances(atoms)
        return total_distance

    def get_total_oxygen_distances_dict(self, atoms_dict):
        """
            atoms_dict: dict object containing ase.Atoms objects

            returns a dict containing total oxygen distances for all given
            Atoms objects
        """
        result = {}
        for number in atoms_dict:
            result[number] = self.get_total_oxygen_distance(atoms_dict[number])
        return result
    
    def do_bond_angle_analysis(self, ideal_angle = 104.4):
        structures = self.read_structures()
        result = {}
        angle_deviations = {}
        vertice_deviations = {}
        for key in structures:            
            atoms = structures[key]
            result[key] = self.do_single_structure_bond_analysis(atoms, ideal_angle)
            angle_deviations[key] = result[key][0]
            vertice_deviations[key] = result[key][1]

        self.write_dict_to_file(angle_deviations, 'angle_deviation', universal = True)
        self.write_dict_to_file(vertice_deviations, 'vertice_deviation', universal = True)
            
        return result 

    def do_single_structure_bond_analysis(self, atoms, ideal_angle=104.4, debug=False):
        angles, vectors = self.get_single_structure_bond_angles_and_vectors(atoms)
        # angle deviation from the ideal water molecule angle
        angle_deviation = sum(numpy.abs(angles-ideal_angle))
        # angle deviation form the O-O vertice line
        vertice_deviation = self.do_angle_analysis(vectors, atoms, debug = debug)
        return [angle_deviation, vertice_deviation]
 
    def do_angle_analysis(self, vectors, atoms, debug = False):
        """
            Calculates the bond angle deviation from corresponding
             O_O vertices for each molecule and sums them
        """
        count = 0
        bond_vectors = self.get_bond_vectors(self.get_oxygen_coordinates_from_atoms(atoms))
        N = self.get_oxygen_atom_count(atoms)
        total_angle_difference = 0.0
        for i in range(N):
            dangling_bond_count = 0
            for vector in vectors[i]:
                closest_vector, dot = self.get_closest_vector(vector, bond_vectors[i])
                
                
                # to get the dangling bonds out, we are using limit value of 0.5
                if dot > 0.5: 
                    angle_rad = numpy.arccos(numpy.dot(vector, closest_vector))
                    count += 1
                    if debug:
                        print vector
                        print closest_vector
                        print abs((angle_rad/(2.0*math.pi))*360.)
                        raw_input()
                    total_angle_difference += abs((angle_rad/(2.0*math.pi))*360.)
                else:
                    dangling_bond_count += 1
            if dangling_bond_count > 1:
                raise Exception('Too many dangling bonds found. Dot product tolerance is propably too big!')
        if debug:
            print "Total angle difference of %i bonds %f" % (count, total_angle_difference)        
        return total_angle_difference / count

    def get_oxygen_atom_count(self, atoms = None):
        if atoms is not None:
            count = 0
            for atom in atoms:
                if atom.get_symbol() == 'O':
                    count += 1
            return count
        else:
            return len(self.oxygen_coordinates)
            
                

    def get_closest_vector(self, vector, vector_list):
        """
            Gets the nearest vector for the 'vector' variable from given 'vector_list' 
            also returns the dot product value of the closest vector
        """
        maximum_value = -2
        maximum_index = 0
        for i, vector_2 in enumerate(vector_list):
            value = numpy.dot(vector, vector_2)
            if maximum_value < value:
                maximum_value = value
                maximum_index = i
        return vector_list[maximum_index], maximum_value
            
            

    def get_single_structure_bond_angles_and_vectors(self, atoms):
        """
            Gets the bond angles and bond vectors for each O-H bond
            in the atoms object
        """
        N = self.get_oxygen_atom_count(atoms)
        result = numpy.zeros(N)
        vectors = numpy.zeros((N, 2, 3))
        for i, atom in enumerate(atoms):
            if atom.get_symbol() == 'O':
                hs = self.find_nearest_hydrogens(atoms, i)
                for j, h in enumerate(hs):
                    vector = numpy.array(atoms.get_positions()[i]-atoms.get_positions()[h])
                    vector /= scipy.linalg.norm(vector)
                    vectors[i][j] = vector

                if len(hs) == 2:
                    angle_rad = numpy.arccos(numpy.dot(vectors[i][0], vectors[i][1]))
                    angle = (angle_rad/(2.0*math.pi))*360.0
                    
                    #vectors[i][1] = vector_2
                    result[i] = angle

        return result, vectors
            
            

    def find_nearest_hydrogens(self, atoms, index):
        """ 
            Returns the indeces of nearby hydrogens for atom with index 'index'
        """
        N = self.get_oxygen_atom_count(atoms)
        result = []
        for i, atom in enumerate(atoms):
            if i != index and atom.get_symbol() == 'H':
                distance = scipy.linalg.norm(atom.get_position()-atoms[index].get_position())
                if distance < 1.3 and distance > 0.7:
                    result.append(i)
        return result
                        

    def run(self):
        if self.result_path != None:
            if os.path.exists(self.result_path):
                atoms = ase.io.read(self.result_path)
                result = self.do_single_structure_bond_analysis(atoms)
                print "Angle deviation from the ideal water molecule angle %f" % result[0]
                print "Angle deviation form the O-O vertice line %f" % result[1]
            else:
                print "Invalid path! (%s)" %  (self.result_path)
        else:
            print "Please specify path with -p or --result_path argument."


def get_water_molecule_type(self, molecule_no, water_orientations, nearest_neighbors_nos):
    
    """
        Return if the molecule is an acceptor or donor
        -1 : not set
         0 : AADD
         1 : AAD
         2 : ADD
         3 : DDD
    """
    # FIXME: When doing at middle of execution, returns false numbers
    # Find the indeces of dangling bonds
    if water_orientations[molecule_no] == -1:
        return -1;
    else:
        bvv = get_bond_variable_values_from_water_orientation(water_orientations[molecule_no])
        index = numpy.where(nearest_neighbors_nos[0][molecule_no]==molecule_no)[0]
        if len(index) == 0:
            return 0
        if water_orientations[molecule_no] > 9:
            return 3
        else:
            if bvv[index] == -1:
                return 2
            else:
                return 1

def get_coordination_numbers(nearest_neighbors_nos):
    try:
        from classification_cython import get_coordination_numbers
        return get_coordination_numbers(nearest_neighbors_nos)
    except:
        import traceback
        import sys
        traceback.print_exc()
        return None
        
        
def get_bond_types(water_orientations, nearest_neighbors_nos):
    from classification_cython import get_first_order_bond_types as gbt
    result, counts = gbt(water_orientations, nearest_neighbors_nos)
    return result, counts
    
def get_bond_type_strings(water_orientations, nearest_neighbors_nos):
    try:
        from classification_cython import get_first_order_bond_type_strings as gbt
        result = gbt(water_orientations, nearest_neighbors_nos)
        return result
    except:
        import traceback
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        return None
        
        
        
def get_water_molecule_types(self, water_orientations):
    try:
        from classification_cython import get_molecule_types
        return get_molecule_types(water_orientations, self.nearest_neighbors_nos)
    except:
        import traceback
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)

    return result

def get_water_molecule_type_strings(water_orientations, nearest_neighbors_nos):
    try:
        from classification_cython import get_molecule_type_strings
        return get_molecule_type_strings(water_orientations, nearest_neighbors_nos)
    except:
        import traceback
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        result = numpy.zeros(len(water_orientations), dtype=int)
        for molecule_no in range(len(water_orientations)):
            molecule_type = get_water_molecule_type(molecule_no, water_orientations, nearest_neighbors_nos)
            result[molecule_no] = molecule_type

    return result

def get_bond_type_string(type_no):
    result = ["AADD_AADD", "AAD_AAD", "AADD_ADD", "AAD_AADD", "AAD_AAD", "AAD_ADD", "ADD_AADD", "ADD_AAD", "ADD_ADD", "DB", "DDD_AADD", "DDD_AAD", "DDD_ADD", "DDD_DDD", "AADD_DDD", "AAD_DDD", "ADD_DDD", "DDD_DDD"]    
    return result[type_no]
    

def get_second_order_bond_type_string(j, k):
    first_bond = get_bond_type_string(j)
    second_bond = get_bond_type_string(k)
    if first_bond == 'DB' or second_bond == 'DB':
        return None
    return first_bond[:first_bond.index("_")+1] + second_bond 

def get_result_file(key, files, folder):
    if key in files:
        return files[key]
    else:
        f = open(folder+key+".data", "w")
        files[key] = f
        return f 

def close_files(dict_files):
    for key in dict_files:
        dict_files[key].close()

def save_result(key, files, result, folder):
    file_f = get_result_file(key, files, folder)
    file_f.write("%i\n" % result)


def main():
    parser = OptionParser()
    parser.add_option("-p", "--result_path", type="string", dest="result_path",
                      help="The path (URI) to structure in ase readable format", metavar="URI")
    (options, args) = parser.parse_args()
    Classification(**vars(options)).run()

if __name__ == "__main__":
    main()
        
        
        
                

