import numpy as np
import scipy
import math
import ase
import os
from time import time
from result_group import ResultGroup, merge_groups
from help_methods import *
from symmetries.symmetry_operation import get_bond_variable_values_from_water_orientation, remove_equals, get_vector_from_periodicity_axis_number, get_opposite_periodicity_axis_number, get_water_orientation_from_bond_variable_values
import symmetries.interface as sym

from water_algorithm_cython import get_mpi_variables
comm, rank, size = get_mpi_variables()

class WaterAlgorithm:

    def __init__(self, filename=None):
        """
            File name is the filename having the oxygen structure
        """
        self.filename = filename
        if filename != None:
            self.initialize()
    
    def initialize(self, periodic=False, slab=False, cell=None, store_bond_variables=False, O_H_distance=1.0, O_O_distance=2.70, intermediate_saves=[], folder=None, group_saves=[], charge=0, dissosiation_count=0, do_symmetry_check = None, order = None):
        """
            periodic: is the cell periodic
            cell : the x y z lengths of the cell i an array
            NOTE: needed only if the periodic is True
        """
        self.O_H_distance = O_H_distance
        self.O_O_distance = O_O_distance
        self.intermediate_saves=intermediate_saves
        self.folder = folder
        self.group_saves = group_saves
        self.periodic = periodic
        self.cell=cell
        self.slab = False
        self.charge = charge
        self.dissosiation_count = dissosiation_count
        self.oxygen_coordinates = self.get_all_oxygen_coordinates()
        self.N = len(self.oxygen_coordinates)
        self.store_bond_variables = store_bond_variables   
        self.atoms = ase.Atoms("O%i" % len(self.oxygen_coordinates), positions=self.oxygen_coordinates, cell=self.cell)
        #ase.visualize.view(self.atoms)
        
        self.nearest_neighbors_nos = all_nearest_neighbors_no(self,  self.oxygen_coordinates, periodic=self.periodic, cell=self.cell) 
        if do_symmetry_check == None:
            self.do_symmetry_check = np.ones(len(self.oxygen_coordinates), dtype=int)
        elif do_symmetry_check == False:
            self.do_symmetry_check = np.zeros(len(self.oxygen_coordinates), dtype=int)
        else:
            self.do_symmetry_check = do_symmetry_check
        if order == None:
            self.order = np.arange(0, len(self.oxygen_coordinates), 1, dtype=int) 
        

        
    
    def load_results(self, nozeros=False):
        if os.path.exists(self.get_folder()+"nozero_results.txt") or os.path.exists(self.get_folder()+"allresults.txt"):
            if nozeros:
                wo = np.loadtxt(self.get_folder()+"nozero_results.txt",  dtype="int8")     
            else:
                wo = np.loadtxt(self.get_folder()+"allresults.txt",  dtype="int8")     
        else:
            return None
        return wo 

    def load_invariants(self):
        from graph_invariants import get_invariants, print_invariants
        if rank == 0:
            for symmetry_operation in self.symmetry_operations:
                symmetry_operation.get_symbolic_bond_variable_matrix(self.nearest_neighbors_nos, self) 
            gi = get_invariants(self.symmetry_operations,  self.nearest_neighbors_nos, folder=self.get_folder())[:55]
        else:
            gi = None
        
        #print_invariants(gi)  
        if size > 1:      
            self.graph_invariants = comm.bcast(gi,  root=0)
        else:
            self.graph_invariants = gi

    def run(self, i=None, invariant_level=0, group=None):
        s = time()
        self.initialize_symmetry_operations()
        self.load_invariants()
        
        water_orientations = self.perform_2(order=self.order, do_symmetry_check=self.do_symmetry_check, i=i, invariant_level=invariant_level, group=group)
        self.print_time_stats()
        self.conversion_time += time() - s
 

    def initialize_symmetry_operations(self):
        if rank == 0:
            so = sym.find_symmetry_operations(get_oxygens(self.oxygen_coordinates, periodic = self.periodic, slab = self.slab, cell = self.cell))
            so = remove_equals(so)
        else:
            so = None
        if size > 1:
            self.symmetry_operations = comm.bcast(so,  root=0)
        else:
            self.symmetry_operations = so

    def get_symmetries(self, result,  neareast_neighbors_nos):
        symmetries = np.zeros((0,len(result)))
        symmetrynames = []
        for symmetry_operation in self.symmetry_operations:
            symmetry = symmetry_operation.apply(result,  self, neareast_neighbors_nos)
            #print symmetry
            if symmetry != None:
                symmetries = np.vstack((symmetries,  np.array(symmetry)))
                symmetrynames.append(symmetry_operation.name)
                
        #print "Symmetries count %i" % len(symmetries) 
        return symmetries,  symmetrynames

    

    def get_hydrogen_coordinates(self, oxygen_positions, water_orientations, nearest_neighbors_nos,  symmetry_operation=0):
        result = np.zeros((0, 3))
        site = 1
        print water_orientations
        for i in range(len(oxygen_positions)):
            result = np.vstack((result, self.get_single_molecule_hydrogen_coordinates(site, water_orientations[i], i, oxygen_positions,  nearest_neighbors_nos[0][i], nearest_neighbors_nos[1][i], nearest_neighbors_nos[2][i], self.cell)))
            if site == 4:
                site = 1
            else:
                site = site +1
        return result

    

    def sort_nearest_neighbors(self, molecule_no, sortlist,  result):
        """
            Deprecated
        """
        return None


    def print_time_stats(self):
        time_now = time() 
        if hasattr(self, 'time_start'):
            self.symmetry_total_time = gather_and_max(self.symmetry_total_time)
            self.conversion_time = gather_and_max(self.conversion_time)
            self.result_group_tryout_time = gather_and_max(self.result_group_tryout_time)
            self.symmetry_load_time = gather_and_max(self.symmetry_load_time)
            self.symmetry_check_time = gather_and_max(self.symmetry_check_time)
            self.iteration_time = gather_and_max(self.iteration_time)
        #Follow the time used
            if rank == 0:
                print "-------------------------------------------------------------------------"
                print "Total time elapsed %f s" % (time_now-self.time_start)
                print "  Total time elapsed in symmetry check %f s" % self.symmetry_total_time
                print "    Time used in bond variable conversion %f s" % self.conversion_time
                print "    Time used trying the result groups %f s" % self.result_group_tryout_time
                print "    Time used in symmetry loading %f s" % self.symmetry_load_time
                print "    Time used in symmetry checking %f s" % self.symmetry_check_time
                print "  Time used in the iteration of water orientations %f s" % self.iteration_time
                print " ------------------------------------------------------------------------"
        self.time_start = time_now 
        self.symmetry_load_time = 0
        self.symmetry_check_time = 0
        self.iteration_time = 0
        self.symmetry_total_time = 0
        self.result_group_tryout_time = 0
        self.conversion_time = 0

    def get_folder(self):
        if self.folder == None:
            folder = "./"
        if self.folder != None:
            folder = self.folder+"/"
        else:
            folder = ""
        return folder

    def save_results(self, i, water_orientations, order, group_save=False, group_number=-1):
        """
            If i is in intermediate saves save only a text file containing water orientations
                -if not, then do nothing
            If i is the last molecule to be handled, then save the water_orientations also
        """
        if rank == 0:
            folder = self.get_folder()
            if i in self.intermediate_saves:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                np.savetxt(folder+"intermediate_%i.txt" % (i), water_orientations)

            if group_number != -1 and i in self.group_saves:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                np.savetxt(folder+"group_%i_%i.txt" % (i, group_number), water_orientations)

            elif not group_save and i == order[len(order)-1]:
                if not os.path.exists(folder+"alkuperaiset/"):
                    os.makedirs(folder+"alkuperaiset/")
                write_results_to_file(self,  self.get_all_oxygen_coordinates(), water_orientations, self.cell,  self.nearest_neighbors_nos,  folder=folder+"alkuperaiset/")
                np.savetxt(folder+"allresults.txt", water_orientations)

            


    def perform_2(self, water_orientations=[],  i=None, order = None, do_symmetry_check=None, group=None, invariant_level=0):
        self.print_time_stats()
        nearest_neighbors_nos = self.nearest_neighbors_nos
        if order == None:
            order = np.arange(0, self.N, 1)
        if do_symmetry_check == None:
            do_symmetry_check = np.ones(self.N)
        if i == None:
            i = order[0]
        # Loaded from i certain state
        if i != order[0] and group == None and water_orientations == []:
            if rank == 0:
                print self.get_folder()+"intermediate_%i.txt" % i
                new_water_orientations = np.loadtxt(self.get_folder()+"intermediate_%i.txt" % i)
                new_water_orientations = np.array(new_water_orientations, dtype=int)
            else:
                new_water_orientations = None
            if size > 1:
                new_water_orientations = comm.bcast(new_water_orientations, root = 0)
        else:
            print_parallel("Handling molecule number %i." % i)
            
            self.result_group = ResultGroup(list(self.graph_invariants[invariant_level:]),  0)
            if group == None:
                if (self.charge != 0 or self.dissosiation_count != 0) and i == order[0]:
                    water_orientations = self.handle_charges()
                    self.result_group = ResultGroup(list(self.graph_invariants[invariant_level:]),  0)
                new_water_orientations = np.zeros((0, self.N),  dtype="int8")
                
                s = time()
                discarded = 0
                if len(water_orientations) == 0:
                    if rank == 0:
                        water_orientations = np.empty((len(nearest_neighbors_nos[0])), dtype=int)
                        water_orientations.fill(-1)
                        new_water_orientations, d = self.handle_molecule_algorithm_2(nearest_neighbors_nos, water_orientations, i, new_water_orientations)
                        discarded += d
                    if size > 1:
                        new_water_orientations = comm.bcast(new_water_orientations,  root = 0)
                else:
                    if size > 1:
                        water_orientations = comm.scatter(split_list(water_orientations,  size),  root = 0)
                    for l, water_orientation in enumerate(water_orientations):
                        new_water_orientations, d = self.handle_molecule_algorithm_2(nearest_neighbors_nos, water_orientation, i, new_water_orientations)
                        discarded += d
                    if size > 1:
                        new_water_orientations = merge(comm.allgather(new_water_orientations))
                discarded = gather_and_sum(discarded)
                print_parallel("Discarded %i geometries due to breakage of ice rules" % (discarded))

                self.iteration_time = (time()-s)
            else:
                if rank == 0:
                    print self.get_folder()+"group_%i_%i.txt" % (i, group)
                    new_water_orientations = np.loadtxt(self.get_folder()+"group_%i_%i.txt" % (i, group))
                    new_water_orientations = np.array(new_water_orientations, dtype=int)
                else:
                    new_water_orientations = None
                if size > 1:
                    new_water_orientations = comm.bcast(new_water_orientations, root = 0)
            
            #Remove the symmetries
            s = time()
            if do_symmetry_check[i] == 1:
                new_water_orientations = self.remove_symmetric_results(new_water_orientations, i) #self.remove_symmetries_no_invariants(new_water_orientations,   nearest_neighbors_nos, i)    
              
                
            self.symmetry_total_time = (time()-s)
            self.save_results(i, new_water_orientations, order)
        
        #Call the next round (i=i + 1) or return the results
        number = 0
        for l,  n in enumerate(order):
            if n == i:
                number = l
                break

        
            
        if(number+1 != len(nearest_neighbors_nos[0])):
            
            return self.perform_2(water_orientations=new_water_orientations, i=order[number+1], order=order, do_symmetry_check=do_symmetry_check)
        else:
            return new_water_orientations

    def get_coordination_number(self, i):
        result = 4
        for j, nn in enumerate(self.nearest_neighbors_nos[0][i]):
            if nn == i and not self.nearest_neighbors_nos[1][i][j]:
                result -= 1
        return result

    def get_sample_charge(self, water_orientations):
        result = 0
        for orientation in water_orientations:
            if orientation > 9:
                result += 1
            elif orientation > 5:
                result -= 1
        return result

    def handle_charges(self):
        H3O_count = self.dissosiation_count + (abs(self.charge) + self.charge) / 2
        OH_count = self.dissosiation_count + (abs(self.charge) - self.charge) / 2
        if OH_count == 0 and H3O_count == 0:
            return np.zeros((0, self.N),  dtype="int8")

        water_orientations = []
        if H3O_count != 0:
            print_parallel("Handling H3O+ molecules")
            print_parallel("---------------------------------------------")
            water_orientations = self.handle_charge(water_orientations, H3O_count, [10, 11, 12, 13])
        if OH_count != 0:
            print_parallel("Handling OH- molecules")
            print_parallel("---------------------------------------------")
            water_orientations = self.handle_charge(water_orientations, OH_count, [6, 7, 8, 9])
        
        return water_orientations

    
        

    def handle_charge(self, water_orientations, charge_left, orientations):
        """
            charge_left means the number of charges particles left to be set
            orientations means the orientations possible for this particle (OH or H30)
        """    
        
        iteration = 1
        while charge_left > 0:
            print_parallel("##    Handling charged particle no %i     ##" % (iteration))
            print_parallel("--------------------------------------------")
            charge_left -= 1
            iteration += 1
            discarded = 0
            new_water_orientations = np.zeros((0, self.N),  dtype="int8")
            if len(water_orientations) == 0:
                if rank == 0:
                    water_orient = np.empty((len(self.nearest_neighbors_nos[0])), dtype=int)
                    water_orient.fill(-1)
                    new_water_orientations, discarded = self.handle_single_charge(water_orient, new_water_orientations, orientations)
                    if size > 1:
                        new_water_orientations = comm.bcast(new_water_orientations,  root = 0)
            else:
                if size > 1:
                    water_orientations = comm.scatter(split_list(water_orientations,  size),  root = 0)
                for l, water_orient in enumerate(water_orientations):
                    new_water_orientations, d = self.handle_single_charge(water_orient, new_water_orientations, orientations)
                    discarded += d
                if size > 1:
                    new_water_orientations = merge(comm.allgather(new_water_orientations))
            discarded = gather_and_sum(discarded)
            print_parallel("Discarded %i geometries due to breakage of ice rules" % (discarded))
            water_orientations = self.remove_symmetric_results(new_water_orientations,  -1) #self.remove_symmetries_no_invariants(new_water_orientations,   nearest_neighbors_nos, i)    
            self.print_time_stats()
        return water_orientations

    def handle_single_charge(self, water_orient, water_orientations, orientations):
        discarded = 0
        for i in range(len(self.oxygen_coordinates)):
            coordination_number = self.get_coordination_number(i)
            if water_orient[i] != -1 or (coordination_number == 4 and orientations[0] > 9):
                continue
            for orientation in orientations:
                s = time()
                if self.water_orientation_is_valid(orientation, water_orient, i):
                    water_orientations_trial = np.copy(water_orient)
                    water_orientations_trial[i] = orientation
                    water_orientations = np.vstack((water_orientations,  water_orientations_trial))
                else:
                    discarded += 1
                  
                self.iteration_time += time()-s
        return water_orientations, discarded

    def water_orientation_is_valid(self, water_orientation, water_orient, molecule_no):
        """
             go through all neighbors and check that there are no breakage of ice rules
                if there is: return False
                else return True
        """
        nn = self.nearest_neighbors_nos[0][molecule_no]
        nps = self.nearest_neighbors_nos[1][molecule_no]
        bvv = get_bond_variable_values_from_water_orientation(water_orientation)
        assert not self.periodic or len(nn)==4
        for l, nearest_neighbor_no in enumerate(nn):
            periodic = nps[l]
            periodicity_axis = self.nearest_neighbors_nos[2][molecule_no][l]
            neighbor_set = True
            neighbor_bv_value = 0
            opposite = get_opposite_periodicity_axis_number(periodicity_axis)
            # get corresponding water_orientation_value of neighbor and store it to neighbor_bv_value 
            #   -first check that the neighbor has a value
            #      - if not then no breakage happens
            if water_orient[nearest_neighbor_no] != -1:
                # current neighbors bond variable values
                nbbv = get_bond_variable_values_from_water_orientation(water_orient[nearest_neighbor_no])
                for x, n in  enumerate(self.nearest_neighbors_nos[0][nearest_neighbor_no]):
                    # find the neighbor that corresponds to the molecule currently handled (i) [and has the same periodicity and the periodicity axis correspond]
                    if n == molecule_no and self.nearest_neighbors_nos[1][nearest_neighbor_no][x] == periodic and self.nearest_neighbors_nos[2][nearest_neighbor_no][x] == opposite:                       
                        neighbor_bv_value = nbbv[x]
                        break
            else:
                neighbor_set = False
            if water_orientation > 9 and nearest_neighbor_no == molecule_no and bvv[l] != -1:
                return False
             # if both bond variables have the same value, then we have not succeeded
            if neighbor_set and bvv[l] == neighbor_bv_value:
                return False 
        return True

            
        
   
    def handle_molecule_algorithm_2(self, nearest_neighbors_nos, water_orient,  i, water_orientations):
        discarded = 0        
        if water_orient[i] != -1:
            water_orientations = np.vstack((water_orientations,  water_orient))
        else:
            for water_orientation in range(6):

                s = time()
                if self.water_orientation_is_valid(water_orientation, water_orient, i):
                    water_orientations_trial = np.copy(water_orient)
                    water_orientations_trial[i] = water_orientation
                    water_orientations = np.vstack((water_orientations,  water_orientations_trial))
                else:
                    discarded += 1
              
                self.iteration_time += time()-s
                #if success:# and not self.is_symmetric_with_another_result(water_orientations_trial, wo, [bond_variables_trial,  periodic_bond_variables_trial],  bv,   nearest_neighbors_nos):
                    #new_results.append(result_trial)

        return water_orientations, discarded

    def do_initial_grouping(self, water_orientations):
        if size > 1:
            water_orientations = comm.scatter(split_list(water_orientations, wanted_parts=size), root=0)
        wo_count = len(water_orientations)
        # Get the result groups from 1st level
        # NOTE: this can't be done to any other level
        counter = 0
        for i,  water_orientation in enumerate(water_orientations):
            s = time()
            bond_variables = get_bond_variables_2(water_orientation,  self.nearest_neighbors_nos)
            self.conversion_time += time() - s
            rg = self.result_group.try_subgroups(bond_variables,  destination_level=1)
            counter += 1
            if self.store_bond_variables:
                rg.add_result(water_orientation, bond_variables=bond_variables)
            else:
                rg.add_result(water_orientation)
        groups = self.result_group.get_subgroups_from_level(0)
        
        # Remove equal groups
        if size > 1:
            groups = comm.gather(groups, root = 0)
        if rank == 0:
            # merge the list
            groups = merge(groups)
            # merge equal groups
            groups = merge_groups(groups)
            group_count = len(groups)
        else:
            group_count = -1
            groups = []

        for f, group in enumerate(groups):
            print_parallel("Group %i has %i results" % (f, len(group.wos)))
        if size > 1:
            groups = comm.scatter(split_list(groups,  size),  root = 0)
        print_parallel("First scatter done: split %i groups for %i processors. Total %i wos handled." % (group_count, size, counter))

        # Do scattering as long as each processor has at least 3 groups
        #   - if number of results is over 500
        # This is done to equalize the load for each processor
        iteration = 0
        while wo_count > 500 and group_count < 3*size and iteration < 20 and size != 1:
            new_groups = []
            for group in groups:
                for i, wo in enumerate(group.wos):
                    if group.bvs != None and len(group.bvs) > 0:
                        rg = group.try_subgroups(group.bvs[i], destination_level=1)
                        rg.add_result(wo, bond_variables=group.bvs[i])
                    else:
                        bond_variables = get_bond_variables_2(wo,  nearest_neighbors_nos)
                        rg = group.try_subgroups(bond_variables, destination_level=1)
                        rg.add_result(wo)
                group.bvs = None
                group.wos = None
                new_groups.extend(group.get_subgroups_from_level(0))
            if size > 1:
                groups = comm.gather(new_groups, root = 0)
            group_count = -1
            if rank == 0:
                groups = merge(groups)
                group_count = len(groups)
                if group_count > 3*size:
                    for i, group in enumerate(groups):
                        self.save_results(molecule_no, group.wos, None, group_save=True, group_number=i)
                groups = split_list(groups, size)
            group_count = gather_and_max(group_count)
            if size > 1:
                groups = comm.scatter(groups,  root = 0)
            iteration += 1
        
        print_parallel("Results split in %i groups after %i iterations" % (group_count, iteration))
        return groups
    
    def remove_symmetric_results(self,  water_orientations,  molecule_no):
        length_before = len(water_orientations)
        print_parallel("Performing symmetry check for %i geometries" % length_before)
        
        
        # Split the water orientations for the processors
        groups = self.do_initial_grouping(water_orientations)
        
        new_water_orientations = []
        # Finally do the symmetry checking
        print "Processor number %i handles %i result groups" % (rank, len(groups))
        for l, group in enumerate(groups):
            print "Processor %i: Current group has %i wos and invariant value is %i" % (rank, len(group.wos), group.value)
            count = 0
            for i,  wo in enumerate(group.wos):
                if not self.is_symmetric_with_another_result(group,  wo,   current_no=i):
                    new_water_orientations.append(wo)
                    count += 1
            print "Processor %i finished with %i:th group, resulted %i symmetry distinct results" % (rank, l, count)

        
        if size > 1:             
            new_water_orientations = merge(comm.allgather(new_water_orientations))
        if rank == 0:
            print "  -Removed  %i geometries due to symmetry" % (length_before-len(new_water_orientations))
            print "  -Finally %i geometries" % len(new_water_orientations)
        
        return new_water_orientations

    def remove_symmetries_no_invariants(self,  water_orientations,   nearest_neighbors_nos, molecule_no):
        """ 
            Removes symmetries wihout using invariants
            NOTE: Very slow and should only be used for debugging purposes
        """
        if rank == 0:
            length_before = len(water_orientations)
            print "Performing symmetry check for %i geometries" % length_before

        results = []
        for i, wo in enumerate(water_orientations):
            symmetries,  symmetrynames = self.get_symmetries(wo, nearest_neighbors_nos)

            symmetry_found = False
            for wo_2 in results:
                for i, symmetry in enumerate(symmetries):
                    symmetry_found = all(np.equal(wo_2, symmetry))
                    if symmetry_found:
                        print "Found %s symmetry" % symmetrynames[i]
                        break
                if symmetry_found:
                    break
            if not  symmetry_found:
                results.append(wo)
        if rank == 0:
            print "  -Removed  %i geometries due to symmetry" % (length_before-len(results))
            print "  -Finally %i geometries" % len(results)
        return results
    

    def is_symmetric_with_another_result(self, result_group, water_orientations,  current_no=-1, find_single = False):
       
        if current_no != -1 and result_group.bvs != None and len(result_group.bvs) > 0:
            bond_variables = result_group.bvs[current_no]
        else:
            s = time()
            bond_variables = get_bond_variables_2(water_orientations,  self.nearest_neighbors_nos)
            self.conversion_time += time() - s
        # Find the result group
        s = time()
        res_group = result_group.try_subgroups(bond_variables)
        self.result_group_tryout_time += time()-s
        s = time()
        #print "Result group has %i results" % len(res_group.wos)
        # Load symmetries only if there are other results to be checked with
        if len(res_group.wos) >0:
            symmetries,  symmetrynames = self.get_symmetries(water_orientations, self.nearest_neighbors_nos)
        self.symmetry_load_time += time()-s 
        s = time()

        # go through all water orientations with same invariant values
        #  in other words all water orientations that belong to same group
        for i, wator in enumerate(res_group.wos):
            for n,  symmetry in enumerate(symmetries):
                are_symmetric = all(np.equal(wator, symmetry))
                if are_symmetric:
                    self.symmetry_check_time += time()-s
                    if find_single:
                        return wator
                    return True
        self.symmetry_check_time += time()-s
        
        res_group.add_result(water_orientations)
        if find_single:
            return None
        return False

    def finalize_grouping(self, group, current_no = -1):
        """ Finalizes the initial grouping for given group made with do_initial_grouping """
        for wo in group.wos:
            if current_no != -1 and group.bvs != None and len(group.bvs) > 0:
                bond_variables = group.bvs[current_no]
            else:
                s = time()
                bond_variables = get_bond_variables_2(wo,  self.nearest_neighbors_nos)
                self.conversion_time += time() - s

            res_group = group.try_subgroups(bond_variables)
            res_group.add_result(wo)
            print "Result group length %i" % len(res_group.wos)


    def get_single_molecule_hydrogen_coordinates(self, site, water_orientation, i, oxygen_positions,  nearest_neighbors_nos, nn_periodicity, periodicity_axis, cell):
        bvv = get_bond_variable_values_from_water_orientation(water_orientation)
        result = np.zeros((0,3))
        index = 0
        
        for n,  x in enumerate(nearest_neighbors_nos):
            if bvv[n] == 1:
                if i != x or nn_periodicity[n]:
                    if nn_periodicity[n]:
                        distance,  real_position = get_periodic_distance(oxygen_positions[i], oxygen_positions[x], cell, periodicity_axis[n])
                        
                    else:
                        distance = get_distance(oxygen_positions[i], oxygen_positions[x], False, None)
                        real_position = oxygen_positions[x]
                    result = np.vstack((result, oxygen_positions[i] - (( self.O_H_distance * (oxygen_positions[i]-real_position)) / distance))) 
                    index += 1
                else:
                    site = np.mod(i, 8)
                    if site == 2 or site == 6:
                        add = np.array(oxygen_positions[i])
                        add[2] += self.O_H_distance
                        result = np.vstack((result, add))
                    elif site == 0 or site == 4:
                        add = np.array(oxygen_positions[i])
                        add[2] -= self.O_H_distance
                        result = np.vstack((result, add))
                    index += 1
        return result
    
    def get_single_molecule_dipole_moment(self, site, water_orientation, i, oxygen_positions, nearest_neighbors_nos, nn_periodicity, nn_periodicity_axis):
        h_coordinates = self.get_single_molecule_hydrogen_coordinates(site, water_orientation, i, oxygen_positions, nearest_neighbors_nos, nn_periodicity, nn_periodicity_axis, self.cell) 
        dipole_moment = ((h_coordinates[0]-oxygen_positions[i]) / (self.O_H_distance * 2)) + ((h_coordinates[1]-oxygen_positions[i]) / (self.O_H_distance * 2))
        return dipole_moment
        
    def get_total_dipole_moment(self, water_orientations):
        total_dipole_moment = [0, 0, 0]
        for i,  water_orientation in enumerate(water_orientations):
            site = np.mod(i, 4)
            total_dipole_moment += self.get_single_molecule_dipole_moment(site, water_orientation, i, self.oxygen_coordinates, self.nearest_neighbors_nos[0][i], self.nearest_neighbors_nos[1][i], self.nearest_neighbors_nos[2][i])
        return total_dipole_moment

    def write_no_zeros(self):
        from classification import Classification
        c = Classification(self, None)  
        cs = self.load_results(nozeros=False)
        no_zeros = c.remove_impossible_angles(cs)
        np.savetxt(self.folder+"/nozero_results.txt", no_zeros.values())
        if not os.path.exists(self.folder+"/nozeros/"):
            os.makedirs(self.folder+"/nozeros/")
        write_results_to_file(self,  self.oxygen_coordinates, no_zeros.values(), self.cell,  self.nearest_neighbors_nos,  folder=self.folder+'/nozeros/')

    def write_geometries(self):
        from classification import Classification
        c = Classification(self, None)  
        cs = self.load_results(nozeros=False)
        no_zeros = c.remove_impossible_angles(cs)
        np.savetxt(self.folder+"/nozero_results.txt", no_zeros.values())
        if not os.path.exists(self.folder+"/original/"):
            os.makedirs(self.folder+"/original/")
        write_results_to_file(self,  self.oxygen_coordinates, no_zeros.values(), self.cell,  self.nearest_neighbors_nos,  folder=self.folder+'/nozeros/')

    def execute_commands(self, options, args = None):
        if options.wa_method != None:
            method = getattr(self, str(options.wa_method))
            if args != None:
                method(*args)
            else:
                method()
            

    # Reverse engineering method
    def find_equivalent(self, filename):
        oxygens_0 = self.atoms.copy()
        atoms = ase.io.read(filename)
        oxygens = remove_hydrogens(atoms)
        
        ivalues, ivectors = sym.get_moments_of_inertia(self.atoms)
        i2values, i2vectors  = sym.get_moments_of_inertia(oxygens)
        new_atoms = atoms.copy()
        new_oxygens = oxygens.copy()
        maxvalue = max(ivalues)
        selected_axis = -1
        evectors = np.zeros_like(i2vectors)
        print ivectors
        print ivalues
        print i2vectors
        print i2values
        equivalence = [-1, -1, -1]
        for i, value in enumerate(ivalues):
            if value == maxvalue:
                selected_axis = i
            for i2, value2 in enumerate(i2values):
                print "%i and %i are equal?  %s" % (i, i2, are_equal(value, value2, error_tolerance=100))
                if are_equal(value, value2, error_tolerance=350) and i2 not in equivalence:
                    equivalence[i] = i2
                    evectors[i] = i2vectors[i2]
                    break
        print equivalence
        # find coordinates  in inertia basis for both
        basis_1 = np.reshape(ivectors, ivectors.size, order='F').reshape((3, 3))
        basis_2 = np.reshape(evectors, evectors.size, order='F').reshape((3, 3))
        
        com = self.atoms.get_center_of_mass()
        com2 = oxygens.get_center_of_mass()
        c1 = get_coordinates_in_basis(self.atoms.get_positions()-com, ivectors)
        c2 = get_coordinates_in_basis(oxygens.get_positions()-com2, i2vectors)
        c2 = get_coordinates_in_basis(c2, ivectors)
        c3 = np.zeros_like(c2)
        
        for i, e in enumerate(equivalence):
            c3[:,i] = c2[:,e]
        self.atoms.set_positions(c1)
        new_oxygens.set_positions(c3)
        moi1 = sym.get_moments_of_inertia(self.atoms)[1]
        moi2 = sym.get_moments_of_inertia(new_oxygens)[1]
        c1[:, 2] = -c1[:, 2]
        c1[:, 1] = c1[:, 1]
        #c1[:, 0] = -c1[:, 0]
        #closest_position = sym.get_closest_position(c1[0], c2)
        
        self.atoms.set_positions(c1)
        new_oxygens.set_positions(c3)
        #self.atoms = rotate_to(self.atoms, c1[0], closest_position)
        ase.visualize.view(new_oxygens) 
        ase.visualize.view(self.atoms)
        
        equals = sym.get_equals(self.atoms, remove_hydrogens(new_oxygens).get_positions(), error_tolerance=1.2)
        cp = sym.get_closest_positions(c1, c3)
        # dodecahedron hack        
        #equals = [19, 14, 18, 15, 9, 13, 5, 17, 10, 16, 4, 8, 0, 12, 6, 11, 3, 1, 7, 2]
        #equals = [12, 17, 19, 16, 10, 6, 14, 18, 11, 4, 8, 15, 13, 5, 1, 3, 9, 7, 2, 0]
        # espp hack
        equals = [0, 8, 1, 7, 19, 4, 3, 17, 11, 16, 18, 5, 13, 14, 10, 15, 12, 6, 2, 9]
        if equals != None:
            bvv, elongated_bonds, elongated_hydrogen_bonds = self.get_bond_varibles_from_atoms(atoms)
            water_orientations = self.get_water_orientations_from_bond_variables_and_molecule_change_matrix(bvv, equals)
            self.view_result(water_orientations)
            print water_orientations
            wos = self.load_results()
            self.initialize_symmetry_operations()
            self.load_invariants()
            self.result_group = ResultGroup(list(self.graph_invariants),  0)
            self.print_time_stats()
            self.do_initial_grouping(wos)
            bond_variables = get_bond_variables_2(water_orientations,  self.nearest_neighbors_nos)
            rg = self.result_group.try_subgroups(bond_variables,  destination_level=1)
            print "Result groups has %i results" % len(rg.wos)
            self.finalize_grouping(rg)
            symmetry = self.is_symmetric_with_another_result(rg,  water_orientations,  find_single=True)

            assert symmetry != None
            for i, wo in enumerate(wos):
                if all(np.equal(wo, symmetry)):
                    print "Structure is symmetric with structure number %i" % i 

            

   

    
                   


def add_options(parser):
    from optparse import OptionGroup
    group = OptionGroup(parser, "Generation methods", "Methods used for generation of proton configurations")
    group.add_option("--execute", dest='wa_method', const="run", action='store_const',
                          help="Execute the proton configuration enumeration algorithm. Generates all possible proton configurations according to other input parameters.")
    group.add_option("--execute_profile_number", dest='wa_method', const="run_profile_number", action='store_const',
                          help="Execute the proton configuration enumeration algorithm for profile number given as input.")
    group.add_option("--execute_profile", dest='wa_method', const="run_profile_args", action='store_const',
                          help="Execute the proton configuration enumeration algorithm for profile given as input. Example: \"--execute_profile O H O H O H O H\".")
    group.add_option("--generate_profile_representative", dest='wa_method', const="generate_profile_representative", action='store_const',
                          help="Generates the first possible representative proton configuration for profile with number given as input. Examples: \"--generate_profile_representative O H O H O H O H\" or \"--generate_profile_representative 5\"")
    group.add_option("--generate_profile_representatives", dest='wa_method', const="generate_profile_representatives", action='store_const',
                          help="Generates the first possible representative proton configurations for all profiles.")
    parser.add_option_group(group)
    
    group = OptionGroup(parser, "Pre-Generation methods", "Methods that can be called before proton configuration generation.")
    group.add_option("--print_allowed_profiles", dest='wa_method', const="print_allowed_dangling_bond_profiles", action='store_const',
                          help="Print allowed symmetry-independent dangling bond profiles.")
    group.add_option("--print_nearest_neighbors", dest='wa_method', const="print_nearest_neighbors_nos", action='store_const',
                          help="Print nearest neighbors nos")
    parser.add_option_group(group)                      
    
    group = OptionGroup(parser, "Debug methods", "Methods that can used for the verification of the enumeration results.")
    group.add_option("--verify_results", dest='wa_method', const="verify_results", action='store_const',
                          help="Verifies that there are no symmetric results")
    group.add_option("--verify_results_and_overwrite", dest='wa_method', const="verify_results_and_overwrite", action='store_const',
                          help="Verifies that there are no symmetric results and overwrites the allresults.txt")
    parser.add_option_group(group)

    group = parser.get_option_group("--view_structure") 
    if group is not None:
        group.add_option("--find_equivalent", dest='wa_method', const="find_equivalent", action='store_const',
                              help="Find the configuration that corresponds to the input structure name")
    

    group = OptionGroup(parser, "Enumeration parameters", "Parameters used in the enumeration of proton configurations, i.e., with --execute")
    group.add_option("--write_geometries", dest='write_geometries', action='store_true',
                      help="Write the geometries after execution")
    group.add_option("--no_symmetry_check", dest='do_symmetry_check', action='store_false', 
                      help="Do not remove symmetric results.")
    group.add_option("--invariant_count", dest='invariant_count', type='int', action='store', default=25,
                      help="Maximum number of invariants used during execution.")
    group.add_option("--dissosiated_molecule_count", dest='dissosiation_count', type='int', action='store', default=0,
                      help="Maximum number of invariants used during execution.")
    #group.add_option("--oxygen_raft_file", dest='filename', type='string', action='store', default=None,
    #                  help="The location of file containing the oxygens that belong to the raft.")
    group.add_option("--no_self_symmetry_groups", dest='no_self_symmetry_groups',  action='store_true', default=False,
                      help="Don't use self symmetry groups during execution.")
    parser.add_option_group(group)

    
    






def write_results_to_file(wa, oxygen_positions, results, cell,  nearest_neighbors_nos,  folder=''):
    for i, water_orientation in enumerate(results):
        write_result_to_file(wa, oxygen_positions, water_orientation, i, cell, nearest_neighbors_nos, folder=folder)

def write_result_to_file(wa, oxygen_positions, water_orientation, i, cell,  nearest_neighbors_nos,  folder=''):
    oxygens = get_oxygens(oxygen_positions)
    hydrogen_positions = []
    hydrogen_positions = wa.get_hydrogen_coordinates(oxygen_positions, water_orientation,  nearest_neighbors_nos)
    hydrogens = get_hydrogens(hydrogen_positions)
    waters = oxygens.extend(hydrogens)
    if cell != None:
        waters.set_cell(cell)
    else:
        waters.center(vacuum=5)   
    ase.io.write(("%sWater_structure_%i.traj" % (folder,  i)), waters, format="traj") 
    


def find_hydrogen_bonds(atom_position, atom_positions, periodic=False, cell=None):
    result = np.array([])
    periodicity = []
    count = 0
    periodic_count = 0
    for i in range(len(atom_positions)):
        distance = get_distance(atom_positions[i], atom_position, False, cell)
        if(distance < 1.95 and distance>1.50):
            result = np.append(result, i)
            periodicity.append(False)
            count = count +1 
        if periodic:
            periodic_distance,  real_position = get_periodic_distance(atom_position,  atom_positions[i], cell)
            if(periodic_distance < 1.95 and periodic_distance>1.50):
                result = np.append(result, i)
                periodicity.append(True)
                count = count +1 
    #print "Molecules hydrogen bond count %i " %  (count)
    return result, periodicity, count
    
def nearest_neighbors(atom_position, atom_positions, periodic=False, cell=None):
    result = np.empty(4)
    count = 0
    for i in range(len(atom_positions)):
        distance = get_distance(atom_positions[i], atom_position, periodic=periodic, cell=cell)
        if(distance == 2.76 or (distance > 2.510 and distance < 2.530)):
            result[count] = atom_positions[i]
            count = count + 1
    return result

def nearest_neighbors_no(water_algorithm,  atom_no,  atom_position, atom_positions, periodic=False, cell=None):
    result = np.array([],  dtype="int8")
    periodicity = []
    periodicity_axis = []
    sortlist = np.zeros((0, 3),  dtype="int8")
    count = 0
    min_distance = water_algorithm.O_O_distance * 0.85
    max_distance = water_algorithm.O_O_distance * 1.15
    for i in range(len(atom_positions)):
        distance = get_distance(atom_position,  atom_positions[i] , periodic=False)
        if (distance > min_distance and distance < max_distance):
            result = np.append(result, i)
            periodicity = np.append(periodicity, False)
            periodicity_axis.append(13)
            count = count + 1
            sortlist = np.vstack((sortlist,  atom_positions[i]))
        if periodic:
            result, periodicity, periodicity_axis, count, sortlist = add_periodic_neighbors(atom_position, atom_positions[i], cell, min_distance, max_distance, i, result, periodicity, periodicity_axis, count, sortlist)
            
    # following lines are deprecated because of the symmetry finding algorithm
    ind = water_algorithm.sort_nearest_neighbors(atom_no,  sortlist,  result)
    if ind != None:
        periodicity = np.array([periodicity[i] for i in ind],  dtype="int8")
        periodicity_axis = np.array([periodicity_axis[i] for i in ind],  dtype="int8")
        result = np.array([result[i] for i in ind],  dtype="int8")
    return result, periodicity, periodicity_axis



def all_nearest_neighbors_no(water_algorithm,  atom_positions, periodic=False, cell=None):
    result = []
    periodicities = []
    periodicity_axis = []
    count = 0
    for i in range(len(atom_positions)):
        nos, periodicity, periodicity_ax = nearest_neighbors_no(water_algorithm,  i,  atom_positions[i], atom_positions, periodic, cell)
        nos, periodicity, periodicity_ax = add_dangling_bonds_to_nearest_neighbors(i,  nos,  periodicity, periodicity_ax)
        result.append(nos)
        periodicities.append(periodicity)
        periodicity_axis.append(periodicity_ax)
    return np.array([result, periodicities, periodicity_axis], dtype="int")


def add_dangling_bonds_to_nearest_neighbors(molecule_no,  nearest_neighbors_nos, periodicity, periodicity_axis):
    if len(nearest_neighbors_nos) == 4:
        return nearest_neighbors_nos,  periodicity, periodicity_axis
    result = nearest_neighbors_nos
    while result.shape[0] < 4:
        result = np.append(result, molecule_no)
        periodicity = np.append(periodicity, False)
        periodicity_axis = np.append(periodicity_axis, 13)
    return result,  periodicity, periodicity_axis


def add_periodic_neighbors(p1, p2, cell, min_distance, max_distance, p2_number, result, periodicities, periodicity_axis, count, sortlist):
    distances, axis = get_periodic_distances(p1, p2, cell)
    for distance, p_ax in zip(distances, axis):
        if distance > min_distance and distance < max_distance and p_ax != 13:
            result = np.append(result, p2_number)
            periodicities = np.append(periodicities, True)
            periodicity_axis.append(p_ax)
            vec = get_vector_from_periodicity_axis_number(p_ax)
            sortlist = np.vstack((sortlist,  p2+vec*cell))
            count += 1
    return result, periodicities, periodicity_axis, count, sortlist

def get_bond_variables_2(water_orientations, nearest_neighbors_nos):
    result = {}
    for i,  nn in enumerate(nearest_neighbors_nos[0]):
        result[i] = {}
        water_orientation = water_orientations[i]
        bvv = get_bond_variable_values_from_water_orientation(water_orientation)
        for j,  neighbor_no in enumerate(nn):
            if neighbor_no not in result[i]:
                result[i][neighbor_no] = {}
            periodic_axis = nearest_neighbors_nos[2][i][j]
            assert periodic_axis not in result[i][neighbor_no]
            result[i][neighbor_no][periodic_axis] = bvv[j]
    return result

def get_bond_variables(water_orientations, nearest_neighbors_nos):
    bond_variables = np.zeros((2, len(nearest_neighbors_nos[0]),  len(nearest_neighbors_nos[0])),  dtype='int8')
    for i,  nn in enumerate(nearest_neighbors_nos[0]):
        water_orientation = water_orientations[i]
        bvv = get_bond_variable_values_from_water_orientation(water_orientation)
        for l,  neighbor_no in enumerate(nn):
            periodic = nearest_neighbors_nos[1][i][l]
            if neighbor_no > i:
                bond_variables[periodic][i][neighbor_no][0] = bvv[l]
                bond_variables[periodic][i][neighbor_no][1] = nearest_neighbors_nos[2][i][l]
                bond_variables[periodic][neighbor_no][i][0] = -bvv[l]
                bond_variables[periodic][neighbor_no][i][1] = get_opposite_periodicity_axis_number(nearest_neighbors_nos[2][i][l])
            elif neighbor_no == i:
                bond_variables[periodic][i][neighbor_no][0] = bvv[l]
                bond_variables[periodic][i][neighbor_no][1] = nearest_neighbors_nos[2][i][l]
    return bond_variables
            
