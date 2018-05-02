import ase
import os
import numpy as np
has_matplotlib = True
try:
    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot
    matplotlib.rcParams['font.size'] = 16
    from matplotlib import pyplot as plt
except ImportError:
    has_matplotlib = False
import sys
import math
from help_methods import *
from optparse import OptionParser
from scipy.constants import k
from classification import Classification
from ase.vibrations import Vibrations
from symmetries.interface import get_periodic_symmetry_group
from structure_commons import StructureCommons
from system_commons import rank, size, comm
from structure_commons import add_options


class HandleResults(StructureCommons):
    def __init__(self, water_algorithm = None, nozeros=False, fmax=0.05, xc='PBE', number_of_relaxation = 0, subplot=False, name=None, program = "GPAW", method="dft", folder = None, file_type = "traj", debug = False, options = None, O_O_distance = None, O_H_distance = None):
        StructureCommons.__init__(self, **vars(options))
        self.options = options
        self.number_of_relaxation = number_of_relaxation
            
        self.water_algorithm = water_algorithm
        self.nozeros = nozeros
        self.subplot = subplot
        self.name=name


    def get_structure_average_O_O_bond_length(self, number):
        assert self.water_algorithm != None
        atoms = self.read(number)
        if atoms == None:
            return None
        oxygens = remove_hydrogens(atoms)
        total_O_O_bond_length = 0.0
        count = 0
        for i, oxygen in enumerate(oxygens):
            nn = self.water_algorithm.nearest_neighbors_nos[0][i]
            for j, neighbor in enumerate(nn):
                if neighbor >= i:
                    axis = self.water_algorithm.nearest_neighbors_nos[2][i][j]
                    
                    if axis == 13:
                        if neighbor != i:
                            count += 1
                            total_O_O_bond_length += get_distance(oxygens[i].get_position(), oxygens[neighbor].get_position())
                    else:
                        print get_periodic_distance(oxygens[i].get_position(), oxygens[neighbor].get_position(), self.water_algorithm.cell, axis)
                        total_O_O_bond_length += get_periodic_distance(oxygens[i].get_position(), oxygens[neighbor].get_position(), self.water_algorithm.cell, axis)
        return total_O_O_bond_length / count

    def print_average_O_O_bond_length(self):
        total = 0.0
        count = 0
        maximum_bond_length = None
        max_number = -1
        minimum_bond_length = None
        min_number = -1
        minimum_energy_number, energy = self.get_minimum_energy()
        minimum_energy_bond_length = None
        for number in range(self.get_number_of_results()):
            av_bond_length = self.get_structure_average_O_O_bond_length(number)
            if number == minimum_energy_number:
                minimum_energy_bond_length = av_bond_length
            if av_bond_length != None:
                if maximum_bond_length == None or av_bond_length > maximum_bond_length:
                    maximum_bond_length = av_bond_length
                    max_number = number
                if minimum_bond_length == None or av_bond_length < minimum_bond_length:
                    minimum_bond_length = av_bond_length
                    min_number = number
                total += av_bond_length
                count += 1

        if max_number != -1:
            contents = "Total number of results: %i\n" % count
            contents+= " -Average oxygen-oxygen bond length of all structures %f\n" % (total/count)
            contents+= " -Maximum average oxygen-oxygen bond lengths are on structure %i: %f A\n" % (max_number, maximum_bond_length) 
            contents+= " -Minimum average oxygen-oxygen bond lengths are on structure %i: %f A\n" % (min_number, minimum_bond_length) 
            contents+= " -Average oxygen-oxygen bond lengths of minimum energy structure %i: %f A\n" % (minimum_energy_number, minimum_energy_bond_length) 
            self.write_text("bond_length_statistics", contents)

    def print_angle_statistics(self):
        c = Classification(options = self.options)  
        results = c.do_bond_angle_analysis() # dict
        total_vertice_deviation = 0.0
        total_angle_deviation = 0.0
        max_vertice_deviation = None
        max_vertice_deviation_no = 0
        min_vertice_deviation = None
        min_vertice_deviation_no = None
        max_angle_deviation = None
        max_angle_deviation_no = None
        min_angle_deviation = None
        min_angle_deviation_no = None
        minimum_energy_number, energy = self.get_minimum_energy()
        count = 0
        for key in results:
            vertice_deviation = results[key][1]
            angle_deviation = results[key][0]
            if min_angle_deviation == None or angle_deviation < min_angle_deviation:
                min_angle_deviation = angle_deviation
                min_angle_deviation_no = key
            if max_angle_deviation == None or angle_deviation > max_angle_deviation:
                max_angle_deviation = angle_deviation
                max_angle_deviation_no = key
            if max_vertice_deviation == None or vertice_deviation > max_vertice_deviation:
                max_vertice_deviation = vertice_deviation
                max_vertice_deviation_no = key
            if min_vertice_deviation == None or vertice_deviation < min_vertice_deviation:
                min_vertice_deviation = vertice_deviation
                min_vertice_deviation_no = key
            total_vertice_deviation += vertice_deviation
            total_angle_deviation += angle_deviation
            count += 1
        if count != 0:
            contents = "Total number of results: %i\n" % count
            contents+= " -Average vertice deviation bond length of all structures %f\n" % (total_vertice_deviation/count)
            contents+= " -Maximum vertice deviation is on structure %i: %f A\n" % (max_vertice_deviation_no, max_vertice_deviation) 
            contents+= " -Minimum vertice deviation is on structure %i: %f A\n" % (min_vertice_deviation_no, min_vertice_deviation) 
            contents+= " -Minimum energy structures vertice deviation is %i: %f A\n" % (minimum_energy_number, results[minimum_energy_number][1]) 
            contents+= " -Average angle deviation bond length of all structures %f\n" % (total_angle_deviation/count)
            contents+= " -Maximum angle deviation is on structure %i: %f A\n" % (max_angle_deviation_no, max_angle_deviation) 
            contents+= " -Minimum angle deviation is on structure %i: %f A\n" % (min_angle_deviation_no, min_angle_deviation) 
            contents+= " -Minimum energy structures angle deviation is %i: %f A\n" % (minimum_energy_number, results[minimum_energy_number][0]) 
            self.write_text("bond_angle_statistics", contents)
    
        

    def print_altered_structures(self):
        """ Print info about the structures which changed during relaxation """
        originals = self.get_results()
        count = 0
        counts = [0, 0, 0, 0, 0, 0, 0]
        contents = [None, None, None, None, None, None, None]
        keywords = ["altered", "dissosiation of water molecules", "assembling of water molecules", "proton migration", "hydrogen bond breaking", "elongated intramolecular O-H bonds", "elongated hydrogen bonds"]
        for number in range(self.get_number_of_results()):
            original_orientation = originals[number]
            altered = self.structure_altered_during_relaxation(number, original_orientation)
            if altered != None:
                count += 1
                if altered[0]:
                    for i, alt_value in enumerate(altered):
                        if alt_value:
                            counts[i] += 1
                            if contents[i] != None:
                                contents[i] += ", "
                            else:
                                contents[i] = ""
                            contents[i] += str(number)
        result = None
        for i, content in enumerate(contents):
            if content != None:
                if i == 0:
                    result = "Of %i structures %i were altered during relaxation.\n" % (count, counts[i])
                else:
                    result += "  - In %i structures there were %s. The numbers of these structures are: \n" % (counts[i], keywords[i]) + content + "\n"

        if result == None:
            result = "Of %i structures none were altered during relaxation" % count

        self.write_text("altered_structures", result)

    def get_structures_altered(self):
        """ 
            Print info about the structures which changed during relaxation
        """
        originals = self.get_results()
        result = {}
        elongated_hydrogen_bonds = {}
        elongated_bonds = {}
        
        for number in range(self.get_number_of_results()):
            original_orientation = originals[number]
            altered = self.structure_altered_during_relaxation(number, original_orientation)
            if altered is not None:
                result[number] = int(altered[0])
                elongated_hydrogen_bonds[number] = int(altered[6])
                elongated_bonds[number] = int(altered[4])
        self.write_dict_to_file(result, "structure_altered")
        self.write_dict_to_file(elongated_hydrogen_bonds, "elongated_hydrogen_bonds")
        self.write_dict_to_file(elongated_bonds, "elongated_bonds")
        return result
        

        
    

    def read_single_structure_mulliken_charges(self, number, folder=None, number_of_relaxation=None, xc=None):
        filename = self.get_filename(number, folder=folder, loading = True, load_structure=number_of_relaxation, xc=xc)
        filename += "_mulliken.dat"
        #print filename
        result = None
        if self.calculation_finished(number, folder=folder, number_of_relaxation=number_of_relaxation, xc=xc) and os.path.exists(filename):
            result = []
            file = open(filename, 'r')
            lines = file.readlines()
            if lines == None or len(lines) == 0:
                return None
            read = False
            for line in lines:
                words = line.split()
                if len(words) > 0:
                    print words
                    if words[0] == "#":
                        read = not read
                    else:
                        if read:
                            result.append([words[1], words[4]])   
        return result

    def get_sum_of_four_lowest_hydrogen_mulliken_charges(self, number, folder = None, number_of_relaxation = None, xc = None):
        mullikens = self.read_single_structure_mulliken_charges(number, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc)
        if mullikens != None:
            lowest_four = [None, None, None, None]
            print mullikens
            for item in mullikens:
                if item[0] == 'H':
                    new_lowest_four = [None, None, None, None]
                    counter = 0
                    for i, low in enumerate(lowest_four):
                        if counter < 4 and  i == counter and (low == None or item[1] < low):
                            print i, counter
                            new_lowest_four[counter] = item[1]
                            counter += 1
                        if counter < 4:
                            new_lowest_four[counter] = low
                            counter += 1
                    lowest_four = new_lowest_four
            result = 0.0
            print lowest_four
            #for item in mullikens:
            #    if item[0] == 'H':
            #        result += float(item[1])
            for low in lowest_four:
                if low != None:
                    result += float(low)
            return result
        else:
            return None

    def get_sums_of_four_lowest_hydrogen_mulliken_charges(self, folder = None, number_of_relaxation = None, xc = None):
        result = {}
        if self.program == "cp2k":
            if folder == None:
                folder = self.folder
            if number_of_relaxation == None:
                number_of_relaxation = self.number_of_relaxation
            if xc == None:
                xc = self.xc
            for number in range(self.get_number_of_results()):
                sum4 = self.get_sum_of_four_lowest_hydrogen_mulliken_charges(number, folder = folder, number_of_relaxation = number_of_relaxation, xc = xc)
                if sum4 != None:
                    result[number] = sum4
            self.write_dict_to_file(result, "mulliken4")
        return result

    def get_sublimation_energy(self, number, unit = "kcalmol"):
        number = self.handle_number_input(number)
        energy = self.read_energy(number, folder=self.folder, number_of_relaxation=self.number_of_relaxation, xc=self.xc)
        wm_energy = self.get_water_molecule_energy()
        N_H2O = len(self.water_algorithm.oxygen_coordinates)
        energy  -= (N_H2O * wm_energy)
        if unit == 'kcalmol':  
            energy *=  23.061
            print "Sublimation energy of %i is %f kcal/mol" % (number,energy)
        else:
            print "Sublimation energy of %i is %f eV" % (number,energy)
        return energy

    

    
        
    def read_energies(self, latest=False):
        """
            Reads the energies in eV
        """
        result = {}
        for number in range(self.get_number_of_results()):
            if latest:
                energy = self.read_latest_energy(number)
                if energy  != None:
                    result[number] = energy
            else:
                atoms = self.read(number)
                if atoms != None:
                    energy = self.read_energy(number)
                    if energy != None:
                        result[number] = energy
        #print "Number of energies %i" % len(result)
        self.write_dict_to_file(result, "energy")
        return result

    def get_sublimation_energies(self, latest = False, unit = "kcalmol"):
        energies = self.read_energies(latest = latest)
        wm_energy = self.get_water_molecule_energy()
        N_H2O = len(self.water_algorithm.oxygen_coordinates)
        result = {}
        for key in energies:
            energy = energies[key] - (N_H2O * wm_energy)
            if unit == 'kcalmol':  
                energy *=  23.061
            result[key] = energy
        self.write_dict_to_file(result, "sublimation_energy_"+unit)
        return result

    def get_relative_energies(self, latest = False, unit = "kcalmol"):
        energies = self.read_energies(latest = latest)
        no, min_energy = self.get_minimum_energy(unit, latest=latest)     
        N_H2O = len(self.water_algorithm.oxygen_coordinates)
        result = {}
        for key in energies:       
            energy = energies[key]     
            energy -= min_energy
            if unit == 'kcalmol':  
                energy *=  23.061
            
            result[key] = energy
        self.write_dict_to_file(result, "relative_energy_"+unit)
        return result
        
            

    
    def classify_by_dangling_bond_classification(self, input):
        classifications = np.loadtxt(self.folder+"dangling_bond_classification.txt",  dtype="int8")
        result = {}
        for key in input.keys():
            classification = classifications[key]
            if  classification in result:
                result[classification].append(input[key])
            else:
                result[classification] = [input[key]]
        return result
    
    def classify(self, input):
        classifications = np.loadtxt(self.folder+"classification.txt",  dtype="int8")
        result = {}
        for key in input.keys():
            b_ff =  classifications[key][0]
            g_3 = classifications[key][1]
            g_4 = classifications[key][1]
            classification = str(b_ff) + str(g_3) #+ str(g_4)
            
            if  classification in result:
                result[classification].append(input[key])
            else:
                result[classification] = [input[key]]
        return result

    def plot_energy_distribution(self, stabilization = False, unit = "eV"):
        energies = self.read_energies()
        enes = np.array(energies.values())
        if stabilization:
            wm_energy = self.get_water_molecule_energy()
            enes -= (20*wm_energy)
        if unit == 'kcalmol':
            enes *= (23.06/20.0)
        new_enes = np.array([])
        #for ene in new_enes:
        #    if ene != 0.0:
        #        new_enes = np.vstack((new_enes, ene))
        matplotlib.pyplot.hist(enes, bins=50, rwidth=1, histtype='stepfilled', edgecolor=None)
        if not self.subplot:
            matplotlib.pyplot.savefig(self.get_image_file_name('energy_distribution'))
            return self.show_plot()
    
    def get_energy_vs_dipole_moment(self):
        energies = self.read_energies()
        selected_dipole_moments = self.read_dipole_moments()
        #dipole_moments = get_dipole_moments()
        #selected_dipole_moments = []
        #for i, number  in enumerate(energies.keys()):
        #   selected_dipole_moments.append(dipole_moments[number])
           
        return energies.values(),  selected_dipole_moments.values()
        #return classify(energies),  classify(selected_dipole_moments)

    def get_pre_dipole_vs_calc_dipole(self):
        selected_dipole_moments = self.read_dipole_moments()
        dipole_moments = self.get_dipole_moments()
        sdm = []
        for i, number  in enumerate(selected_dipole_moments.keys()):
           sdm.append(dipole_moments[number])
        return sdm,  np.array(selected_dipole_moments.values()) 

    def get_dipole_moments(self):
        
        wa = self.water_algorithm
        wos = self.get_results()
        try:
            from handle_results_cython import get_dipole_moments as gdm
            norms, normxys = gdm(wos, self.oxygen_coordinates, self.nearest_neighbors_nos, self.O_H_distance, self.cell) 
        except ImportError:
            print "Falling back to Python"
            if wos is not None:
                norms = np.array([])
                normxys = np.array([])
                for i,  water_orientations in enumerate(wos):
                    dipole_moment = wa.get_total_dipole_moment(water_orientations)
                    norm = np.linalg.norm(dipole_moment)
                    normxy = np.linalg.norm(dipole_moment[0:2])
                    norms = np.append(norms,  norm)
                    normxys = np.append(normxys,  normxy)
                    if i % 1000 == 0:
                        print "%i / %i" % (i, self.get_number_of_results())
                
            else:
                raise Exception("Result structures were not found. Please do --execute before this operation.")
        self.write_dict_to_file(norms, "estimated_dipole_moment")
        self.write_dict_to_file(normxys, "estimated_dipole_moment_xy")
        return norms
        
    def calculate_original_core_electrostatic_energies(self):
        """
            Calculate the core repulsions for the original structure
            with given input parameters
        """
        wa = self.water_algorithm
        wos = self.get_results()
        result = {}
        from energetics_cython import calculate_core_electrostatic_potential_energy_for_atoms
        for i, wo in enumerate(wos):
            print i
            atoms = self.get_atoms(wo)
            result[i] = calculate_core_electrostatic_potential_energy_for_atoms(atoms)
            
        self.write_dict_to_file(result, "original_core_electrostatic_energy")
        return result
        
    def calculate_core_electrostatic_energies(self):
        
        wa = self.water_algorithm
        wos = self.get_results()
        result = {}
        from energetics_cython import calculate_core_electrostatic_potential_energy_for_atoms
        for i, wo in enumerate(wos):
            atoms = self.read_latest_structure_from_file(i, basis_set = self.basis_set, xc = self.xc, program = self.program, method = self.method)
            if atoms is not None:
                result[i] = calculate_core_electrostatic_potential_energy_for_atoms(atoms)
            
        self.write_dict_to_file(result, "core_electrostatic_energy")
        return result
        

    def get_boltzmann_weights(self, T = 243):
        eV_to_J = 1.62177 * 10**-19
        energies = self.read_energies()
        #dipole_moments = get_dipole_moments()
        exponentials = {}
        total = 0
        minimum_energy = min(energies.values())
        for i, number  in enumerate(energies.keys()):
            exponentials[number] = np.exp(-((energies[number]-minimum_energy) * eV_to_J) / (k * T))
            total += exponentials[number]
        
        weights = {}
        for number in exponentials:
            weights[number] = exponentials[number] / total
        return weights
    

    def get_dipole_moment_boltzmann(self, T=243):
        weights = self.get_boltzmann_weights(T=T)
        return np.array(selected_dipole_moments.values()) , np.array(weights.values())

    def plot_dipole_moments_boltzmann(self, Ts=[203, 273]):
        
        matplotlib.rcParams['font.size'] = 14
        matplotlib.rcParams['text.color'] = "r"
        for i, T in enumerate(Ts):
            number = 100 + 20 + (i+1)
            matplotlib.pyplot.subplot(number)
            dms, weights = self.get_dipole_moment_boltzmann(T)
            print "Weight between 0 and 1: %f" % self.get_weight_between(dms, weights, 0, 1)
            print "Weight between 4 and 5: %f" % self.get_weight_between(dms, weights, 4, 5)
            matplotlib.pyplot.hist(dms, histtype="barstacked", extent=None, bins=100, weights=weights)
            matplotlib.pyplot.ylim(0,0.9)
            matplotlib.pyplot.xlim(0,6)
            
            if i == 0:
                matplotlib.pyplot.xlabel("Dipole moment (Debye)")
                matplotlib.pyplot.ylabel("Weight")
            else:
                frame1 = matplotlib.pyplot.gca()
                frame1.axes.get_yaxis().set_visible(False)
        if not self.subplot:
            matplotlib.pyplot.savefig(self.get_image_file_name('dipole_moments_bolzmann_%iK_%iK' % (Ts[0], Ts[1])))
            return self.show_plot()

    def get_weight_between(self, dms, weights, start, end):
        result = 0
        for i, dm in enumerate(dms):
            if dm >= start and dm < end:
                result += weights[i]
        return result



     

    def get_indeces(self, indeces, results):
        result = {}
        if type(results) == dict:
            for i in indeces:
                if i in results.keys():
                    result[i] = results[i]
        else: # Handle lists and np.ndarrays
            for i in indeces:
                if i < len(results):
                    if results[i] is not None:
                        result[i] = results[i]
        return result

    def list_as_dict(self, list):
        result = {}
        for i, item in enumerate(list):
            result[i] = item
        return result


    def print_classes(self, only_impossible_angles=False):
        """
            only_impossible_angles : boolean - if true then the only the 
            structures with below 85 or over 130 O-O-O bond_angles are printed
        """
        c = Classification(options = self.options)  
        classi = c.do_default_grouping_for_all() 
        cs = self.get_results()
        file_name = "classes"
        contents = "Classes of water structures\n"
        if only_impossible_angles:
            contents += " - Contains only impossible angles\n"
            cs = c.remove_possible_angles(cs)
            file_name += "_impossible"
        else:
            cs = self.list_as_dict(cs)
        contents += "--------------------------------------------\n\n"
        keys = classi.keys()
        keys.sort()
        
        for i, key in enumerate(keys):
            contents += "Class %s : %i results \n" % (key, len(classi[key]))
            contents += "--------------------------------------------\n"
            contents += "%s\n" % classi[key]
            contents += "--------------------------------------------\n"
        self.write_text(file_name, contents)
            


    def write_text(self, file_name, contents):
        c_file_name = self.get_text_file_name(file_name)
        fo = open(c_file_name, "w")
        fo.write(contents)
        fo.close()
            
            
        


    def plot_energy_vs_dipole_moment_classified(self, remove_impossible=True):
        """
            Plots the energy as a function of dipole moment and classifies the 
            results with the number of AAD-AAD bonds
            - Also removes the angles that are impossible if the parameter 'remove_impossible_angles'
              is set True
        """
        c = Classification(options = self.options)
        #classi = c.do_default_grouping(nozeros=self.nozeros)   
        classi = c.do_default_grouping(nozeros=self.nozeros) 
        #classi = c.do_default_grouping_for_all() 
        energy_list =  self.read_energies()
        dipole_moment_list = self.read_dipole_moments()
        cs = self.get_results()
        image_name = "energy_dipole_moment_classified"
        if remove_impossible:
            cs = c.remove_impossible_angles(cs)
            image_name += "_nz"
        else:
            cs = self.list_as_dict(cs)

        wm_energy = self.get_water_molecule_energy()
        cis_factors = c.get_cis_factor_values(nozeros = self.nozeros)
        markers = ['o', 'v', '^', 's']
        legend = []
        
        # sort class keys by name
        keys = classi.keys()
        keys.sort()
        N_H2O = len(self.water_algorithm.oxygen_coordinates)

        for i, key in enumerate(keys):
            energyd = self.get_indeces(cs, (self.get_indeces(classi[key], energy_list)))
            #ciss = self.get_indeces(classi[key], cis_factors)
            energies = (np.array(energyd.values()) - (N_H2O * wm_energy)) #/ N_H2O 
            energies *=  23.061
            dm =  np.array(self.get_indeces(cs, self.get_indeces(classi[key], dipole_moment_list)).values())
            if True:
                #    self.print_structures(classi[key])
                legend.append(key)
                print "Key: %s - number of results: %i" % (key, len(energies)) 
                matplotlib.pyplot.plot(dm, energies, markers[np.mod(i, len(markers))])
        matplotlib.pyplot.legend(legend, loc=4)
        #matplotlib.pyplot.xlim(5.5,7.4)
        matplotlib.pyplot.xlabel("Dipole moment (D)")
        matplotlib.pyplot.ylabel("Stabilization energy (kcal / mol)")
        if not self.subplot:
            matplotlib.pyplot.savefig(self.get_image_file_name(image_name))
            return self.show_plot()
        print "Number of samples %i" % len(energy_list)

    def print_structures(self, list_of_numbers):
        for number in list_of_numbers:
            self.print_structure(number)

    def plot_energy_vs_bond_length(self):
        """
            Plots the stabilization energy as a function of total O-O bond length
        """
        structures = self.read_structures()
        c = Classification(options = self.options)
        total_bond_lengths = c.get_total_oxygen_distances_dict(structures) 
        energies =  self.read_energies()
        image_name = "energy_bond_length"

        classi = c.group_by_aad(nozeros=self.nozeros) 
        energy_list = self.get_indeces(structures.keys(), (self.get_indeces(classi[6], energies)))
        bl_list = self.get_indeces(structures.keys(), (self.get_indeces(classi[6], total_bond_lengths)))
        bl_list = np.array(bl_list.values())

        wm_energy = self.get_water_molecule_energy()
        energy_list = (np.array(energy_list.values()) - (20 * wm_energy)) / 20
        energy_list *= 23.06
        assert len(energy_list) == len(bl_list) 
        matplotlib.pyplot.plot(bl_list, energy_list,  'o')
        
        matplotlib.pyplot.xlabel("Bond length ($\AA$)")
        matplotlib.pyplot.ylabel("Stabilization energy (kcal/mol / molecule)")
        if not self.subplot:
            matplotlib.pyplot.savefig(self.get_image_file_name(image_name))
            return self.show_plot()


    

    def get_water_molecule_energy(self, unit="eV"):
        """
            Returns the energy [unit] of a single water molecule in vacuum
            or raises an Exception if no water molecule energies are found

            Used in sublimation energy calculations
        """
        number_of_relaxation = self.number_of_relaxation
        xc = self.xc
        folder = "water_molecule/"
        while number_of_relaxation >= 0:
            energy = self.read_energy(0, folder=folder, number_of_relaxation=number_of_relaxation, xc=xc)
            if energy != None:
                if unit == "kcal/mol":
                    energy *= 23.06
                return energy
            number_of_relaxation -= 1
        # Fall back to PBE
        number_of_relaxation = self.number_of_relaxation
        xc = 'PBE'
        while number_of_relaxation >= 0:
            energy = self.read_energy(0, folder=folder, number_of_relaxation=number_of_relaxation, xc=xc)
            if energy != None:
                if unit == "kcal/mol":
                    energy *= 23.06
                return energy
            number_of_relaxation -= 1
        raise Exception("No water energies found")



    def plot_energy_vs_dipole_moment(self):
        """
            Plots the energy as a function of dipole moment
        """
        energy_list,  dipole_moment_list = self.get_energy_vs_dipole_moment()
        #for key in energy_list.keys():
        #    matplotlib.pyplot.plot(np.array(dipole_moment_list[key]) / 0.20822678, energy_list[key],  'o',  label=str(key))
        #matplotlib.pyplot.legend(energy_list.keys())
        wm_energy = self.get_water_molecule_energy()
        energies = (np.array(energy_list) - (20 * wm_energy)) / 20 
        energies *=  23.061
        matplotlib.pyplot.plot(np.array(dipole_moment_list), energies,  'o')
        matplotlib.pyplot.xlabel("Dipole moment (Debye)")
        matplotlib.pyplot.ylabel("Sublimation energy (kcal/mol / molecule)")
        if not self.subplot:
            matplotlib.pyplot.savefig(self.get_image_file_name("energy_dipole_moment"))
            return self.show_plot()
        print "Number of samples %i" % len(energy_list)

    def plot_energy_vs_angle_deviation(self):
        c = Classification(options = self.options)
        energy_list =  self.read_energies()
        baa = c.do_bond_angle_analysis()
        #for key in energy_list.keys():
        #    matplotlib.pyplot.plot(np.array(dipole_moment_list[key]) / 0.20822678, energy_list[key],  'o',  label=str(key))
        #matplotlib.pyplot.legend(energy_list.keys())
        wm_energy = self.get_water_molecule_energy()
        energies = (np.array(energy_list.values()) - (20 * wm_energy)) / 20 
        energies *=  23.061
        matplotlib.pyplot.plot(np.array(baa.values())[:, 0], energies,  'o')
        matplotlib.pyplot.xlabel("Angle deviation (Debye)")
        matplotlib.pyplot.ylabel("Sublimation energy (kcal/mol / molecule)")
        print "Number of samples %i" % len(energy_list)
        
        if not self.subplot:
            matplotlib.pyplot.savefig(self.get_image_file_name('energy_angle_deviation'))
            return self.show_plot()

    def get_small_dipole_moments(self, limit=2):
        energies = self.read_energies()
        dipole_moments = self.read_dipole_moments()
        count = 0
        for i, number  in enumerate(energies.keys()):
            if dipole_moments[number] < limit:
                count += 1
                print "Dipole moment and energy for %i: %f Debye %f eV" % (number,dipole_moments[number], energies[number])
        print "Total number of clusters below %f Debye: %i" % (limit, count)

    def get_small_dipole_moment_estimates(self, limit=0.7):
        dipole_moments = self.get_dipole_moments()
        count = 0
        result = []
        for i, dipole_moment  in enumerate(dipole_moments):
            if dipole_moment < limit:
                count += 1
                result.append(i)
                print "Dipole moment and energy for %i: %f WM" % (i, dipole_moment)
        print "Total number of clusters below %f WM: %i" % (limit, count)
        return result

    def get_large_dipole_moment_estimates(self, limit=10):
        dipole_moments = self.get_dipole_moments()
        count = 0
        result = []
        for i, dipole_moment  in enumerate(dipole_moments):
            if dipole_moment > limit:
                count += 1
                result.append(i)
                print "Dipole moment and energy for %i: %f WM" % (i, dipole_moment)
        print "Total number of clusters above %f WM: %i" % (limit, count)
        return result
            
    def get_sample(self):
        energies = self.read_energies()
        dipole_moments = self.read_dipole_moments()
        minimum_numbers = np.zeros((10),  dtype='int')
        minimum_energies = np.zeros((10))
        max_dipole_moment = max(dipole_moments.values())
        for i, number  in enumerate(energies.keys()):
            group = math.floor((dipole_moments[number] / max_dipole_moment) * 9)
            if energies[number] < minimum_energies[group]:
                minimum_energies[group] = energies[number]
                minimum_numbers[group] = number
        for number in minimum_numbers:
            print "Dipole moment and energy for %i: %f Debye %f eV" % (number,dipole_moments[number], energies[number])
        return minimum_numbers

    def print_default_sample(self, estimate=True):
        self.print_sample(self.get_sample(), estimate=estimate, only_dm=True)

    def plot_energies(self, xcs=["PBE", "vdW-DF2", "TS09", "tml"]):
        """
            xcs: Exchange correlation methods calculated 
        """
        N = len(self.handle_result_instances)
        M = len(xcs)
        ys = numpy.zeros((M, N))
        
        for i in range(self.get_number_of_results()):
            for j, xc in enumerate(xcs):
                self.xc = xc
                ys[j][i] = self.get_stabilization_energy(i, unit="kcal/mol", latest=True)
        
        print ys

        # Plot       
        markers = ['o-', 'v-', '^-', 's-'] 
        x = numpy.arange(N)+1
        for i, y in enumerate(ys):
            matplotlib.pyplot.plot(x, y, markers[i])

        matplotlib.pyplot.xlim((0, N+1))
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.legend(xcs)
        #matplotlib.pyplot.xticks(numpy.arange(1, N+1), names )

    def print_sample(self, numbers, estimate=False, only_dm=False):
        if not only_dm:
            energies = self.read_energies()
        if estimate:
            dipole_moments = self.get_dipole_moments()
            print "Doing estimate"
        else:
            dipole_moments = self.read_dipole_moments()
        if not only_dm:
            energy =  "["
        dm = "["
        first = True
        for number in numbers:
            if first:
                first = False
            else:
                if not only_dm:
                    energy += ", "
                dm += ", "
            dm_value = dipole_moments[number]
            if estimate:
                dm_value *= (25.90/6.58730844238)
            dm += str(dm_value)
            if not only_dm:
                energy += str(energies[number])
        dm += "]"
        if not only_dm:
            energy += "]"
        print dm
        if not only_dm:
            print energy

    def get_minimum_energy(self, unit="eV", latest=False):
        result = -1
        minimum = 0
        energies = self.read_energies(latest=latest)
        for key, energy in energies.iteritems():
            if energy < minimum:
                minimum = energy
                result = key
        if minimum != None:
            if unit == "kcal/mol":
                return result, minimum*23.06
        
        return result, minimum


    def get_minimum_stabilization_energy(self, unit="eV", latest=False):
        no, energy = self.get_minimum_energy(unit, latest=latest)
        if energy != None:
            wm_energy = self.get_water_molecule_energy(unit=unit)
            energy -= (20.0 * wm_energy)
        return energy

    def get_stabilization_energy(self, number, unit="eV", latest=False):
        if latest:
            energy = self.read_latest_energy(number)
        else:
            energy = self.read_energy(number)

        if energy != None:
            wm_energy = self.get_water_molecule_energy(unit=unit)
            energy -= (20.0 * wm_energy)
        return energy


    def print_structure(self, number):
        energy = self.read_energy(number)
        dipole_moment = self.read_dipole_moment(number)
        if dipole_moment != None and energy != None:
            N_H2O = self.water_algorithm.N
            dipole_moment_mag = np.linalg.norm(dipole_moment)
            wm_energy = self.get_water_molecule_energy()
            print "The index of structure is %i" % number
            print "  - Energy is %f eV (%f kcal/mol)" % (energy, energy*23.061)
            print "  - Sublimation energy is %f eV (%f kcal/mol)" % ((energy-(N_H2O* wm_energy)), (energy-(N_H2O * wm_energy))*23.061)
            print "  - Sublimation energy per molecule is %f eV (%f kcal/mol)" % ((energy-(N_H2O * wm_energy))/N_H2O, (energy-(N_H2O * wm_energy))/N_H2O*23.061)
            print "  - It's dipole moment's magnitude is %f D" % (dipole_moment_mag)
            print "  - It's dipole moment vector is %s" % (dipole_moment)



    def get_maximum_energy(self):
        result = -1
        maximum = -100000
        energies = self.read_energies()
        for key, energy in energies.iteritems():
            if energy > maximum:
                maximum = energy
                result = key
        return result, maximum
        
    def print_minimum_energys_number(self):
        key, energy = self.get_minimum_energy()
        print "# Minimum energy structure #"
        self.print_structure(key)
        """
        print "The index of minimum energy structure is %i" % result
        print "  - Energy is %f eV (%f kcal/mol)" % (minimum, minimum*23.061)
        print "  - Sublimation energy is %f eV (%f kcal/mol)" % ((minimum-(20 * wm_energy)), (minimum-(20 * wm_energy))*23.061)
        print "  - Sublimation energy per molecule is %f eV (%f kcal/mol)" % ((minimum-(20 * wm_energy))/20, (minimum-(20 * wm_energy))/20*23.061)
        print "  - It's dipole moment is %f D" % (self.read_dipole_moments()[result])
        """

    def print_minimum_dm_number(self):
        result = -1
        minimum = 100
        dms = self.read_dipole_moments()
        for key, dm in dms.iteritems():
            if dm < minimum:
                minimum = dm
                result = key
        print "The index of minimum dipole moment structure is %i" % result
        print "  - Dipole moment is %f D" % (minimum)
        minimum = self.read_energy(result)
        wm_energy = self.get_water_molecule_energy()
        print "  - Sublimation energy is %f eV (%f kcal/mol)" % ((minimum-(20 * wm_energy)), (minimum-(20 * wm_energy))*23.061)
        print "  - Sublimation energy per molecule is %f eV (%f kcal/mol)" % ((minimum-(20 * wm_energy))/20, (minimum-(20 * wm_energy))/20*23.061)
        
    def get_energy_range(self):
        key, max_energy = self.get_maximum_energy()
        key, min_energy = self.get_minimum_energy()
        return max_energy-min_energy, max_energy, min_energy

    def print_energy_range(self):
        rang, maxi, mini = self.get_energy_range()
        print "Energy range is %f eV (%f kcal/mol)" % (rang, rang*23.06)
        print "Energy range per molecule is %f eV (%f kcal/mol)" % (rang/20.0, rang/20.0*23.06)  


    def print_maximum_energys_number(self):
        key, energy = self.get_maximum_energy()
        print "# Maximum energy structure #"
        self.print_structure(key)
    
    def plot_dipole_moments_r(self):
        matplotlib.rcParams['font.size'] = 16
        result = np.array(self.read_dipole_moments().values())
        matplotlib.pyplot.hist(result, bins=100)
        matplotlib.pyplot.xlabel("Dipole moment (Debye)")
        matplotlib.pyplot.ylabel("Amount of water clusters")
        if not self.subplot:
            matplotlib.pyplot.savefig(self.image_folder+'calculated_dipole_moments.png')
            return self.show_plot()
    
    def plot_dipole_moments_pre(self):
        x = self.get_dipole_moments()
        matplotlib.pyplot.hist(x, bins=100)
        matplotlib.pyplot.xlabel("Dipole moment (water molecules)")
        matplotlib.pyplot.ylabel("Amount of water clusters")
        if not self.subplot:
            matplotlib.pyplot.savefig(self.image_folder+'estimated_dipole_moments.png')
            return self.show_plot()

    def plot_ir_spectrum(self, number_list):
        number = int(number_list[0])
        wavenumbers, intensities, absorbance = self.get_ir_spectrum(number)
        matplotlib.pyplot.plot(wavenumbers, intensities, "-")
        matplotlib.pyplot.xlabel("$cm^{-1}$")
        matplotlib.pyplot.ylabel("Absorbance")
        if not self.subplot:
            matplotlib.pyplot.savefig(self.get_image_file_name("ir_spectra_%i" % number))
            return self.show_plot()
        
    def get_ir_spectrum(self, number):
        filename = self.folder+"relaxed/relaxed_structure_%i/ir-spectra.dat" % (number)
        f = open(filename)
        lines = f.readlines()
        f.close()
        intensities = []
        absorbance = []
        wavenumbers = []
        for line in lines:
            words = line.split()
            wavenumbers.append(words[0])
            intensities.append(words[1])
            absorbance.append(words[2])
        return np.array(wavenumbers, dtype=float), np.array(intensities, dtype=float), np.array(absorbance, dtype=float)

    
    def parse_where(self, where_string):
        sentences = []
        sentence = []
        if where_string is not None:
            for word in where_string:
                if word.lower() != 'and' and word.lower() != 'or':
                    sentence.append(word)
                else:
                    #print "AND or OR found" 
                    #assert word.lower() != 'or'
                    sentences.append(sentence)
                    sentence = []
            sentences.append(sentence)
            return sentences

    def filter_data(self, data,  where):
        if where == None or len(where) == 0:
            return data
        sentences = where
        # ALL separated with ANDS for now
        remove = []
        if "number" not in data:
            data["number"] = self.initialize_data("number")
        keys = data["number"]
        final_keys = None
        for sentence in sentences:
            length_before = len(remove)
            words = ""
            for word in sentence:
                words += word + " " 
            if sentence[0] not in data:
                data[sentence[0]] = self.initialize_data(sentence[0])
            if final_keys != None:
                keys = final_keys
            final_keys = []    
            words = words.replace(" =", " ==")
            for key in keys:
                if (type(data[sentence[0]]) == dict and key in data[sentence[0]]) or ((type(data[sentence[0]]) == list or type(data[sentence[0]]) == np.ndarray) and len(data[sentence[0]]) > key):
                    new_words = words.replace(sentence[0], str(data[sentence[0]][key]))
                    sentence_fulfilled = eval(new_words)
                else:
                    sentence_fulfilled = False
                #
                if sentence_fulfilled:
                    final_keys.append(key)
                   
            
            #print "Filtered %i results with %s" % (len(data['number']) - len(final_keys), sentence)
        result = {}
        for column in data:
            result[column] = {}
            for key in final_keys: 
                #if key not in data[column]:
                #    #result[column][key] = 0.0
                #    continue
                #else:
                if type(data[column]) == dict and key not in data[column]:
                    result[column][key] = None
                else:
                    result[column][key] = data[column][key]
                
            
        return result



    

    def initialize_data(self, column_name):
        fields = ["ring_type"]
        result = self.read_column_data(column_name)
        
        
        
        if result == None:
            product_split = column_name.split("*")
            sum_split = column_name.split("+")
            minus_split = column_name.split("-")
            if len(product_split) > 1:
                result = {}
                temp_data = {}
                for cname in product_split:
                    if not is_number(cname):
                        temp_data[cname] = self.initialize_data(cname)
                # get the indeces
                if self.islist(temp_data[product_split[0]]):
                    indeces = range(0, len(temp_data[product_split[0]]))
                else:
                    indeces = temp_data[product_split[0]].keys()
                # do the summations
                for number in indeces:
                    for cname in product_split:
                        if number not in result:
                            result[number] = 1
                        if is_number(cname):
                            result[number] *= float(cname)
                        else:
                            result[number] *= temp_data[cname][number]
                        
            elif len(sum_split) > 1:
                result = {}
                temp_data = {}
                # initialize individual columns
                for cname in sum_split:
                    temp_data[cname] = self.initialize_data(cname)
                # get the indeces
                if self.islist(temp_data[sum_split[0]]):
                    indeces = range(0, len(temp_data[sum_split[0]]))
                else:
                    indeces = temp_data[sum_split[0]].keys()
                # do the summations
                for number in indeces:
                    for cname in sum_split:
                        if number not in result:
                            result[number] = 0
                        if is_number(cname):
                            result[number] += float(cname)
                        else:
                            result[number] += float(temp_data[cname][number])
            elif len(minus_split) > 1:
                result = {}
                temp_data = {}
                for cname in minus_split:
                    if not is_number(cname):
                        temp_data[cname] = self.initialize_data(cname)
                # get the indeces
                if self.islist(temp_data[minus_split[0]]):
                    indeces = range(0, len(temp_data[minus_split[0]]))
                else:
                    indeces = temp_data[minus_split[0]].keys()
                # do the summations
                for number in indeces:
                    for i, cname in enumerate(minus_split):
                        if i == 0:
                            mul = 1
                        else:
                            mul = -1
                        if number not in result:
                            result[number] = 0
                        if is_number(cname):
                            result[number] += mul * float(cname)
                        else:
                            result[number] += mul * float(temp_data[cname][number])
            
            else:
                if column_name.startswith("ring_type") or column_name.startswith("g_ring_type") or column_name.startswith('non_trans'):
                    c = Classification(options = self.options)
                    c.get_ring_types(5)
                    c.get_ring_types(4)
                    c.get_ring_types(6)
                elif column_name.startswith("DD") or column_name.startswith("AA") or column_name.startswith("AD"):
                    c = Classification(options = self.options)
                    if "_" not in column_name:
                        c.get_molecule_type_counts(self.get_results())
                    else:
                        c.get_all_bond_types(self.get_results())
                elif column_name == 'energy':
                    self.read_energies()
                elif  column_name == 'relative_energy':
                    self.get_relative_energies(unit = "eV")
                elif column_name == 'relative_energy_kcalmol':
                    self.get_relative_energies(unit = "kcalmol")   
                elif column_name == 'sublimation_energy_kcalmol':
                    self.get_sublimation_energies(unit="kcalmol")
                elif column_name == 'sublimation_energy_eV':
                    self.get_sublimation_energies(unit="eV")
                elif column_name == 'symmetry_group':
                    self.get_symmetry_groups()
                elif column_name == 'dipole_moment':
                    self.read_dipole_moments()
                elif column_name.startswith("estimated_dipole_moment"):
                    self.get_dipole_moments()
                elif column_name == 'impossible_angle_count':
                    c = Classification(options = self.options)
                    c.get_impossible_angle_counts(self.get_results())
                elif column_name == 'structure_altered' or column_name == "elongated_hydrogen_bonds" or column_name == "elongated_bonds":
                    self.get_structures_altered()
                elif column_name == 'original_core_electrostatic_energy':
                    self.calculate_original_core_electrostatic_energies()
                elif column_name == 'core_electrostatic_energy':
                    self.calculate_core_electrostatic_energies()
                elif column_name == 'number':
                    result = {}
                    for number in range(self.get_number_of_results()):
                        result[number] = number
                    return result
                    

                result = self.read_column_data(column_name)
                if result == None:
                    raise Exception("No such column name: %s" % column_name)
        #print "Found %i with column %s" % (len(result), column_name)
        return result    
                
                    
            
    def parse_groups(self, indeces, group_by, data):
        """
            Grouping method that does the GROUP BY parts of
            SQL-like queries. Mostly depends on classification.py
            module
            
            
        """
        if group_by == None or len(group_by) == 0:
            return None 
        c = Classification(options = self.options)
        # make sure that all the data that we are grouping with,
        # are initialized
        for i, column_name in enumerate(group_by):
            group_by[i] = column_name.replace(",","")
            if group_by[i] not in data:
                data[group_by[i]] = self.initialize_data(group_by[i])
        
        
        # do the actual grouping
        groups = c.group_by(indeces, group_by, data)
        # collapse the group tree to groups with comma separated keys
        groups = c.collapse_groups(groups)
        return groups

    def select_data(self, column_names=['dipole_moment', 'energy'], where=None, group_by=None, order_by = None):
        data = {}
        for i, column_name in enumerate(column_names):
            data[column_name] = self.initialize_data(column_name)
        if 'number' not in data:
            data['number'] = self.initialize_data('number')
        data = self.filter_data(data, where)
        groups = self.parse_groups(data['number'],  group_by, data)
        tables = []
        if groups is not None:
            keys = groups.keys()
            keys.sort()
            for i, key in enumerate(keys):
                title = "Group %s" % (key) 
                tables.append(self.initialize_table(column_names, data, numbers = groups[key], title = title)) 
                    
        else:
            tables.append(self.initialize_table(column_names, data))
            
        return tables

    def select(self, column_names=['dipole_moment', 'energy'], where=None, group_by=None, order_by = None):
        where_sentences = self.parse_where(where)
        result = self.do_select(column_names = column_names, where = where_sentences, group_by = group_by, order_by = order_by)
        file_name = ""
        for column_name in column_names:
            file_name += "_"+column_name
        if where != None:
            file_name += "_WHERE"
            for where_arg in where:
                file_name += where_arg
        if group_by != None:
            file_name += "_GROUP_BY"
            for groupby_arg in group_by:
                file_name += groupby_arg
        print result
        self.write_text(file_name, result)

    def do_select(self, column_names=['dipole_moment', 'energy'], where=None, group_by=None, order_by = None):
        """
            Method responsible for SELECT queries
        """
        tables = self.select_data(column_names = column_names, where = where, group_by = group_by, order_by = order_by)
        result = ""
        for table in tables:
            result += self.print_table(table)
        return result

    def fit(self, column_names, fit_to = ['energy'], where = None):
        """
            Method responsible for FIT queries
        """
        data = {}
        for i, column_name in enumerate(column_names):
            data[column_name] = self.initialize_data(column_name)
        data[fit_to[0]] = self.initialize_data(fit_to[0])
        data = self.filter_data(data, where)
        coefficient_matrix, dependent_variable = self.create_fit_matrix(column_names, fit_to, data)
        fit = np.linalg.lstsq(coefficient_matrix, dependent_variable)
        fit_dict = self.get_fit_dict(column_names, fit[0])
        self.print_fit(column_names, fit[0])
        
        print "Residues"
        print fit[1]
        print "per cluster"
        print fit[1] / len(dependent_variable)
        print "Rank"
        print fit[2]
        print "Singular"
        print fit[3]
        fit = fit[0]
        print "Pearson r"
        import scipy.stats
        fit_values = self.get_fit_x_values(coefficient_matrix, fit)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(fit_values, dependent_variable)
        
        print "r"
        print r_value
        print "r^2"
        print r_value**2
        print "standard_error"
        print std_err
        print "RMSE"
        rms_err = rmse(fit_values, dependent_variable)
        self.fit_values = {'fit_dict': fit_dict, 'r': r_value, 'standard_error': std_err, 'rms_err': rms_err } 

        import matplotlib.pyplot as plt
        y = self.get_fit_x_values(coefficient_matrix, fit)
        plt.plot(dependent_variable, y, 'o', label='Data')
        x_label = self.get_column_label(fit_to[0])
        plt.plot(y, y, 'r', label='Fit')
        plt.xlabel(x_label)
        plt.ylabel("Fitted function value")
        plt.legend(loc = 4, fontsize=12)
        
        return self.show_plot()
        
    def print_fit(self, column_names, fit):
        result = None
        for i, column_name in enumerate(column_names):
            if result != None:
                result += "+"
            else:
                result = ""
            result += "%g [%s] " % (fit[i], column_name)
        result += "+ %g" % fit[-1] 
        print result
        
    def get_fit_dict(self, column_names, fit):
        result = {}
        for i, column_name in enumerate(column_names):
            result[column_name] = fit[i]
        result['constant'] = fit[-1]
        return result 


    def get_fit_x_values(self, coefficient_matrix, fit):
        shape = coefficient_matrix.shape
        l =  shape[0]
        result = np.zeros_like(coefficient_matrix[:, 0])    
        for i, fit_parameter in enumerate(fit):
            result += fit_parameter * coefficient_matrix[:, i] 
        return result

    def create_fit_matrix(self, column_names, fit_to, data):
        coefficient_matrix = []
        dependent_variable = []
        for number in range(self.get_number_of_results()):
            perfect_data = True
            c = np.zeros((len(column_names)+1))
            for i, column_name in enumerate(column_names):
                data_exists = (type(data[column_name]) == dict and number in data[column_name]) or (self.islist(data[column_name]) and number < len(data[column_name]))
                if not data_exists:
                    perfect_data = False
                    print column_name +" not perfect for %i" % number
                    break
                else:
                    c[i] = data[column_name][number]
            c[i + 1] = 1
            
            perfect_data = perfect_data and number in data[fit_to[0]]
            if perfect_data:
                coefficient_matrix.append(c)
                dependent_variable.append(data[fit_to[0]][number])
        return np.array(coefficient_matrix), np.array(dependent_variable)

    def initialize_table(self, column_names, data, row_table = False, numbers = None, title = None):
        if numbers is None:
            if "number" not in data: 
                data["number"] = self.initialize_data("number")
            numbers = data["number"]
        nan = float('NaN')
        table_data = np.empty((len(numbers), len(column_names)))
        for i, number in enumerate(numbers):
            for j, column_name in enumerate(column_names):
                if column_name == "number" or column_name == "id":
                    value = number
                else:
                    if type(data[column_name]) == dict:
                        if number in data[column_name]:
                            value = data[column_name][number]
                        else:
                            value = nan
                    else:
                        if len(data[column_name]) > number:
                            value = data[column_name][number]
                        else:
                            value = nan
                table_data[i, j] = value
        table = Table(column_names = column_names, data = table_data, row_table = row_table, title = title, page = 0)
        return table
    

    def print_table(self, table, latex = False):
        
        if latex:
            separator = " & "
        else:
            separator = " "
        assert table.column_names != None and len(table.column_names) > 0
        #if numbers == None:
        #    if "number" not in data: 
        #        data["number"] = self.initialize_data("number")
        #    numbers = data["number"]
        title = None
        if table.title is not None:
            title = "%s, Group has %i configurations" % (table.title, table.data.shape[0])
        header = None
        if table.row_table:
            header = ""
        else:
            for column_name in table.column_names:
                if header is None:
                    header = "%-20s" % column_name
                else:
                    header += separator + "%-19s " % column_name
        if latex:
            header += "\\"
        header += "\n"
        
        contents = ""
        for i, result_i in enumerate(table.all_rows()):
            row = ""
            if table.row_table:
                row += "%-20s & " % column_name
            for j, result_ij in enumerate(result_i): 
                if j == 0 and not table.row_table:
                    row += "%-20.10g" % result_ij
                else:
                    row += separator + "%-20.10g" % result_ij           
            if latex:
                row += "\\" 
            row += "\n"
            contents += row
        return header + contents
            
    def sort_by(self, sort_by, numbers, data):
        import operator
        sort_columns = sort_by.replace(" ", "").split(",")
        sorted_column_tuple = None
        reverse = None
        sort_list = np.empty((len(sort_columns), len(numbers)), dtype=np.float)
        for i, column in enumerate(sort_columns):
            if column != "ASC" and column != "DESC":
                if column not in data:
                    data[column_name] = self.initialize_data(column_name)
                for j, number in enumerate(numbers):
                    sort_list[i, number] = data[column][number]
            elif column == "DESC" or column == "ASC":
                if i != len(sort_columns)-1:
                    Exception("DESC and ASC can only occur at the end of sort statement.")
                if column == "DESC":
                    reverse = True
                else:
                    reverse = False
        if reverse is not None:
            indexes = np.lexsort(sort_list[0: len(sort_columns)-1])
            if reverse:
                indexes = indexes[::-1]
        else:
            indexes = np.lexsort(sort_list)
        return [numbers[i] for i in indexes]
                
                    
                                     
            

    def plot(self, column_names=['dipole_moment', 'energy'], where=None, group_by=None, order_by=None):
        data = {}
        # initialize columns we are representing
        for i, column_name in enumerate(column_names):
            data[column_name] = self.initialize_data(column_name)
            
        # initialize numbers, which will be used as indeces in grouping
        if 'number' not in data:
            data['number'] = self.initialize_data('number')
        
        # do the WHERE part of the query
        data = self.filter_data(data, where)
        
        # do the GROUP BY part of the query
        groups = self.parse_groups(data['number'],  group_by, data)
        # check if there were nothing to group with
        if groups is not None:
            # groups found, sort keys alphabetically
            legend = []
            markers = ['o', 'v', '^', 's']
            keys = groups.keys()
            keys.sort()
            # plot each group as "scatter" plot
            for i, key in enumerate(keys):
                x = self.get_indeces(groups[key], data[column_names[0]])
                y = self.get_indeces(groups[key], data[column_names[1]])
                matplotlib.pyplot.plot(x.values(), y.values(), markers[np.mod(i, len(markers))])
                legend.append(key)
                
            legend_title = ""
            for i, group_by_column in enumerate(group_by):
                if i != 0:
                    legend_title += ", "
                legend_title += self.get_column_label(group_by_column)
              
            #matplotlib.pyplot.annotate(legend_title, (0,1), (-0.2, -0.2), xycoords='figure fraction', textcoords='offset points', va='top',  fontsize = 10)
            #matplotlib.pyplot.legend(legend, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0. , prop = {'size':12})
            legend_instance = matplotlib.pyplot.legend(legend, loc=4, numpoints = 1,  prop = {'size':12}) 
            #legend_instance.get_title().size=10
                
        else:
            # no groups found: just represent everything as one dataset
            x = data[column_names[0]]
            y = data[column_names[1]]
            # handle the different data types (list, np.ndarrays, and dicts)
            if type(x) == dict and type(y) == dict:
                matplotlib.pyplot.plot(x.values(), y.values(),   'o')
            elif self.islist(x) and self.islist(y):
                if len(x) > len(y):
                    x = x[:len(y)]
                else:
                    y = y[:len(x)]
                matplotlib.pyplot.plot(x, y,   'o')
            elif type(x) == dict and self.islist(y):
                y = self.get_indeces(x.keys(), y)
                matplotlib.pyplot.plot(x.values(), y.values(),   'o')
            elif self.islist(x) and type(y) == dict:
                x = self.get_indeces(y.keys(), x)
                matplotlib.pyplot.plot(x.values(), y.values(),   'o')
        # get the labels for the plot
        matplotlib.pyplot.xlabel(self.get_column_label(column_names[0]))
        matplotlib.pyplot.ylabel(self.get_column_label(column_names[1]))
        
        # generate the image name        
        image_name = ""
        
        for column_name in column_names[0:2]:
            image_name += "_"+column_name
        if where != None:
            image_name += "_WHERE"
            for where_arg in where:
                for word in where_arg:
                    image_name += "_%s" % word
        if group_by != None:
            image_name += "_GROUP_BY"
            for groupby_arg in group_by:
                image_name += groupby_arg
        
        # finally if we are not subplotting, show the plot
        if not self.subplot:
            matplotlib.pyplot.savefig(self.get_image_file_name(image_name))
            return self.show_plot()
        
    def get_column_label(self, column_name):
        """
            Gets the column label for pyplot
            Handles the +,-,*,/ symbols and passes the 
            word determination to get_single_column_label
        """
        label = ""
        product_split = column_name.split("*")
        divide_split = column_name.split("/")
        sum_split = column_name.split("+")
        minus_split = column_name.split("-")
        if len(product_split) > 1:
            for i, word in enumerate(product_split):
                if i > 0:
                    label += " $\\times$ "
                label += self.get_column_label(word)
            return label
        
        elif len(divide_split) > 1:
            for i, word in enumerate(divide_split):
                if i > 0:
                    label += " $/$ "
                label += self.get_column_label(word)
            return label
        
        elif len(sum_split) > 1:
            for i, word in enumerate(sum_split):
                if i > 0:
                    label += " $+$ "
                label += self.get_column_label(word)
            return label
        elif len(minus_split) > 1:
            for i, word in enumerate(minus_split):
                if i > 0:
                    label += " $-$ "
                label += self.get_column_label(word)
            return label
        else:
            return self.get_single_column_label(column_name)
            
    def get_single_column_label(self, column_name):
        labels = {'estimated_dipole_moment': 'Dipole Moment (water molecule)', 
                  'original_core_electrostatic_energy': 'E$_{c, coul}$'}
        label = ""
     
        if column_name in labels:
            print "here", column_name
            label = labels[column_name]
            print label
        if column_name == "A" or column_name == "B" or column_name.startswith("AA") or column_name.startswith("AD") or column_name.startswith("DD"):
            label = self.get_chain_label(column_name)
        label += self.get_column_unit_label(column_name)  
        return label
        
    def get_chain_label(self, column_name):  
        """
            Returns the pyplot label for chains like AAD_AAD, ADD_ADD_ADD, etc.
        """  
        label = ""
        components = column_name.split("_")
            
        # three molecule chains with directions
        if len(components) == 4:
            chain_type = int(components[3])
            if chain_type == 0:
                arrow1 = arrow2 = "\\rightarrow"
            elif chain_type == 1:
                arrow1 = "\\rightarrow"
                arrow2 = "\\leftarrow"
            elif chain_type == 2:
                arrow1 = "\\leftarrow"
                arrow2 = "\\rightarrow"
            elif chain_type == 3:
                arrow1 = "\\leftarrow"
                arrow2 = "\\leftarrow"
            label += "$a_{%s %s %s %s %s}$" % (components[0], arrow1, components[1], arrow2, components[2])
        # three molecule chains without directions
        elif len(components) == 3:
            label += "$a_{%s-%s-%s}$" % (components[0], components[1], components[2])
        # two molecule chains
        elif len(components) == 2:
            label += "$b_{%s \\rightarrow %s}$" % (components[0], components[1])
        return label
            
    def get_column_unit_label(self, column_name):
        column_unit = None
        if self.is_energy_column(column_name):
            column_unit = self.original_energy_unit(column_name)
        if column_unit is not None:
            return "(%s)" % column_unit
        else:
            return ""
  
    def is_energy_column(self, column_name):
        """
            Checks if the 'column_name' is an energy column
        """
        energy_columns = {}
        return 'energy' in column_name or column_name in energy_columns
        
    def original_energy_unit(self, column_name):
        column_units = {'original_core_electrostatic_energy': 'J'}
        if column_name in column_units:
            return column_units[column_name]
        return "eV" 
        
    def islist(self, object1):
        return object1 is not None and (type(object1) == list or type(object1) == np.ndarray)

    def write_relaxed_structure_image(self, number):
        number = self.handle_number_input(number)
        atoms = self.read(number)
        if atoms is not None:
            dir, filename = self.get_directory_and_filename(number, extension = "eps")
            filename = dir + "/" + filename
            ase.io.write(filename, atoms) 
    
    def view_relaxed_structure(self, number):
        number = self.handle_number_input(number)
        try:
            from subprocess import call
            trajectory_path = self.get_latest_structure_path(number, extension = "traj", basis_set = self.basis_set, xc = self.xc, program = self.program, method = self.method)
            if trajectory_path is not None:
                print "ag:ing %s" % trajectory_path
                call(["ag", trajectory_path])
            else:
                print "Structure with number %i has not been relaxed, with given settings." % number
        except:
            self.handle_system_error()
            import ase.visualize
            atoms = self.read_latest_structure_from_file(number, basis_set = self.basis_set, xc = self.xc, program = self.program, method = self.method)
            if atoms is not None:
                ase.visualize.view(atoms)
            else:
                print "Structure with number %i has not been relaxed, with given settings." % number

    def view_charge_transfer(self, number):
        """
            Reads the latest additional_atoms - ice structure charge transfer calculation
             that best matches the input parameters and plots it averaged along 'axis'
        """
        
        number = self.handle_number_input(number)
        try:
            cube = self.read_latest_cube_from_file(number, extension = "_ct.cube", basis_set = self.basis_set, xc = self.xc, program = self.program, method = self.method)
            if cube is not None:
                number_of_dimensions, axis, axis_range, mark_positions = self.get_charge_plot_input()
                if number_of_dimensions == 1:
                    self.charge_plot(cube, axis = axis, label="Charge transfer ($e/\AA$)", mark_positions = mark_positions)
                else:
                    self.charge_2d_plot(cube, axis = axis, label = r"Charge Transfer ($e/\AA^2$)", axis_range = axis_range, mark_positions = mark_positions)
            else:
                print "Charge transfer calculations have not been performed for structure with number %i." % number
        except:
            import traceback
            import sys
            traceback.print_exc()

    def view_hydrogen_bond_charge_transfer(self, number):
        """
            Reads the latest charge hydrogen bond  transfer calculation that best matches the input parameters and 
            plots it averaged along 'axis'
        """
        number = self.handle_number_input(number)
        try:
            cube = self.read_latest_cube_from_file(number, extension = "_hbct.cube", basis_set = self.basis_set, xc = self.xc, program = self.program, method = self.method)
            if cube is not None:
                number_of_dimensions, axis, axis_range, mark_positions = self.get_charge_plot_input()
                if number_of_dimensions == 1:
                    self.charge_plot(cube, axis = axis, label="Charge transfer ($e/\AA$)", mark_positions = mark_positions)
                else:
                    self.charge_2d_plot(cube, axis = axis, label = r"Charge Transfer ($e/\AA^2$)", axis_range = axis_range, mark_positions = mark_positions)
            else:
                print "Charge transfer calculations have not been performed for structure with number %i." % number
        except:
            import traceback
            import sys
            traceback.print_exc()
            
    def get_charge_plot_input(self):
        
        # Handle number of dimensions input
        number_of_dimensions_str = raw_input("Please specify the number of dimensions you want to plot. [default: 1]")
        if number_of_dimensions_str == "2":
            number_of_dimensions = 2
        elif number_of_dimensions_str == 1 or number_of_dimensions_str == "":
            number_of_dimensions = 1
        else:
            raise CommandLineInputException("Invalid value for axis start range. Please give 'x', 'y', or 'z'.")
            
        # Handle axis input
        axis_str = raw_input("Please give the axis [x, y, z], from which you want to do the plot. [default: z]: ")
        if axis_str == "":
            axis = 2
        elif axis_str == "x":
            axis = 0
        elif axis_str == "y":
            axis = 1
        elif axis_str == "z":
            axis = 2
        else:
            raise CommandLineInputException("Invalid value for axis. Please give 'x', 'y', or 'z'.")
                
        # Get axis plot range for 2d plot
        if number_of_dimensions == 2: 
            start_str = raw_input("Please give starting value for axis range, from which you want to do a 2d-plot. [Default: None]: ")
            if start_str == "":
                start = None
            else:
                try:
                    start = float(start_str)
                except:
                    raise CommandLineInputException("Invalid value (%s) for axis start range. Please give a floating point number.")
                        
            # If no input is given for start, the 'axis_range' is None
            if start is not None:
                end_str = raw_input("Please give ending value for axis range, from which you want to do a 2d-plot: ")
                try:
                    end = float(end_str)
                except:
                    raise CommandLineInputException("Invalid value for axis end range. Please give a floating point number.")
                axis_range = [start, end]
            else:
                axis_range = None
        else:
            axis_range = None
                
        # Mark atom positions input
        mark_positions_str = raw_input("Are the atom positions marked in the output? (y/n) [default: y]: ")
        if mark_positions_str == "" or mark_positions_str == "y" or mark_positions_str == "yes":
            mark_positions = True
        elif mark_positions_str == "n" or mark_positions_str == "no":
            mark_positions = False;
        else:
            print "Invalid value for input. Please give 'y' or 'n'."
            import sys
            sys.exit()
        return number_of_dimensions, axis, axis_range, mark_positions
        
    
    def handle_number_input(self, number):
        number_of_results = self.get_number_of_results()
        if number_of_results == 0:
            raise CommandLineInputException("Result structures were not found. Please do --execute before this operation.")
            
        fail = False
        if number is None:
            fail = True
        elif  type(number) == list:
            number = number[0]
        
        # If there has not been failures, try to parse an int from the 
        # input
        if not fail:
            try:
                number = int(number)
            except: 
                fail = True
            
        # If parsing or something else failed, return None
        if fail:
            raise CommandLineInputException("Invalid input for visualize_bonding_energy: please give an integer between 0 and %i." % number_of_results )
            return None
            
        return number
            
            
    

    def view_charge_density(self, number):
        
        number = self.handle_number_input(number)
        if number is not None:
            try:
                cube = self.read_latest_cube_from_file(number, extension = "cd.cube", basis_set = self.basis_set, xc = self.xc, program = self.program, method = self.method)
                if cube is not None:
                    # prompt for some information
                    number_of_dimensions, axis, axis_range, mark_positions = self.get_charge_plot_input()
                    if number_of_dimensions == 1:
                        self.charge_plot(cube, axis = axis, mark_positions = mark_positions)
                    else:
                        self.charge_2d_plot(cube, axis = axis, label = r"Charge Transfer ($e/\AA^2$)", axis_range = axis_range, mark_positions = mark_positions)
                else:
                    print "Charge density calculations have not been performed for structure with number %i." % number
            except:
                import traceback
                import sys
                traceback.print_exc()
    
    def view_electron_density(self, number):
        
        number = self.handle_number_input(number)
        if number is not None:
            try:
                cube = self.read_latest_cube_from_file(number, extension = "ed.cube", basis_set = self.basis_set, xc = self.xc, program = self.program, method = self.method)
                if cube is not None:
                    # set negative values to 0
                    #cube.density_matrix[cube.density_matrix < 0] = 0
                    # prompt for some information
                    number_of_dimensions, axis, axis_range, mark_positions = self.get_charge_plot_input()
                    if number_of_dimensions == 1:
                        self.charge_plot(cube, axis = axis, mark_positions = mark_positions)
                    else:
                        self.charge_2d_plot(cube, axis = axis, label = r"Electron Density ($e/\AA^2$)", axis_range = axis_range, mark_positions = mark_positions)
                else:
                    print "Charge density calculations have not been performed for structure with number %i." % number
            except:
                import traceback
                import sys
                traceback.print_exc()

    def view_electrostatic_potential(self, number, axis = 2):
        number = self.handle_number_input(number)
        if number is not None:
            try:
                cube = self.read_latest_cube_from_file(number, extension = "_ht.cube", basis_set = self.basis_set, xc = self.xc, program = self.program, method = self.method)
                
                if cube is not None:
                    print "Plotting electrostatic potential"
                    self.plot_electrostatic_potential(cube, axis = axis, label="Electrostatic Potential (V)")
                else:
                    print "Electrostatic potential calculations have not been performed for structure with number %i." % number
            except:
                import traceback
                import sys
                traceback.print_exc()

    def charge_plot(self, cube, axis = 2, label = "Charge Density ($e/\\AA$)", mark_positions = True):
        from matplotlib import pyplot as plt 
        # determine the axises we sum over
        if axis == 0:
            a = 1
            b = 1
        elif axis == 1:
            a = 0
            b = 1
        else:
            a = 0
            b = 0
        
        # sum over a, then b then divide by the dimensions to get the average charge e/Ang^3 
        # or in the case of electrostatic potential 
        a_shape = cube.density_matrix.shape[a]
        density_matrix = cube.density_matrix.sum(a)
        b_shape = density_matrix.shape[b]
        density_matrix = density_matrix.sum(b)
        #density_matrix /= a_shape * b_shape
        # multiply the number with the size of the single voxel in dimensions other than 'axis'
        # to get the e/Ang or in the case of electrostatic potential
        for i in range(3):
            if axis != i:
                density_matrix *= cube.matrix_vectors[i, i]
        x = np.linspace(0, cube.matrix_vectors[axis][axis] * (density_matrix.shape[0]), density_matrix.shape[0])
   
        # get the dimensions of the plot
        y_max = density_matrix.max()
        y_min = density_matrix.min()
        atom_positions = cube.atom_positions
  

        # plot the lines that represent the atoms
        if mark_positions:
            colors = {1: "black", 8: "red", 11: "purple", 78: "brown"}
            for i, atomic_number in enumerate(cube.atomic_numbers):
                if atomic_number in colors:
                    plt.plot([atom_positions[i, axis], atom_positions[i, axis]], [y_min, y_max], color=colors[atomic_number])
                else:
                    plt.plot([atom_positions[i, axis], atom_positions[i, axis]], [y_min, y_max], color=colors[atomic_number])

        plt.plot(x, density_matrix, color = "black", linewidth = 3)
        plt.plot(x, np.cumsum(density_matrix), color="blue", linewidth = 3)
        x1,x2,y1,y2 = plt.axis()
        plt.axis((np.min(x),np.max(x),y_min,y_max))
        
        plt.xlabel("xyz"[axis] + " ($\AA$)" )
        plt.ylabel(label)
        return self.show_plot()

    def charge_2d_plot(self, cube, axis = 2, label = r"Charge Density ($e\AA^2$)", axis_range = None, mark_positions = True):
        from matplotlib import pyplot as plt
        assert axis <= 2
        density_matrix = cube.density_matrix
        # get only the specified range of 'axis'
        if axis_range is not None:
            # assert that axis has starting and ending points
            assert len(axis_range) == 2
            start = int(axis_range[0] / cube.matrix_vectors[axis, axis])
            end = int(axis_range[1] / cube.matrix_vectors[axis, axis])
            if axis == 0:
                density_matrix = density_matrix[start:end, :, :]
            elif axis == 1:
                density_matrix = density_matrix[:, start:end, :]
            else:
                density_matrix = density_matrix[:, :, start:end]
         
        # sum over a, then b then divide by the dimensions to get the average charge e/Ang^3 
        # or in the case of electrostatic potential 
        density_matrix = density_matrix.sum(axis)
        if axis == 0:
            a = 1
            b = 2
        elif axis == 1:
            a = 0
            b = 2
        else:
            a = 0
            b = 1    
        
        #density_matrix /= a_shape * b_shape
        
        # multiply the number with the size of the single voxel in dimensions of 'axis'
        # to get the e/Ang^2 or in the case of electrostatic potential
        density_matrix *= cube.matrix_vectors[axis, axis]
        # get the dimensions of the plot
        x_min = 0
        x_max = (cube.density_matrix.shape[a] -1) * cube.matrix_vectors[a][a]
        y_min = 0
        y_max = (cube.density_matrix.shape[b] -1) * cube.matrix_vectors[b][b]

        # formulate the plot
        import matplotlib.cm as cm
        maximum = np.max(density_matrix)
        abs_minimum = np.abs(np.min(density_matrix))
        if maximum > abs_minimum:
            v = maximum
        else:
            v = abs_minimum
        p = plt.imshow(density_matrix.T, cmap='seismic', vmin = -v, vmax = v, extent=(x_min, x_max, y_min, y_max), origin="lower")
        cb = plt.colorbar(p)
        cb.set_label(label)
        
        #tom_positions = cube.atom_positions
  

        # plot the dots that represent the atoms
        if mark_positions:
            colors = {1: "black", 8: "red", 11: "purple", 78: "brown"}
            # remove axis columns from positions
            positions = cube.atom_positions[:, [a, b]]
            for i, atomic_number in enumerate(cube.atomic_numbers):
                draw = axis_range is None or cube.atom_positions[i, axis] > axis_range[0] and cube.atom_positions[i, axis] < axis_range[1]  
                if draw:
                    if atomic_number in colors:
                        plt.scatter([positions[i, 0]], [positions[i, 1]], color=colors[atomic_number])
                    else:
                        plt.scatter([positions[i, 0]], [positions[i, 1]], color='black')
    
        #plt.plot(x, density_matrix, color = "black", linewidth = 3)
        #x1,x2,y1,y2 = plt.axis()
        #plt.axis((np.min(x),np.max(x),y_min,y_max))
        
        plt.xlabel("xyz"[a] + " ($\AA$)" )
        plt.ylabel("xyz"[b] + " ($\AA$)" )
        return self.show_plot()

    def plot_electrostatic_potential(self, cube, axis = 2, label = "Electrostatic Potential (V)"):
        # We assert that the cube units are in atomic units, 
        # and change the unit from atomic units to Volts
        cube.density_matrix *= 27.21138505 
        # determine the axises we sum over
        if axis == 0:
            a = 1
            b = 1
        elif axis == 1:
            a = 0
            b = 1
        else:
            a = 0
            b = 0
        
        # sum over a, then b then divide by the dimensions to get the average electrostatic potential
        # at the axis point 
        a_shape = cube.density_matrix.shape[a]
        density_matrix = cube.density_matrix.sum(a)
        b_shape = density_matrix.shape[b]
        density_matrix = density_matrix.sum(b)
        density_matrix /= a_shape * b_shape

        # Determine the 'jump' in the electrostatic potential by finding the gradient zero points
        # and determining their minimum and maximum values, and calculating their separation
        grad = np.gradient(density_matrix)
        min_value = None
        max_value = None
        for i, grad_value in enumerate(np.abs(grad)):
            if grad_value < 1e-3:
                if min_value is None or density_matrix[i] < min_value:
                    min_value = density_matrix[i]
                elif max_value is None or density_matrix[i] > max_value:
                    max_value = density_matrix[i]
        print "'Jump' in the electrostatic potential: %f V" % (max_value - min_value)

        # determine the x axis
        x = np.arange(0, cube.matrix_vectors[axis][axis] * cube.density_matrix.shape[axis], cube.matrix_vectors[axis][axis])
   
        # get the dimensions of the plot
        y_max = density_matrix.max()
        y_min = density_matrix.min()
        atom_positions = cube.atom_positions

        # plot the lines that represent the atoms
        colors = {1: "white", 8: "red", 11: "purple", 78: "brown"}
        for i, atomic_number in enumerate(cube.atomic_numbers):
            if atomic_number in colors:
                plt.plot([atom_positions[i, axis], atom_positions[i, axis]], [y_min, y_max], color=colors[atomic_number])
            else:
                plt.plot([atom_positions[i, axis], atom_positions[i, axis]], [y_min, y_max], color=colors[atomic_number])

        plt.plot(x, density_matrix, color = "black", linewidth = 3)

        # edit the dimonsions to better fit the plot
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,y_min,y_max))
        
        plt.xlabel("xyz"[axis] + " ($\AA$)" )
        plt.ylabel(label)
        return self.show_plot()
        
    def show_plot(self):
        p = plt.show()
        return None

    def view_additional_atoms_charge_density(self, number, axis = 2):
        number = self.handle_number_input(number)
        if number is not None:
            try:
                cube = self.read_latest_cube_from_file(number, extension = "_cda.cube", basis_set = self.basis_set, xc = self.xc, program = self.program, method = self.method)
                if cube is not None:
                    print "Plotting electron density"
                    self.charge_plot(cube, axis = axis)
                else:
                    print "Charge transfer calculations have not been performed for structure with number %i." % number
            except:
                import traceback
                import sys
                traceback.print_exc()

    def view_ice_charge_density(self, number, axis = 2):
        number = self.handle_number_input(number)
        if number is not None:
            try:
                cube = self.read_latest_cube_from_file(number, extension = "_cdi.cube", basis_set = self.basis_set, xc = self.xc, program = self.program, method = self.method)
                if cube is not None:
                    print "Plotting ice charge density"
                    self.charge_plot(cube, axis = axis)
                else:
                    print "Charge transfer calculations have not been performed for structure with number %i." % number
            except:
                import traceback
                import sys
                traceback.print_exc()


    def execute_commands(self, options, args=None):
        """
            The commandline function. Handles all actions called from the 
            command line.
        """
        self.xc = options.xc
        self.fmax = options.fmax
        self.program = options.program
        self.method = options.method
        self.number_of_relaxation = options.number_of_relaxation
        #self.nozeros = options.nozeros
        #if options.nozeros != self.nozeros and self.water_algorithm != None:
        #    self.results = self.get_results()
        try:
            # Check if the called functionality is a plain function
            if hasattr(options, 'function') and options.function is not None: 
                try:
                    method = getattr(self, str(options.function))
                    if options.function == 'plot' or options.function == 'select':
                        while True:
                            command = raw_input("GIVE the SQL type command: \n")
                            if command == "quit" or command == "exit" or command == "q":
                                break
                            arguments = command.split()
                            arguments, where, group_by, order_by = parse_args(arguments)
                            method(arguments, where, group_by, order_by)
                    elif args != None and len(args) > 0:
                        method(args)
                    else:
                        method()
                except TypeError:
                    print "Not enough arguments specified for '--%s'. Please check --help for more information." % options.function
                    self.handle_system_error()
            # If it is not a function, check if the arguments are an SQL-like query
            elif args != None and len(args) > 0:
                first_argument = args[0].upper()
                if first_argument.startswith("SELECT"):
                    method = getattr(self, "select")
                elif first_argument.startswith("PLOT"):
                    method = getattr(self, "plot")  
                elif first_argument.startswith("FIT"):
                    method = getattr(self, "fit")  
                else:
                    return 
                
                # Parse the SQL-like query FROM THE FIRST argument
                arguments = args[0].split()[1:]
                arguments, where, to, group_by, order_by = parse_args(arguments)
                
                # Do the actual function calling. Fitting does not need 'order_by' parameters
                if first_argument.startswith("SELECT") or first_argument.startswith("PLOT"):
                    method(arguments, where, group_by, order_by)
                else:
                    method(arguments, to, where)
        
        except CommandLineInputException as e:
            print "HERE"
            print e.message
        except:
            self.handle_system_error()
            

    def print_nearest_neighbors_number(self, number):
        number = self.handle_number_input(number)
        if number is not None:
            if self.get_results() != None:
                if number > self.get_number_of_results():
                    raise Exception("The structure number is too big.")
                self.water_algorithm.print_nearest_neighbors_nos_wo(self.get_results()[number])
            else:
                raise Exception("Result structures were not found. Please do --execute before this operation.")

    def view_result_number(self, number):
        number = self.handle_number_input(number)
        result = self.read(number)
        #if number > self.get_number_of_results():
        #    raise Exception("The structure number is too big.")
        ase.visualize.view(result)

    def write_structure(self, number):
        from structure_commons import write_result_to_file
        number = self.handle_number_input(number)
        wo = self.load_single_result(number)
        if wo is not None:
            folder = self.get_folder()
            if not os.path.exists(folder+"originals/"):
                os.makedirs(folder+"originals/")
            additional_atoms = None
            if (self.additional_atoms is not None and self.additional_atoms != '' and os.path.exists(self.additional_atoms)):
                additional_atoms = ase.io.read(self.additional_atoms)
            write_result_to_file(self.oxygen_coordinates, wo, number, self.cell,  self.nearest_neighbors_nos,  folder=folder+"originals/", periodic = self.periodic, additional_atoms = additional_atoms, vacuum = self.vacuum)
            print "Geometry number %i written to folder \"%s\" successfully." % (number, folder+"originals/")
        else:
            print "Geometry could not be loaded. Assert that there is enought results or that the algorithm has been executed with \"--execute\"."


    def write_structure_image(self, number):
        number = self.handle_number_input(number)
        wo = self.load_single_result(number)
        if wo is not None:
            folder = self.get_folder()
            if not os.path.exists(folder+"images/"):
                os.makedirs(folder+"images/")
            atoms = self.get_atoms(wo)
            rotation = '%fx,%fy,%fz' % (self.x_rotation, self.y_rotation, self.z_rotation)
            kwargs = {
               
            }
            if self.image_type == 'pov':
                

                kwargs.update({
                    'run_povray'   : False, # Run povray or just write .pov + .ini files
                    'display'      : False,# Display while rendering
                    'pause'        : True, # Pause when done rendering (only if display)
                    'transparent'  : False,# Transparent background
                    'canvas_width' : 500, # Width of canvas in pixels
                    'canvas_height': None, # Height of canvas in pixels
                    'camera_dist'  : 50.,  # Distance from camera to front atom
                    'image_plane'  : None, # Distance from front atom to image plane
                    'camera_type'  : 'perspective', # perspective, ultra_wide_angle
                    'point_lights' : [],             # [[loc1, color1], [loc2, color2],...]
                    'area_light'   : [(2., 3., 40.), # location
                                      'White',       # color
                                      .7, .7, 3, 3], # width, height, Nlamps_x, Nlamps_y
                    'background'   : 'White',        # color
                    'textures'     : None, # Length of atoms list of texture names
                    'celllinewidth': 0.1,  # Radius of the cylinders representing the cell
	                    })
            ase.io.write(folder+'images/original_structure_%i.%s' % (number, self.image_type), atoms, rotation = rotation, radii = self.atom_radius, **kwargs)

    def write(self):
        wos = self.load_results(False)
        if wos is not None:
            folder = self.get_folder()
            if not os.path.exists(folder+"originals/"):
                os.makedirs(folder+"originals/")
            additional_atoms = None
            if (self.additional_atoms is not None and self.additional_atoms != '' and os.path.exists(self.additional_atoms)):
                additional_atoms = ase.io.read(self.additional_atoms)
            write_results_to_file(self,  self.oxygen_coordinates, wos, self.cell,  self.nearest_neighbors_nos,  folder=folder+"originals/", periodic = self.periodic, additional_atoms = additional_atoms, vacuum = self.vacuum)
            print "Geometries written to folder \"%s\" successfully." % (folder+"originals/")
        else:
            print "Geometries could not be loaded. Execute the algorithm first with \"--execute\"."

    def get_symmetry_group(self, number):
        number = self.handle_number_input(number)
        if self.get_results() != None:
            if number > self.get_number_of_results():
                raise Exception("The structure number is too big.")
            atoms = self.read(number)
            return get_periodic_symmetry_group(atoms)
            
        else:
            raise Exception("Result structures were not found. Please do --execute before this operation.")

    def get_symmetry_groups(self):
        result = {}
        for number in range(self.get_number_of_results()):
            result[number] = self.get_symmetry_group(number)
        self.write_dict_to_file(result, "symmetry_group")
        
                
    def read_column_data(self, column_name):
        try:
            if column_name.startswith("AA") or column_name.startswith("AD") or column_name.startswith("DD") or column_name.startswith("molecule_"):
                fname = self.get_data_file_name(column_name, universal = True)
                if self.debug:
                    print "Trying to read %s" % fname
                lines = open(fname).read().splitlines()
                result = []
                for i, line in enumerate(lines):
                    result.append(int(line))    
            else:
                result = self.read_dict_from_file(column_name)
        except IOError:
            return None
        return result


    def read_dict_from_file(self, filename):
        import pickle
        pkl_file = open(self.get_data_file_name(filename), 'rb')
        result = pickle.load(pkl_file)
        pkl_file.close()
        return result

    def visualize_lattice_constant_optimization(self, number = None):
        import pickle
        if number is not None:
            number = self.handle_number_input(number)
            directory, filename = self.get_directory_and_filename(number, lattice_constant = self.O_O_distance)    
        else:
            directory, filename = self.get_directory_and_filename(number, lattice_constant = self.O_O_distance, additional_atoms_calculation = True)  
              
        filename = directory + "/" + self.get_identifier_for_result_file("lc_optimization_summary") + ".p"  
        if os.path.exists(filename):
            output = open(filename, 'rb')
            results = pickle.load(output)
            output.close()
            x = []
            y = []
            minimum = None
            minimum_lattice_constant = None
            for lattice_constant in results:
                energy = results[lattice_constant]
                print "%.4f: %.4f" % (lattice_constant, energy)
                x.append(lattice_constant)
                y.append(results[lattice_constant])
                if minimum is None or energy < minimum:
                    minimum = energy
                    minimum_lattice_constant = lattice_constant
            print "Minimum energy lattice constant: %.4f Ang (%.4f eV)" % (minimum_lattice_constant, minimum)
            
            matplotlib.pyplot.plot(x, y,  'o')
            matplotlib.pyplot.xlabel("Lattice constant (Ang)")
            matplotlib.pyplot.ylabel("Energy ($eV$)")
            return self.show_plot()
        else:
            print "Path '%s' not found" % filename
      
      
    def visualize_bonding_energy(self, number):
        """
            Visualizes all bonding energy calculations for the given structure
        """
        # validate input
        number = self.handle_number_input(number)
        symbols = ["o-", "o-", "o-", "o-", "^-", "^-", "^-", "^-"]
        if number is not None:
            file_paths = self.get_latest_file_paths(number, extension = "benes.txt")
            number_of_molecules = len(self.atoms)
            x = np.arange(number_of_molecules + 1, 0, -1)
            plot_count = 0
            labels, order_indices = self.get_labels_from_file_paths(file_paths)
            for l, j in enumerate(order_indices):
                file_path = file_paths[j]
                try:
                    energies = np.loadtxt(file_path)
                except IOError:
                    print "Could not read '%s'" % file_path
                    continue
                # Divide the energies by the number of molecules times the energy of a single molecule
                for i in range(number_of_molecules + 1):
                    molecule_count = (number_of_molecules + 1 - i)
                    energies[i] -= (energies[-1] *  molecule_count)
                    energies[i] /= molecule_count
                    
                # Call plot
                matplotlib.pyplot.xlabel("Number of molecules")
                matplotlib.pyplot.ylabel("Bonding energy / molecule ($eV$)")
                matplotlib.pyplot.plot(x, energies,  symbols[l], label=labels[j])
                plot_count += 1
            
            # if anything has been plotted, show it
            if plot_count != 0:  
                matplotlib.pyplot.legend()   
                return self.show_plot()  
            else:
                print "No bonding energy calculations have been performed for structure number %i. You can perform the calculations using '--calculate_bonding_energies' functionality of 'icepkg_energy'." % number

    def print_neutron_powder_diffraction(self, number = None):
        try:
            import misc.diffraction as diffraction
        except ImportError:
            print "Diffraction module was not found."
        
        if number is not None:
            # validate input
            number = self.handle_number_input(number)
            atoms = self.read(number)
        else:
            # if no number is specified do the diffraction for 
            # atoms = self.atoms
            atoms = self.get_average_atoms()
        peak_intensities, peak_angles = diffraction.powder_diffraction_for_atoms(atoms)
        diffraction.print_peak_information(peak_intensities, peak_angles)
        diffraction.plot_diffraction_spectrum(peak_intensities, peak_angles)
        
            
        
        

def parse_args(args):
    arguments = []
    group_by = None
    order_by = None
    to = None
    where = None
    where_passed = False
    group_passed = False
    order_passed = False
    group_by_passed = False
    order_by_passed = False
    sum_passed = False
    to_passed = False
    for arg in args:
        if arg.lower() == 'group':
            group_passed = True
            order_passed = False
            where_passed = False
            to_passed = False
        elif arg.lower() == 'order':
            order_passed = True
            group_passed = False
            where_passed = False
            to_passed = False
        elif arg.lower() == 'where':
            order_passed = False
            group_passed = False
            where_passed = True
            to_passed = False
        elif arg.lower() == 'to':
            order_passed = False
            group_passed = False
            where_passed = False
            to_passed = True
        elif arg.lower() == 'by':
            assert group_passed or order_passed
            if group_passed:                
                group_by_passed = True
                order_by_passed = False
            else:
                group_by_passed = False
                order_by_passed = True
            where_passed = False
            to_passed = False
            group_by = []
        elif where_passed:
            if where == None:
                where = []
            where.append(arg)
        elif group_by_passed:
            if group_by == None:
                group_by = []
            group_by.append(arg)
        elif order_by_passed:
            if order_by == None:
                order_by = []
            order_by.append(arg)
        elif to_passed:
            if to == None:
                to = []
            to.append(arg)
        else:
            arguments.append(arg.replace(",", ""))   
    return arguments, where, to, group_by, order_by



def get_parser():
    from optparse import OptionGroup
    parser = OptionParser(description="Handle the results")
    
    if (has_matplotlib):
        group = OptionGroup(parser, "Additional functions", "Functions that can be performed after the enumeration algorithm (--execute) is complete and allresults.txt has been saved to the folder.")   
        group.add_option("--view_structure", dest='function', const="view_result_number", action='store_const',
                          help="View result number (specify number)")
        
        group.add_option("--print_bond_variable_values", dest='function', const="print_nearest_neighbors_number", action='store_const',
                          help="Print nearest neighbors bond values  for water orientation number (specify number)")
        parser.add_option_group(group)

        group = OptionGroup(parser, "Additional post-energetic calculation functions", "Functions that can be performed after the enumeration algorithm (--execute) and energetic calculations are completed.")
        group.add_option("--write_relaxed_structure_image", dest="function", const="write_relaxed_structure_image", action='store_const',
                          help="Write the image of relaxed structure.")
        group.add_option("--view_relaxed_structure", dest='function', const="view_relaxed_structure", action='store_const',
                          help="View relaxed structure number (specify number)")
        group.add_option("--view_charge_transfer", dest='function', const="view_charge_transfer", action='store_const',
                          help="View charge transfer between the additional atoms and the ice.")
        group.add_option("--view_hydrogen_bond_charge_transfer", dest='function', const="view_hydrogen_bond_charge_transfer", action='store_const',
                          help="View charge transfer between the water molecules in the structure caused by the formation of hydrogen bonds.")
        group.add_option("--view_charge_density", dest='function', const="view_charge_density", action='store_const',
                          help="View charge density of the system.")
        group.add_option("--view_electron_density", dest='function', const="view_electron_density", action='store_const',
                          help="View electron density of the system.")
        group.add_option("--view_ice_charge_density", dest='function', const="view_ice_charge_density", action='store_const',
                          help="View charge density of the ice.")
        group.add_option("--view_additional_atoms_charge_density", dest='function', const="view_additional_atoms_charge_density", action='store_const',
                          help="View charge density of the additional atoms.")
        group.add_option("--view_electrostatic_potential", dest='function', const="view_electrostatic_potential", action='store_const',
                          help="View electrostatic potential of the system.")
        group.add_option("--visualize_lattice_constant_optimization", dest='function', const="visualize_lattice_constant_optimization", action='store_const',
                          help="Visualize lattice constant optimization (specify number)")
        group.add_option("--visualize_bonding_energy", dest='function', const="visualize_bonding_energy", action='store_const',
                          help="Visualize bonding energy calculations (specify number)")
        parser.add_option_group(group)

        group = OptionGroup(parser, "General analysis functions", "Functions used to do SQL-like queries from the execution, classification, and energetic data.")
        group.add_option("--plot", dest="function", const="plot", action='store_const',
                          help="Plot parameter as a function of another. For example: 'icepkg \"PLOT estimated_dipole_moment, energy WHERE energy != None\"'")
        group.add_option("--select", dest="function", const="select", action='store_const',
                          help="Select parameters with SQL-like syntax. \"SELECT energy, AAD_AAD, ADD_ADD WHERE energy != None GROUP BY DD\"'")
        parser.add_option_group(group)



        group = OptionGroup(parser, "Pre-energetic analysis functions", "Analysis functions that can be performed withouth the energy calculations. However --execute must have been performed.")
        group.add_option("--print_small_estimated_dipole_moments", dest='function', const="get_small_dipole_moment_estimates", action='store_const',
                          help="Find small dipole moments within estimated dipole moments")
        group.add_option("--print_large_estimated_dipole_moments", dest='function', const="get_large_dipole_moment_estimates", action='store_const',
                          help="Find large dipole moments within estimated dipole moments")
        group.add_option("--plot_estimated_dipole_moments", dest='function', const="plot_dipole_moments_pre", action='store_const',
                          help="Plot estimated dipole moments")
        group.add_option("--print_neutron_powder_diffraction", dest='function', const="print_neutron_powder_diffraction", action='store_const',
                          help="Print the intensities of neutron powder diffraction peaks.")
        parser.add_option_group(group)
        

        group = OptionGroup(parser, "Energetic analysis functions", "Functions that can be used to classify structures according to energetic. These operation are useful after structure energies are (partly) calculated.")
        #group.add_option("--get_sublimation_energy", dest='function', const="get_sublimation_energy", action='store_const',
        #                  help="Find the minimum energy among the relaxed results")
        group.add_option("--find_minimum_energy", dest='function', const="print_minimum_energys_number", action='store_const',
                          help="Find the minimum energy structure among the relaxed results")
        group.add_option("--find_maximum_energy", dest='function', const="print_maximum_energys_number", action='store_const',
                          help="Find the maximum energy structure among the relaxed results")
        group.add_option("--print_energy_range", dest='function', const="print_energy_range", action='store_const',
                          help="Print the energy range from minimum to maximum energy structure.")
        group.add_option("--find_minimum_dipole_moment", dest='function', const="print_minimum_dm_number", action='store_const',
                          help="Find the minimum dipole moment structure among the relaxed results")
        group.add_option("--boltzmann", dest='function', const="plot_dipole_moments_boltzmann", action='store_const',
                          help="Plot the boltzmann distribution at certain temperatures")
        group.add_option("--plot_energy_dipole_moment", dest='function', const="plot_energy_vs_dipole_moment", action='store_const',
                          help="Plot the energy as a function of dipole moment")
        #group.add_option("--energy_dipole_moment_classified", dest='function', const="plot_energy_vs_dipole_moment_classified", action='store_const',
        #                  help="Plot the energy as a function of dipole moment, classified results")
        #group.add_option("--energy_angle_deviation", dest='function', const="plot_energy_vs_angle_deviation", action='store_const',
        #                  help="Plot the energy as a function of angle deviation")
        group.add_option("--find_altered_structures", dest='function', const="print_altered_structures", action='store_const',
                          help="Print the numbers of structures that were altered during relaxation")
        group.add_option("--plot_dipole_moments", dest='function', const="plot_dipole_moments_r", action='store_const',
                          help="Plot calculated dipole moments")
        group.add_option("--print_small_dipole_moments", dest='function', const="get_small_dipole_moments", action='store_const',
                          help="Find small dipole moments within calculated dipole moments of relaxed structures")
        #group.add_option("--plot_energies", dest='function', const="plot_energies", action='store_const',
        #                  help="Plot the calculated energies.")
        
        group.add_option("--plot_energy_distribution", dest='function', const="plot_energy_distribution", action='store_const',
                          help="Plot the energy distribution") 
        group.add_option("--print_O_O_statistics", dest='function', const="print_average_O_O_bond_length", action='store_const',
                          help="Print statistics of O-O bond lengths in structures")
        group.add_option("--print_angle_statistics", dest='function', const="print_angle_statistics", action='store_const',
                          help="Print statistics of bond angles in structures")
        #parser.add_option("--plot_energy_vs_bond_length", dest='function', const="plot_energy_vs_bond_length", action='store_const',
        #                  help="Plot the energies as a function of bond length")
        parser.add_option_group(group)

        group = OptionGroup(parser, "IR-analysis functions", "Functions that can be used to manage IR-spectra data. These operation are useful after structure IR-spectra are (partly) calculated.")
        group.add_option("--plot_ir_spectrum", dest="function", const="plot_ir_spectrum", action='store_const',
                          help="Plot IR-spectrum. ")
        parser.add_option_group(group)
        #parser.add_option("--nozeros", 
        #              action="store_true", dest="nozeros", default=False,
        #              help="If impossible angles are removed from results")
        #parser.add_option('--additional_arguments', metavar='N', type=int, nargs='+',
        #           help='Additional input argumets for --boltzmann and --sample options')
    add_options(parser)
    return parser
 
class CommandLineInputException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)


class Table:
    def __init__(self, column_names = None, data = None, row_table = False, title = "", page = 0):
        self.column_names = column_names
        self.page = page
        self.data = data
        self.row_table = row_table
        self.title = title
        self.page = page
        self.results_per_page = 50
        self.initialize_page(0)

    def initialize_page(self, page):
        allrows = self.all_rows()
        self.page = page
        if allrows is None:
            self.rows = None
            return
        self.total = allrows.shape[0]
        start = self.results_per_page * self.page
        if allrows.shape[0] > self.results_per_page * (self.page+1):
            end = self.results_per_page * (self.page + 1)
        else:
            end = allrows.shape[0]
        self.start = start
        self.end = end
        self.rows = allrows[start:end]

    
    def all_rows(self):
        if self.row_table:
            return self.data.T
        else:
            return self.data
    
    def rows(self):
        return self.rows

    def header_row_fields(self):
        if self.row_table:
            return self.column_names
        else:
            return None

    def header_column_fields(self):
        if self.row_table:
            return None
        else:
            return self.column_names

        
    


    
