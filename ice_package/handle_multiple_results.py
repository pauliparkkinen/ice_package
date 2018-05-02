import matplotlib
import os
import numpy
from scipy.constants import k
from optparse import OptionParser
from handle_results import parse_args
import sys 
class HMR (object) :
    def __init__(self, handle_result_instances, folder=".", rows=1, columns=1, font_size=12):
        #assert len(handle_result_instances) > 1
        self.handle_result_instances = handle_result_instances
        for instance in handle_result_instances:
            instance.subplot = True
        self.folder = folder
        self.image_folder = self.folder+"images/"
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
        self.rows = rows
        self.columns = columns
        self.font_size = font_size
 
    

    def set_parameters(self, options):
        for instance in self.handle_result_instances:
            instance.xc = options.xc
            self.xc = options.xc
            instance.fmax = options.fmax
            self.fmax = options.fmax
            instance.number_of_relaxation = options.number_of_relaxation
            self.number_of_relaxation = options.number_of_relaxation
            self.font_size = options.font_size
            self.rows = options.rows
            self.columns = options.columns
            self.x_label = options.x_label
            self.y_label = options.y_label

    def get_image_file_name(self, image_identifier):
        if self.xc != 'PBE':
            return self.image_folder+image_identifier+'_%i_%s.png' % (self.number_of_relaxation, self.xc)
        else:
            return self.image_folder+image_identifier+'_%i.png' % (self.number_of_relaxation)

    def get_parser(self):
        parser = OptionParser(description="Handle the results")
        parser.add_option("--minimum-energy", dest='method', const="print_minimum_energys_number", action='store_const', 
                          help="Find the minimum energy among the relaxed results")
        parser.add_option("--maximum-energy", dest='method', const="print_maximum_energys_number", action='store_const',
                          help="Find the maximum energy among the relaxed results")
        parser.add_option("--boltzmann", dest='method', const="plot_dipole_moments_boltzmann", action='store_const', 
                          help="Plot the boltzmann distribution at certain temperatures")
        parser.add_option("--energy_dipole_moment", dest='method', const="plot_energy_vs_dipole_moment", action='store_const', 
                          help="Plot the energy as a function of dipole moment")
        parser.add_option("--energy_dipole_moment_classified", dest='method', const="plot_energy_vs_dipole_moment_classified", action='store_const',
                          help="Plot the energy as a function of dipole moment, classified results")
        parser.add_option("--energy_angle_deviation", dest='method', const="plot_energy_vs_angle_deviation", action='store_const', 
                          help="Plot the energy as a function of angle deviation")
        parser.add_option("--small_estimated_dipole_moments", dest='method', const="get_small_dipole_moment_estimates", action='store_const', 
                          help="Find small dipole moments within estimated dipole moments")
        parser.add_option("--small_dipole_moments", dest='method', const="get_small_dipole_moments", action='store_const', 
                          help="Find small dipole moments within calculated dipole moments of relaxed structures")
        parser.add_option("--estimated_dipole_moments", dest='method', const="plot_dipole_moments_pre", action='store_const',
                          help="Plot estimated dipole moments")
        parser.add_option("--dipole_moments", dest='method', const="plot_dipole_moments_r", action='store_const', 
                          help="Plot calculated dipole moments")
        parser.add_option("--dm_dm", dest='method', const="plot_d_d", action='store_const', 
                          help="Plot estimated dipole moments compared with the calculated dipole moments")
        parser.add_option("--print_default_sample", dest='method', const="print_default_sample", action='store_const', 
                          help="Print the estimated dipole moments of default sample")   
        parser.add_option("--energy_distribution", dest='method', const="plot_energy_distribution", action='store_const',
                          help="Plot the energy distribution") 
        parser.add_option("--free_energy_distribution", dest='local_method', const="get_free_energy_distributions", action='store_const',
                          help="Plot the free energy distribution") 
        parser.add_option("--energy-range", dest='method', const="print_energy_range", action='store_const',
                          help="Print the energy range")
        parser.add_option("--minimum_energies", dest='local_method', const="plot_minimum_energies", action='store_const',
                          help="Plot the minimum energies of different exchange correlation functions")
        parser.add_option("--get_boltzmann_weights", dest='local_method', const="boltzmann_weight_table", action='store_const',
                          help="Plot the minimum energies of different exchange correlation functions")
        parser.add_option("--plot_weighted_ir", dest='local_method', const="plot_weighted_ir_spectrum", action='store_const',
                          help="Plot the weighted IR spectrum")
        parser.add_option("-x", "--exchange_correlation", type="string",
                      action="store", dest="xc", default="PBE",
                      help="The exchange and correlation method used.")
        parser.add_option("-f", "--fmax", type="float",
                      action="store", dest="fmax", default=0.05,
                      help="The exchange and correlation method used.")
        parser.add_option("-n", "--number_of_relaxation", type="int",
                      action="store", dest="number_of_relaxation", default=0,
                      help="The exchange and correlation method used.")
        parser.add_option("--x_label", type="string",
                      action="store", dest="x_label", default="",
                      help="X-label on plot.")
        parser.add_option("--y_label", type="string",
                      action="store", dest="y_label", default="",
                      help="Y-label on plot.")
        parser.add_option("-r", "--rows", dest="rows", type="int", default=1, help="The number of rows in the subplots")
        parser.add_option("-c", "--columns", dest="columns", type="int", default=1, help="The number of columns in the subplots")
        parser.add_option("--font_size", dest="font_size", type="int", default=12, help="The font size used in plots")
        parser.add_option('--additional_arguments', metavar='N', type=int, nargs='+',
                   help='Additional input argumets for --boltzmann and --sample options')
        return parser

    def plot_minimum_energies(self, xcs=["PBE", "vdW-DF2", "TS09", "tml"], nors=[0, 3, 1, 2]):
        """
            xcs: Exchange correlation methods calculated with
            nors: Number of relaxations that are taken in the graph
        """
        assert len(xcs) == len(nors)
        names = []
        legend = []
        N = len(self.handle_result_instances)
        M = len(xcs)
        ys = numpy.zeros((M, N))
        
        for i, instance in enumerate(self.handle_result_instances):
            names.append(instance.name) 
            print instance.name + ":"
            for j, xc in enumerate(xcs):
                instance.xc = xc
                instance.number_of_relaxation  = nors[j]
                ys[j][i] = instance.get_minimum_stabilization_energy(unit="kcal/mol", latest=True) 
        for j, xc in enumerate(xcs):
            minimum = min(ys[j])
            ys[j] -= minimum
        for xc in xcs:
            if xc == 'tml':
                legend.append("MP2")
            else:
                legend.append(xc)        
        
        print ys

        # Plot       
        markers = ['o-', 'v-', '^-', 's-'] 
        x = numpy.arange(N)+1
        for i, y in enumerate(ys):
            matplotlib.pyplot.plot(x, y, markers[i])
        print legend
        matplotlib.pyplot.xlim((0, N+1))
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.legend(legend)
        matplotlib.pyplot.xticks(numpy.arange(1, N+1), names )
        matplotlib.pyplot.ylabel("Stabilization energy (kcal mol$^{-1}$)")

    def boltzmann_weight_table(self):
        import matplotlib.pyplot as plt
        for hr in self.handle_result_instances:
            hr.water_algorithm.initialize_symmetry_operations()
        Ts = [273.15, 233.15, 193.15, 100]
        table = numpy.zeros(( 3, len(self.handle_result_instances), len(Ts)))
        for i, T in enumerate(Ts):
            table[0][:, i] = self.get_electronic_energy_boltzmann_weights(T, minimum_energies_only = True) * 100
            table[1][:, i] = self.get_free_energy_boltzmann_weights(T, minimum_energies_only = True) * 100
            #table[1][:, i] = self.get_electronic_energy_boltzmann_weights(T, minimum_energies_only = False) * 100
            table[2][:, i] = self.get_free_energy_boltzmann_weights(T, minimum_energies_only = False) * 100
        #table[1][:, len(Ts)] = [0, 100, 0, 0]
        #table[0][:, len(Ts)] = [0, 100, 0, 0]
        #Ts = [273.15, 253.15, 233.15, 213.15, 193.15, 100, 0]
        
        ind = numpy.arange(len(Ts))
        width = 0.5
        colors = ['b', 'r', 'g', 'y']
        #plt.legend( (p1[0], p2[0]), ('Men', 'Women') )
        for i in range(len(table)):
            plt.subplot(130+(i+1))
            plt.yticks(numpy.arange(0,101,10))
            bottom = numpy.zeros((len(table[i][0])))
            for hr_number in range(len(table[i])):
                plt.bar(ind, table[i][hr_number], width, color=colors[hr_number], bottom=bottom)
                bottom += table[i][hr_number]
                #cellText.append(['%1.1f' % (x/1000.0) for x in yoff])
            plt.xticks(ind+width/2., Ts )
            plt.ylabel('Weight (%)')
            plt.xlabel('Temperature (K)')
        plt.show()
            

    def get_electronic_energy_boltzmann_weights(self, T = 253.15, minimum_energies_only = False, single_weights = False):
        try:
            eV_to_J = 1.62177 * 10**-19
            
            #dipole_moments = get_dipole_moments()
            exponentials = {}
            energies = {}    
            total = 0
            minimum = None
            for i, hr in enumerate(self.handle_result_instances):
                hr.xc = self.xc
                hr.number_of_relaxation = self.number_of_relaxation
                energies[i] = hr.read_energies()
                minimum_energy = min(energies[i].values()) 
                if minimum == None or minimum_energy < minimum:
                    minimum = minimum_energy
            print minimum_energy
            for i, hr in enumerate(self.handle_result_instances):
                exponentials[i] = {}
                if minimum_energies_only:
                    minimum_energy = min(energies[i].values()) 
                    print "%s minimum energy is %f kcal/mol" % (hr.name, (minimum_energy-minimum)*23.06)
                    exponentials[i][0] = numpy.exp(-((minimum_energy -minimum) * eV_to_J) / (k * T))
                    total += exponentials[i][0]
                else:
                    for j, number  in enumerate(energies[i].keys()):
                        exponentials[i][number] = numpy.exp(-(((energies[i][number]) -minimum) * eV_to_J) / (k * T))
                        total += exponentials[i][number]
            
            weights = {}
            total_weights = numpy.zeros(len(self.handle_result_instances))
            for i,hr in enumerate(self.handle_result_instances):
                weights[i] = {}
                nof = 0
                nof_weights = 0.0
                for number in exponentials[i]:
                    weights[i][number] = exponentials[i][number] / total
                    total_weights[i] += weights[i][number]
                    if weights[i][number] > 0.002:
                        
                        #print weights[i][number]
                        #print (energies[i][number] + H[T][i] - S[T][i]*T) -minimum
                        nof += 1
                        nof_weights += weights[i][number]
                        #print "%i has weight of %f" % (number, weights[i][number])
                #print "%s total weight is %f" % (hr.name, total_weights[i])
                #print " has %i with weight over 0.00001, (Weight: %f)" % (nof, nof_weights)
            if single_weights:
                return weights
            return total_weights
        except:
            e = sys.exc_info()[0]
            print e 
            print "Getting electronic energy boltzmann weights failed"
            return None

    def get_free_energies(self, T = 0):
        T = float(T)
        energies = {} 
        S = {}
        H = {}
        S[273.15] = [0.0081293, 0.0078033, 0.0080337, 0.0074435]
        S[253.15] = [0.0076803, 0.0073596, 0.0075788, 0.0069976]
        S[233.15] = [0.0072222, 0.0069080, 0.0071147, 0.0065452]
        S[213.15] = [0.0067558, 0.0064494, 0.0066417, 0.0060878]
        S[193.15] = [0.0062818, 0.005985, 0.0061609, 0.0056275]
        S[100] = [ 0.0040511, 0.0038366, 0.0038990, 0.0035930]
        S[0] = [0, 0, 0, 0]
        H[273.15] = [14.863, 14.998, 14.962, 15.022]
        H[253.15] = [14.746, 14.882, 14.843, 14.906]
        H[233.15] = [14.633, 14.771, 14.729, 14.795]
        H[213.15] = [14.53, 14.67, 14.624, 14.694]
        H[193.15] = [14.434, 14.575, 14.527, 14.6]
        H[100] = [14.106, 14.26, 14.195, 14.299]
        H[0] = [14.004, 14.168, 14.098, 14.224]
        minimum = None
        maximum = None
        
        for i, hr in enumerate(self.handle_result_instances):
            hr.xc = self.xc
            hr.number_of_relaxation = self.number_of_relaxation
            energies[i] = hr.read_energies()
            minimum_energy = min(energies[i].values()) 
            maximum_energy = min(energies[i].values()) 
            H_ = H[T][i]
            S_ = S[T][i]
            minimum_energy += H[T][i] - S[T][i]*T
            maximum_energy += H[T][i] - S[T][i]*T
            if minimum == None or minimum_energy < minimum:
                minimum = minimum_energy
            if maximum == None or maximum_energy > maximum:
                maximum = maximum_energy
            for j in energies[i]:
                energies[i][j] += H[T][i] - S[T][i]*T
        return energies, minimum, maximum

    def get_electronic_energies(self):
        energies = {} 
        maximum = None
        minimum = None
        for i, hr in enumerate(self.handle_result_instances):
            hr.xc = self.xc
            hr.number_of_relaxation = self.number_of_relaxation
            energies[i] = hr.read_energies()
            minimum_energy = min(energies[i].values()) 
            maximum_energy = min(energies[i].values()) 
            
            if minimum == None or minimum_energy < minimum:
                minimum = minimum_energy
            if maximum == None or maximum_energy > maximum:
                maximum = maximum_energy
            for j in energies[i]:
                energies[i][j] += H_ - S_*T
        return energies, minimum, maximum
    
    def get_free_energy_distributions(self):
        Ts = [0, 213.15, 273.15]
        for i, T in enumerate(Ts):
            number = 100 + len(Ts) * 10 + (i+1)
            matplotlib.pyplot.subplot(number)
            
            self.get_free_energy_distribution(T, subplot = True)
        matplotlib.rcParams['font.size'] = self.font_size
        matplotlib.pyplot.savefig(self.get_image_file_name("free_energy_distribution_%f" % T))
        matplotlib.pyplot.show()
        

    def get_free_energy_distribution(self, T = 213.15, subplot = False):
        T = float(T)
        energies, minimum, maximum = self.get_free_energies(T)
        wm_energy = self.handle_result_instances[0].get_water_molecule_energy()
        for i, hr in enumerate(self.handle_result_instances):
            self.plot_single_instance_energy_distribution(energies[i], minimum, maximum, wm_energy)
        fig = matplotlib.pyplot.gcf()
        
        fig.set_size_inches(self.columns*6,self.rows*6) # Set size of plot window
        if not subplot:
            matplotlib.rcParams['font.size'] = self.font_size
            matplotlib.pyplot.savefig(self.get_image_file_name("free_energy_distribution_%f" % T))
            matplotlib.pyplot.show()

    def get_energy_distribution(self, subplot = False):
        T = float(T)
        energies = self.get_free_energies(T)
        wm_energy = self.handle_result_instances[0].get_water_molecule_energy()
        for i, hr in enumerate(self.handle_result_instances):
            self.plot_single_instance_energy_distribution(energies[i], minimum, maximum, wm_energy)
        fig = matplotlib.pyplot.gcf()
        
        fig.set_size_inches(self.columns*6,self.rows*6) # Set size of plot window
        if not subplot:
            matplotlib.rcParams['font.size'] = self.font_size
            matplotlib.pyplot.savefig(self.get_image_file_name("free_energy_distribution_%f" % T))
            matplotlib.pyplot.show()
            

    def plot_single_instance_energy_distribution(self, energies, minimum_energy, maximum_energy, wm_energy):
        enes = numpy.array(energies.values())
        enes -= minimum_energy
        
        #enes -= (20*wm_energy)
        enes *= (23.06)
            
        bins = (max(enes) - min(enes)) / 0.1
        print bins
        
        matplotlib.pyplot.hist(enes, bins = bins, histtype='stepfilled', edgecolor=None)
        matplotlib.pyplot.xlim(0, 4.5) 
        matplotlib.pyplot.ylim(0, 23) 


    def get_free_energy_boltzmann_weights(self, T = 273.15, minimum_energies_only = False, single_weights = False):
        try:
            print "Free energy bolztmann weights at %f K" % T
            print "-----------------------------------------"
            eV_to_J = 1.62177 * 10**-19
            
            #dipole_moments = get_dipole_moments()
            exponentials = {}
               
            total = 0
            energies, minimum, maximum = self.get_free_energies(T)
            for i, hr in enumerate(self.handle_result_instances):
                exponentials[i] = {}
                if minimum_energies_only:
                    minimum_energy = min(energies[i].values()) 
                    print "%s minimum energy is %f kcal/mol" % (hr.name, (minimum_energy-minimum)*23.06)
                    exponentials[i][0] =  numpy.exp(-((minimum_energy-minimum) * eV_to_J) / (k * T))
                    total += exponentials[i][0]
                else:
                    for j, number  in enumerate(energies[i].keys()):
                        #if hr.name == 'D':
                        #    exponentials[i][number] = numpy.exp(-(((energies[i][number] - 0.03) -minimum) * eV_to_J) / (k * T))
                        #else:
                        exponentials[i][number] = len(hr.water_algorithm.symmetry_operations) *  numpy.exp(-(((energies[i][number]) -minimum) * eV_to_J) / (k * T))
                        total += exponentials[i][number]
            
            weights = {}
            total_weights = numpy.zeros(len(self.handle_result_instances))
            for i,hr in enumerate(self.handle_result_instances):
                weights[i] = {}
                nof = 0
                nof_weights = 0.0
                for number in exponentials[i]:
                    weights[i][number] = exponentials[i][number] / total
                    total_weights[i] += weights[i][number]
                    if weights[i][number] > 0.001:
                        
                        #print weights[i][number]
                        #print (energies[i][number] + H[T][i] - S[T][i]*T) -minimum
                        nof += 1
                        nof_weights += weights[i][number]
                        print "%i has weight of %f" % (number, weights[i][number])
                
                print "%s total weight is %f" % (hr.name, total_weights[i])
                print " has %i with weight over 0.1 per cent, (Weight: %f)" % (nof, nof_weights)
            if single_weights:
                return weights
            return total_weights
        except:
            print "Getting free energy Boltzmann weights failed"
            return None

    def plot(self, options, args):
        if options.method != None:
            subplot = ((self.rows > 1 or self.columns > 1) and self.rows * self.columns >= len(self.handle_result_instances))
            print "Subplotting %s" % subplot
            for i, instance in enumerate(self.handle_result_instances):
                if subplot:
                    number = 100*self.rows + 10*self.columns + (i+1)
                    matplotlib.pyplot.subplot(number)
                method = getattr(instance, str(options.method))
                method()
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(self.columns*6,self.rows*6) # Set size of plot window
            matplotlib.rcParams['font.size'] = self.font_size
            matplotlib.pyplot.savefig(self.get_image_file_name(options.method))
            
            matplotlib.pyplot.show()
        elif options.local_method != None:
            method = getattr(self, str(options.local_method))
            if args == None or len(args) == 0:            
                method()
            else:
                method(*args)
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(8,10) # Set size of plot window
            matplotlib.rcParams['font.size'] = self.font_size
            matplotlib.pyplot.savefig(self.get_image_file_name(options.local_method))
            
            matplotlib.pyplot.show()
        elif args != None and len(args) > 0:
            subplot = ((self.rows > 1 or self.columns > 1) and self.rows * self.columns >= len(self.handle_result_instances))
            arguments = args[0].split()[1:]
            arguments, where, to, group_by, order_by = parse_args(arguments)
            
            for i, instance in enumerate(self.handle_result_instances):
                if args[0].upper().startswith("SELECT"):
                    method = getattr(instance, "select")
                elif args[0].upper().startswith("PLOT"):
                    method = getattr(instance, "plot")  
                elif args[0].upper().startswith("FIT"):
                    method = getattr(instance, "fit")  
                else:
                    return 

                if args[0].upper().startswith("PLOT"):
                    instance.subplot = subplot
                    if subplot:
                        number = 100*self.rows + 10*self.columns + (i+1)
                        matplotlib.pyplot.subplot(number)
                    matplotlib.pyplot.xlabel(self.x_label)
                    matplotlib.pyplot.ylabel(self.y_label)
                if args[0].upper().startswith("SELECT") or args[0].upper().startswith("PLOT"):
                    method(arguments, where, group_by, order_by)
                else:
                    method(arguments, to, where)
                print self.x_label
                print self.y_label
                matplotlib.pyplot.xlabel(self.x_label)
                matplotlib.pyplot.ylabel(self.y_label)
            if args[0].upper().startswith("PLOT"):
                fig = matplotlib.pyplot.gcf()
                fig.set_size_inches(self.columns*6,self.rows*6) # Set size of plot window
                matplotlib.rcParams['font.size'] = self.font_size
                matplotlib.pyplot.savefig(self.get_image_file_name(args[0]))
                matplotlib.pyplot.show()

    def plot_weighted_ir_spectrum(self, temperature = 273.15, data_numbers = [4, 6, 3], minimum_energy_morphology_name = 'ESPP'):
        numbers = self.numbers
        for hr in self.handle_result_instances:
            hr.water_algorithm.initialize_symmetry_operations()
        Ts = [233.15]
        table = {}
        for i, T in enumerate(Ts):
            table[T] = {}
            table[T][0] = self.get_electronic_energy_boltzmann_weights(T, minimum_energies_only = True, single_weights = True) 
            table[T][1] = self.get_electronic_energy_boltzmann_weights(T, minimum_energies_only = False, single_weights = True) 
            table[T][2] = self.get_free_energy_boltzmann_weights(T, minimum_energies_only = True, single_weights = True)
            table[T][3] = self.get_free_energy_boltzmann_weights(T, minimum_energies_only = False, single_weights = True)
  
        final_wavenumbers = {}
        final_intensities = {}
        counter = 0
        total_weight = {}
        for data_number in data_numbers:
            final_wavenumbers[data_number] = {}
            final_intensities[data_number] = {}
            total_weight[data_number] = {}
            if (data_number % 4) in table[T] and table[T][(data_number % 4)] != None:
                for i, hr in enumerate(self.handle_result_instances):
                    if data_number in [1, 3] or (data_number in [5, 7] and hr.name ==  minimum_energy_morphology_name):
                        for number in numbers[i]:
                            for T in Ts:
                                counter += 1
                                wavenumbers, intensity, absorbance = hr.get_ir_spectrum(number)
                                if number in table[T][data_number % 4][i]:
                                    weight = table[T][data_number % 4][i][number]
                                    if T not in final_wavenumbers[data_number]:
                                        final_wavenumbers[data_number][T] = wavenumbers
                                        total_weight[data_number][T] = weight
                                        final_intensities[data_number][T] = weight * intensity
                                    else:
                                        total_weight[data_number][T] += weight
                                        final_intensities[data_number][T] += weight * intensity
                    elif data_number in  [0, 2] or (data_number in [4, 6] and hr.name ==  minimum_energy_morphology_name):
                        if hr.name != 'FC':
                            number, energy = hr.get_minimum_energy()
                            print number, energy
                            for T in Ts:
                                counter += 1
                                
                                wavenumbers, intensity, absorbance = hr.get_ir_spectrum(number)
                                if 0 in table[T][data_number % 4][i]:
                                    weight = table[T][data_number % 4][i][0]
                                    if T not in final_wavenumbers[data_number]:
                                        final_wavenumbers[data_number][T] = wavenumbers
                                        total_weight[data_number][T] = weight
                                        final_intensities[data_number][T] = weight * intensity
                                    else:
                                        total_weight[data_number][T] += weight
                                        final_intensities[data_number][T] += weight * intensity
                      

        import matplotlib.pyplot as plt
        
        ax = plt.gca()
        ax.set_autoscale_on(False)
        
        for j, data_number in enumerate(data_numbers):
            number = len(data_numbers)* 100 + 10 + (j + 1)
            plt.subplot(number)
            for i, T in enumerate(Ts):
                if T in final_wavenumbers[data_number]:
                    subplot = (i != len(Ts) -1 or data_number != 2)
                    self.plot_ir_spectrum(final_wavenumbers[data_number][T] , final_intensities[data_number][T] / max(final_intensities[data_number][T]), T, subplot)
                    print "Total weight of spectrum %f" % total_weight[data_number][T]

    def plot_ir_spectrum(self, wavenumbers, intensities, temperature, subplot = True, x_start = 2000, x_end = 3850):
        import matplotlib.pyplot as plt
        plt.plot(wavenumbers, intensities, "-")
        plt.xlim(x_start, x_end)
        plt.xlabel("$cm^{-1}$")
        plt.ylabel("Intensity")
        if not subplot:
            plt.savefig(self.get_image_file_name("ir_spectra_%f" % temperature))
            plt.show()
        
        

    def run(self):
        (options, args) = self.get_parser().parse_args()
        self.set_parameters(options)
        self.plot(options, args)


def run():
    from ice_20_box import Boxes
    from ice_20 import PentagonPrisms
    from ice_20_dodeca import Dodecahedron
    from ice_20_fspp import FSPP
    from handle_results import HandleResults
    hri = []
    wa = Dodecahedron(intermediate_saves=[8, 11, 12, 13, 14, 18], folder="dodecahedron", group_saves=[19])
    hri.append(HandleResults(wa, name="D"))
    
    wa = PentagonPrisms()    
    hri.append(HandleResults(wa, name="ESPP"))
    wa = FSPP(folder="fspp")    
    hri.append(HandleResults(wa, nozeros=True, name="FSPP"))
    numbers = [[11269,27888,29208,29209,29994,29987,30018,30025,27891,4316,29813,30022,27900,29211,29214,29992,30014,24938,24943,24944,20818,27903,29986,29995,30017,24945,24949,24950,27898,24940,24947,27902,4315,29812,29814,29988,29991,29993,29996,30016,30023,27896,27897], [20224,20232,20262,20266,20465,20469,20484,20486], [97,98,107,108,197,198,207,208], []]
    
    wa =  Boxes(folder="box")
    hri.append(HandleResults(wa, nozeros = True, name="FC"))
    hmr = HMR(hri, "aggregation/")
    hmr.numbers = numbers
    hmr.run()

    
if __name__ == '__main__':
    run()
