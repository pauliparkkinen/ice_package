from handle_multiple_results import HMR

class HandleWales(HMR):
    def __init__(self, handle_result_instances, folder = "wales_aggregation/"):
        super(HandleWales, self).__init__(handle_result_instances, folder)
        self.numbers = [[474,475,484,324,139,470,656,98,102,107,108,118,327,325,393,305,483,486,667,666,658,142,485,625,276,633,350,348,328,671,308,48,407,274,140,141,390,405,138,11,144,147,20,277,22,664,288,289,291,422,688,298,47,309,398,352,354,614,490,491,492,493,643,120,396,8,15,145,410,411,286,555,297,46,49,563,670,310,397,143,353,356,357,373,121,264,132,392,10,273,146,404,21,23,687,287,544,545,547,548,549,39,424,41,43,44,642,689,564,565,566,300,571,447,453,583,386,344,605,606,610,741,716,423,621,495,245,504,249,639,636]]
        #663, 497, 275,406,374,524
    def plot_weighted_ir_spectrum(self, temperature = 273.15, data_numbers = [0,3], minimum_energy_morphology_name = 'Wales'):
        super(HandleWales, self).plot_weighted_ir_spectrum(temperature, data_numbers, minimum_energy_morphology_name)


def run():
    from ice_21_wales import Wales
    from handle_results import HandleResults
    hri = []
    wa = Wales()
    hri.append(HandleResults(wa, name="Wales"))
    
    HandleWales(hri, "wales_aggregation/").run()

    
if __name__ == '__main__':
    run()
