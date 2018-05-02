import os
comm = None
size = 1
rank = 0
"""
try:
    from water_algorithm_cython import get_mpi_variables
    comm, size, rank = get_mpi_variables()
except ImportError:
    comm = None
    size = 1
    rank = 0
    #from mpi4py import MPI
    #comm = MPI.COMM_WORLD
    #try:
    #    size = comm.size
    #    rank = comm.rank
    #except ImportError:
    #    comm = None
    #    size = 1
    #    rank = 0
"""
class SystemCommons(object):
    def __init__(self, folder, debug):
        if folder is None or folder == "":
            folder = "./"
        elif folder[-1] != "/":
            folder += "/"
        if rank == 0 and not os.path.exists(folder):
            os.makedirs(folder)
        self.folder = folder
        self.debug = debug
    
    def get_folder(self):
        return get_folder(self.folder)

    def get_data_folder(self):
        data_folder = self.get_folder()+"data/"
        if rank == 0 and not os.path.exists(data_folder):
            os.makedirs(data_folder)
        return data_folder

    def get_image_folder(self):
        image_folder = self.get_folder()+"images/"
        if rank == 0 and not os.path.exists(image_folder):
            os.makedirs(image_folder)
        return image_folder
        

    def get_text_folder(self):
        text_folder = self.get_folder()+"output/"
        if rank == 0 and not os.path.exists(text_folder):
            os.makedirs(text_folder)
        return text_folder

    def get_all_oxygen_coordinates(self):
        return None
        
    def handle_system_error(self):
        print "An unexpected error occurred"
        import traceback
        import sys
        traceback.print_exc()

    

def get_folder(folder):
    if folder is None or folder == "":
        folder = "./"
    elif folder[-1] != "/":
        folder += "/"
    
    return folder

def add_options(parser, folder = None):
    from optparse import OptionGroup
    from energy_commons import add_options
    group = OptionGroup(parser, "General parameters", "Parameters that are used in every part of the program.")      
    group.add_option("-f", "--folder", type="string", dest="folder", default = folder,
                      help="The name of the folder used", metavar="FOLDER")
    group.add_option("--debug",
                      action="store_true", dest="debug",
                      help="Print debug output")
    parser.add_option_group(group) 
