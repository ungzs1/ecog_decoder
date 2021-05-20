# ***********************
# NEEDS TO BE FIXED !!!!!
# ***********************

try:
    from ecog.decoder.preprocessor import *
except ImportError:
    from ecog.decoder.preprocessor import *

class Bci3Preprocessor(Preprocessor):

    Preprocessor.fs = 1000 # sampling rate
    Preprocessor.line_freq = 60 # line freq = 60 Hz
    #Preprocessor.PSD_time_range = (1000,-500) # set time range of trials to use in a tuple of (first data, last data), eg (1000,-500) ignores first 1000 and last 500 datapoints. Default: PSD_time_range=(0,-1) to use whole range
    Preprocessor.PSD_freq_range = (0,200) # range of Power Spectrum, min and max freq in a tuple eg.(0,200) gives power spectrum from 0 to 199 Hz. Default: PSD_freq_range=(0,200). NOTE_: max_freq not included!

    def __init__(self, *args, **kwargs):
        super(Bci3Preprocessor, self).__init__(*args, **kwargs)
        self.config["data_source"] = "BCI Comp III"
        self.config["save_dir"] = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                               "../../data/preprocessed_data/")
        self.config["save_name"] = "bci_comp_iii_preprocesed.hdf5"
        self.config["default_config_name"] = 'BCI_COMP_III_CONFIG'
        self.config["create_validation_bool"] = False

        self.subject_ids = ['subject']
        
    def load_data_and_labels(self, filename):
        # should return a pair of numpy arrays of dimensions ( [timestep, channels], [timestep, label] )

        # load data
        data = np.load(filename[0])
        stim = np.load(filename[1])

        return data, stim

    def train_files_from_dir(self):
        # return all the valid train files in a list
    
        my_path = os.path.abspath(os.path.dirname(__file__))
        path_to_data = os.path.join(my_path, "../../data/raw_data/bci_comp_iii/Competition_train_cnt.npy")
        path_to_label = os.path.join(my_path, "../../data/raw_data/bci_comp_iii/Competition_train_lab.npy")

        file_list = [(path_to_data, path_to_label)]

        return file_list

    def test_files_from_dir(self):
        # return all the valid test files in a list
        file_list = []

        return file_list 
