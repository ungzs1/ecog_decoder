try:
    from .preprocessor_base import *
except ImportError:
    from preprocessor_base import *

from scipy.io import loadmat

class StanfordPreprocessor(Preprocessor):

    Preprocessor.fs = 1000 # sampling rate
    Preprocessor.line_freq = 60 # line freq = 60 Hz
    Preprocessor.blocksize = 3000 # minimum block size of trials, shorter trials ignored. Default: blocksize=floor(fs*0.25) window length of PSD
    Preprocessor.PSD_time_range = (1000,-500) # set time range of trials to use in a tuple of (first data, last data), eg (1000,-500) ignores first 1000 and last 500 datapoints. Default: PSD_time_range=(0,-1) to use whole range
    Preprocessor.PSD_freq_range = (0,200) # range of Power Spectrum, min and max freq in a tuple eg.(0,200) gives power spectrum from 0 to 199 Hz. Default: PSD_freq_range=(0,200). NOTE_: max_freq not included!

    def __init__(self, *args, **kwargs):
        super(StanfordPreprocessor, self).__init__(*args, **kwargs)
        self.config["data_source"] = "stanford motor basic dataset"
        self.config["save_dir"] = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data/raw_data/stanford_motorbasic/")
        self.config["save_name"] = "stanf_mot_preprocesed.hdf5"
        self.config["default_config_name"] = 'STANFORD_MOTORBASIC_CONFIG'
        self.config["create_validation_bool"] = False

        self.subject_ids = ['bp','ca','cc','de','fp','gc','hh','hl','jc','jm','jt','rh','rr','ug','wc','zt']

    def load_data_and_labels(self, filename):
        # should return a pair of numpy arrays of dimensions ( [timestep, channels], [timestep, label] )

        # load data
        raw_data = loadmat(filename)
        data = raw_data['data']
        stim = np.squeeze(raw_data['stim'].astype('int32'))

        # ignore unknown activity before and after trials
        trials_start = np.where(stim!=0)[0][0]
        trials_end = np.where(stim!=0)[0][-1] + StanfordPreprocessor.blocksize

        data = data[trials_start:trials_end,:]
        stim = stim[trials_start:trials_end]

        # rest after hand=120, rest after tongue=110
        prev_cue = 0
        for i, curr_stim in enumerate(stim):
            if curr_stim != 0 and curr_stim != prev_cue:
                prev_cue = curr_stim
            elif curr_stim == 0:
                stim[i] = prev_cue*10

        return data, stim

    def train_files_from_dir(self):
        # return all the valid train files in a list
        file_list = []

        # define path to folder
        my_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(my_path, "../data/raw_data/stanford_motorbasic/")

        # create list of file path
        ### xfile_list = [path+filename for filename in os.listdir(path)] ######### EZ A VEGLEGES, DE EGYELORE A KOVETKEZO HASZNALHATO CSAK, MERT A TOBB FAJL FORMATUMARA KULON BEOLVASAST KELL MEG IRNI!!!!
        
        for subject in self.subject_ids:
            filename = path + subject + '_mot_t_h.mat'
            file_list.append(filename)

        return file_list

    def test_files_from_dir(self):
        # return all the valid test files in a list
        file_list = []

        return file_list 
