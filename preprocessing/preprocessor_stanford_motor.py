try:
    from .preprocessor_base import *
except ImportError:
    from preprocessor_base import *

from scipy.io import loadmat

class StanfordPreprocessor(Preprocessor):

    Preprocessor.fs = 1000 # sampling rate
    Preprocessor.line_freq = 60 # line freq = 60 Hz
    Preprocessor.blocksize = 3000 # minimum block size of trials, shorter trials ignored. Default: blocksize=floor(fs*0.25) window length of PSD ##################ALTALANOSITANI KELL, WINDOW SIZE LEGYEN ALLITHATO
    Preprocessor.PSD_time_range = (1000,-500) # set time range of trials to use in a tuple of (first data, last data), eg (1000,-500) ignores first 1000 and last 500 datapoints. Default: PSD_time_range=(0,-1) to use whole range
    Preprocessor.PSD_freq_range = (0,200) # range of Power Spectrum, min and max freq in a tuple eg.(0,200) gives power spectrum from 0 to 199 Hz. Default: PSD_freq_range=(0,200). NOTE_: max_freq not included!


    def __init__(self, *args, **kwargs):
        super(StanfordPreprocessor, self).__init__(*args, **kwargs)
        self.config["data_source"] = "stanford"

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
        return file_list

    def test_files_from_dir(self):
        # return all the valid test files in a list
        file_list = []

        return file_list 