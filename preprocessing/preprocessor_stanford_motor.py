try:
    from .preprocessor_base import *
except ImportError:
    from preprocessor_base import *

from scipy.io import loadmat
from sklearn.model_selection import train_test_split


class StanfordPreprocessor(Preprocessor):
    def __init__(self, *args, **kwargs):
        super(StanfordPreprocessor, self).__init__(*args, **kwargs)
        # define path to folder
        my_path = os.path.abspath(os.path.dirname(__file__))
        path_to_data = os.path.join(my_path, '..', 'data', 'raw_data', 'stanford_motorbasic')
        self.lp = path_to_data  # data load path
        
        self.config["data_source"] = self.lp
        self.config["save_dir"] = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data', 
                                               'preprocessed_data', 'stanford_motorbasic')
        self.config["save_name"] = "stanford_mot.hdf5"
        self.config["default_config_name"] = 'stanford_mot_CONFIG'
        self.config["create_validation_bool"] = False

        self.fs = 1000  # sampling rate
        self.line_freq = 60  # line freq = 60 Hz
        self.time_range = (0, -1)  # set time range of trials to use in a tuple of (first data, last data),
        # eg (1000,-500) ignores first 1000 and last 500 datapoints. Default: time_range=(0,-1) to use whole range
        self.freq_range = (0, 200)  # range of Power Spectrum, min and max freq in a tuple eg.(0,200) gives
        # power spectrum from 0 to 199 Hz. Default: PSD_freq_range=(0,200). NOTE_: max_freq not included!
        self.subject_ids = ['bp', 'ca', 'cc', 'de', 'fp', 'gc', 'hh', 'hl', 'jc', 'jm', 'jt', 'rh', 'rr', 'ug', 'wc',
                            'zt']

        self.blocksize = 3000  # ignore trials shorter than blocksize
        self.test_size = 0.20  # for train test split, in percentage

    def get_train_test_data(self, subject):
        """
        Returns train and test data.
        :param subject: id of the patient that identifies them during reading of the data
        :return: X_train, y_train, X_test, y_test. Return shape: [(train_trials, channels), (train_trials, 1),
        (test_trials, channels), (test_trials, 1)], type is ndarray.
        """

        filename = os.path.join(self.lp, subject+'_mot_t_h.mat')

        # load data
        raw_data = loadmat(filename)
        data = raw_data['data']
        stim = np.squeeze(raw_data['stim'].astype('int32'))

        # if recording starts with movement, ignore first trial (usually short artefacts), important at subject 'de'
        if stim[0] != 0:
            data = data[np.where(stim == 0)[0][0]:, :]
            stim = stim[np.where(stim == 0)[0][0]:]

        # ignore unknown activity before and after trials
        trials_start = np.where(stim != 0)[0][0]
        trials_end = np.where(stim != 0)[0][-1] + self.blocksize + 1

        data = data[trials_start:trials_end, :]
        stim = stim[trials_start:trials_end]

        # set labels for rest after hand=120, rest after tongue=110
        prev_cue = 0
        for i, curr_stim in enumerate(stim):
            if curr_stim != 0 and curr_stim != prev_cue:
                prev_cue = curr_stim
            elif curr_stim == 0:
                stim[i] = prev_cue * 10

        # catalogue trials in list "trial_time ": [[trial onset, trial type]]
        trial_time = [(0, stim[0])]  # initialize with [time=0, trial type=y[0]]
        for n in range(1, len(stim)):
            if stim[n] != stim[n-1] and n - trial_time[-1][0] >= self.blocksize:
                trial_time.append((n, stim[n]))  # save onset time and type of trial if block was longer than blocksize

        # split up data by trials
        data_temp = []
        for i, trial in enumerate(trial_time):  # loop through all trials
            # get current trial
            trial_onset = trial[0]
            # trial_type = trial[1]
            curr_data = np.squeeze(data[trial_onset:trial_onset + self.blocksize, :])

            # ignore data outside the range of time_freq (transition between movement types)
            # curr_data = curr_data[self.time_range[0]:self.time_range[1], :]
            data_temp.append(curr_data)

        # reassign data and stim variables in the correct form
        data = np.transpose(np.asarray(data_temp), axes=[0, 2, 1])  # to output (trials, channels, time step)
        stim = np.asarray(trial_time)[:, 1]

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(data, stim, test_size=self.test_size, shuffle=False)

        return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    StanfordPreprocessor().run()
