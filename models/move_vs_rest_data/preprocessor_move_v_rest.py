import sys, os
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..'))

from ecog.decoder.preprocessor import *

import xarray as xr


class HtnetPreprocessor(Preprocessor):
    def __init__(self, *args, **kwargs):
        super(HtnetPreprocessor, self).__init__(*args, **kwargs)
        # define path to folder
        my_path = os.path.abspath(os.path.dirname(__file__))
        path_to_data = os.path.join(my_path, '..', '..', 'data', 'raw_data', 'naturalistic_move_v_rest')
        self.lp = os.path.join(path_to_data, 'ecog_dataset')  # data load path

        self.config["data_source"] = self.lp
        self.config["save_dir"] = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'data',
                                               'preprocessed_data', 'HTNet_data')
        self.config["save_name"] = "HTNet_data_nperseg250.hdf5"
        self.config["default_config_name"] = "HTNet_CONFIG"
        self.config["create_validation_bool"] = False

        self.fs = 250  # sampling rate
        self.line_freq = 60  # line freq = 60 Hz

        #noverlap = noverlap  # np.floor(fs * 0.1)
        #nperseg = nperseg  # np.floor(fs * 0.25)  # set window size and offset for PSD
        self.nperseg = 250
        self.noverlap = self.nperseg//2
        self.time_range = (500, -1)  # ignore first half of the measurement (before movement onset)
        self.freq_range = (0, 125)  # range of Power Spectrum, min and max freq in a tuple eg.(0,200) gives
        # power spectrum from 0 to 199 Hz. Default: PSD_freq_range=(0,200). NOTE_: max_freq not included!
        self.subject_ids = ['EC01', 'EC02', 'EC03', 'EC04', 'EC05', 'EC06', 'EC07', 'EC08', 'EC09', 'EC10',
                            'EC11', 'EC12']

    def get_train_test_data(self, subject):
        """
        returns X_train, y_train, X_test, y_test for a given subject
        """

        filename = os.path.join(self.lp, subject + '_ecog_data.nc')

        # get data. This dataset contains metadata as well in the form of column names. last element in the channels
        # is the label (0 or 1). label of the trials column is the day of the recording. label of the time step
        # column is the actual time relative to the movement onset (for example -1.2 is 1200 ms prior to movement
        # onset)
        data_xr = xr.open_dataset(filename)

        # transform to np array and extract X and y. data is an array of shape (trials, channels, time_step).
        # Sampling rate is 250 Hz for this dataset, and there are 1000 time steps for each trial, containing data
        # starting -2 sec prior to the movement onset to 2s after the movement onset. The middle of the array is
        # the 0 point.
        data = np.squeeze(np.asarray(data_xr.to_array()))
        X = data[:, :-1, :]
        X[np.isnan(X)] = 0  # set all NaN's to 0
        y = np.squeeze(data[:, -1, 0])

        # get train and test set
        test_day = int(np.unique(data_xr['events'][-1]))  # define test day as last day
        train_where = np.where(data_xr['events'] != test_day)  # not test days
        test_where = np.where(data_xr['events'] == test_day)  # test day

        X_train = np.squeeze(X[train_where, :, :])
        X_test = np.squeeze(X[test_where, :, :])

        y_train = y[train_where]
        y_test = y[test_where]

        return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    HtnetPreprocessor().run()
