try:
    from .preprocessor_base import *
except ImportError:
    from preprocessor_base import *

import sys
import xarray as xr


class HtnetPreprocessor(Preprocessor):
    Preprocessor.fs = 250  # sampling rate
    Preprocessor.line_freq = 60  # line freq = 60 Hz
    Preprocessor.time_range = (500, -1)  # ignore first half of the measurement (before movement onset)
    Preprocessor.freq_range = (0, 125)  # range of Power Spectrum, min and max freq in a tuple eg.(0,200) gives
    # power spectrum from 0 to 199 Hz. Default: PSD_freq_range=(0,200). NOTE_: max_freq not included!
    Preprocessor.subject_ids = ['EC01', 'EC02', 'EC03', 'EC04', 'EC05', 'EC06', 'EC07', 'EC08', 'EC09', 'EC10', 'EC11',
                                'EC12']

    def __init__(self, *args, **kwargs):
        super(HtnetPreprocessor, self).__init__(*args, **kwargs)
        self.config["data_source"] = "naturalistic hand move v. rest"
        self.config["save_dir"] = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data',
                                               'preprocessed_data', 'HTNet_data')
        self.config["save_name"] = "HTNet_data_{}_{}.hdf5".format(Preprocessor.time_range[0],
                                                                  Preprocessor.time_range[1])
        self.config["default_config_name"] = 'HTNet_CONFIG'
        self.config["create_validation_bool"] = False

        # define path to folder
        my_path = os.path.abspath(os.path.dirname(__file__))
        path_to_data = os.path.join(my_path, "../data/raw_data/naturalistic_move_v_rest/")
        self.ecog_lp = path_to_data + 'ecog_dataset/'  # data load path

    def get_train_test_data(self, subject):
        """
        returns X_train, y_train, X_test, y_test for a given subject
        """

        filename = self.ecog_lp + subject + '_ecog_data.nc'

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

        '''for subject in self.subject_ids:
        filename = ecog_lp + subject + '_ecog_data.nc'
        
        #get number of channels
        ep_data_in = xr.open_dataset(filename)
        n_chans_all = ep_data_in['channels'].shape[0]
        test_day='last' #
        tlim=[-1,1]

        # load data
        X,y,X_test,y_test,sbj_order,sbj_order_test = load_data(subject, ecog_lp,n_chans_all=n_chans_all,test_day=test_day, tlim=tlim)
        X[np.isnan(X)] = 0 # set all NaN's to 0

        X_all.append(X)
        y_all.append(y)
        X_test_all.append(X_test)
        y_test_all.append(y_test)'''

    def load_data_and_labels(self, filename):
        # should return a pair of numpy arrays of dimensions ( [timestep, channels], [timestep, label] )
        raise NotImplementedError

        '''# should return a pair of numpy arrays of dimensions ( [trials, channels, timestep], [trials] )

        patient = filename[-17:-13]
        ecog_lp = filename[:-17]

        # get number of channels
        ep_data_in = xr.open_dataset(filename)
        n_chans_all = ep_data_in['channels'].shape[0]
        test_day = 'last'
        tlim = [-1, 1]

        # load data
        X, y, X_test, y_test, sbj_order, sbj_order_test = load_data(patient, ecog_lp, n_chans_all=n_chans_all,
                                                                    test_day=test_day, tlim=tlim)
        X[np.isnan(X)] = 0  # set all NaN's to 0

        data = np.concatenate((X, X_test), axis=0)
        stim = np.concatenate((y, y_test), axis=0)

        return data, stim'''

    def train_files_from_dir(self):
        # return all the valid train files in a list
        raise NotImplementedError

    def test_files_from_dir(self):
        # return all the valid test files in a list
        raise NotImplementedError


if __name__ == '__main__':
    HtnetPreprocessor().run()
