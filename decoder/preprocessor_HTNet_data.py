try:
    from .preprocessor import *
except ImportError:
    from preprocessor import *

import sys
import xarray as xr


class HtnetPreprocessor(Preprocessor):
    def __init__(self, *args, **kwargs):
        super(HtnetPreprocessor, self).__init__(*args, **kwargs)
        '''        self.config["data_source"] = "naturalistic hand move v. rest"
        self.config["save_dir"] = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data',
                                               'preprocessed_data', 'HTNet_data')
        self.config["save_name"] = "HTNet_data.hdf5"
        self.config["default_config_name"] = "HTNet_CONFIG"
        self.config["create_validation_bool"] = False'''

    def get_train_test_data(self, subject):
        """
        returns X_train, y_train, X_test, y_test for a given subject
        """

        filename = self.lp + subject + '_ecog_data.nc'

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


if __name__ == '__main__':
    HtnetPreprocessor().run()
