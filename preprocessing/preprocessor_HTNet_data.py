try:
    from .preprocessor_base import *
except ImportError:
    from preprocessor_base import *

import sys
sys.path.insert(0, "/media/ungzs10/F8426F05426EC7C8/Zsombi/MTA/HTNet/HTNet_generalized_decoding")

import xarray as xr
#from model_utils import load_data

class HTNet_preprocessor(Preprocessor):

    Preprocessor.fs = 250 # sampling rate
    Preprocessor.line_freq = 60 # line freq = 60 Hz
    #Preprocessor.PSD_time_range = (1000,-500) # set time range of trials to use in a tuple of (first data, last data), eg (1000,-500) ignores first 1000 and last 500 datapoints. Default: PSD_time_range=(0,-1) to use whole range
    Preprocessor.PSD_freq_range = (0,125) # range of Power Spectrum, min and max freq in a tuple eg.(0,200) gives power spectrum from 0 to 199 Hz. Default: PSD_freq_range=(0,200). NOTE_: max_freq not included!

    def __init__(self, *args, **kwargs):
        super(HTNet_preprocessor, self).__init__(*args, **kwargs)
        self.config["data_source"] = "naturalistic hand move v. rest"
        self.config["save_dir"] = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data/preprocessed_data/")
        self.config["save_name"] = "HTNet_data_preprocesed.hdf5"
        self.config["default_config_name"] = 'HTNet_CONFIG'
        self.config["create_validation_bool"] = False

        self.subject_ids = ['EC01','EC02','EC03','EC04','EC05','EC06', 'EC07','EC08','EC09','EC10','EC11','EC12']
        
    def get_train_test_data(self):
        '''returns an array of [[X_train1, y_train1], [X_train2,y_train2]...]'''
        # define path to folder
        my_path = os.path.abspath(os.path.dirname(__file__))
        path_to_data = os.path.join(my_path, "../data/raw_data/naturalistic_move_v_rest/")
        ecog_lp = path_to_data + 'ecog_dataset/' # data load path
        
        X_all = []
        y_all = [] 
        X_test_all = []
        y_test_all = []

        # loop through each subject
        for subject in self.subject_ids:
            filename = ecog_lp + subject + '_ecog_data.nc'

            #get data
            data_xr = xr.open_dataset(filename)

            #transform to np array and extract X and y
            data = np.squeeze(np.asarray(data_xr.to_array()))
            X = data[:,:-1, 501:]
            X[np.isnan(X)] = 0 # set all NaN's to 0
            y = np.squeeze(data[:, -1, 0])

            #get train and test set
            test_day = int(np.unique(data_xr['events'][-1])) # test day is last day
            train_where = np.where(data_xr['events'] != test_day) 
            test_where = np.where(data_xr['events'] == test_day)

            X_train = np.squeeze(X[train_where,:,:])
            X_test = np.squeeze(X[test_where,:,:])

            y_train = y[train_where]
            y_test = y[test_where]

            #append to list
            X_all.append(X_train)
            y_all.append(y_train)
            X_test_all.append(X_test)
            y_test_all.append(y_test)

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

        return X_all, y_all, X_test_all, y_test_all

    def load_data_and_labels(self, filename):
        # should return a pair of numpy arrays of dimensions ( [trials, channels, timestep], [trials] )
        
        patient = filename[-17:-13]
        ecog_lp = filename[:-17]

        #get number of channels
        ep_data_in = xr.open_dataset(filename)
        n_chans_all = ep_data_in['channels'].shape[0]
        test_day='last'
        tlim=[-1,1]

        # load data
        X,y,X_test,y_test,sbj_order,sbj_order_test = load_data(patient, ecog_lp,n_chans_all=n_chans_all,test_day=test_day, tlim=tlim)
        X[np.isnan(X)] = 0 # set all NaN's to 0

        data = np.concatenate((X, X_test), axis=0)
        stim = np.concatenate((y, y_test), axis=0)

        return data, stim

    def train_files_from_dir(self):
        # return all the valid train files in a list
        file_list = []

        # define path to folder
        my_path = os.path.abspath(os.path.dirname(__file__))
        path_to_data = os.path.join(my_path, "../data/raw_data/naturalistic_move_v_rest/")
        ecog_lp = path_to_data + 'ecog_dataset/' # data load path

        for subject in self.subject_ids:
            filename = ecog_lp + subject + '_ecog_data.nc'
            file_list.append(filename)

        return file_list

    def test_files_from_dir(self):
        # return all the valid test files in a list
        file_list = []

        return file_list 
