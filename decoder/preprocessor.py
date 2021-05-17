from base import Ecog

import os
import sys

import glob
import numpy as np
import xmltodict
import h5py
import xarray as xr

from scipy.signal import butter, lfilter, iirnotch, filtfilt, welch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


def butter_bandpass(lowcut, highcut, fs, order=5, btype="band"):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype=btype)
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=0, btype="band"):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order, btype=btype)
    y = lfilter(b, a, data, axis=axis)
    return y


def notch_filtering(data, fs, line_freq, freq_range):
    # removes line frequency noise and multiples from the signal

    multiples_linefreq = range(line_freq, freq_range[1], line_freq)  # line frequency and multiples eg. [60,120,180]

    for f0 in multiples_linefreq:
        w0 = f0 / (fs / 2)  # Normalized Frequency
        b, a = iirnotch(w0, fs)  # Numerator ('b') and denominator ('a') polynomials of the IIR filter
        data = filtfilt(b, a, data, axis=2)

    return data


def car(data):
    # this function calculates and returns the common avg reference of 2-d matrix "data" of shape [timestep, channel].
    data = np.double(data)
    num_chans = data.shape[1]

    # create a CAR spatial filter
    spatfiltmatrix = -np.ones((num_chans, num_chans))
    for i in range(num_chans):
        spatfiltmatrix[i, i] = num_chans - 1
    spatfiltmatrix = spatfiltmatrix / num_chans

    # perform spatial filtering
    for i in range(data.shape[0]): data[i] = np.dot(spatfiltmatrix, data[i])

    return data


def get_spectra(x, fs, freq_range):
    """
    returns PSD as "all_psd" in form of [frequencies, channels, trials]. power spectrum densitiy for each channel
    for each trial
    """

    noverlap = np.floor(fs * 0.1);
    nperseg = np.floor(fs * 0.25)  # set window size and offset for PSD
    all_psd = []  # PSD for all trials and channels

    for curr_data in x:  # loop through all trials
        block_psd = []  # PSD for features in current 'block' ie all channels for this trial
        # get Power Spectrum Density with signal.welch
        for ch in range(x.shape[1]):  # loop through all channels
            [f, temp_psd] = welch(curr_data[ch, :], nfft=fs, fs=fs, noverlap=noverlap, nperseg=nperseg)
            temp_psd = temp_psd[
                np.where((freq_range[0] <= f) & (f <= freq_range[1]))]  # to get the desired freq range only
            block_psd.append(temp_psd)

        block_psd = np.asarray(block_psd)
        all_psd.append(block_psd)

    all_psd = np.asarray(all_psd)

    return all_psd


def config_post(path, key, value):
    if key.endswith("value"):
        try:
            return key, int(value)
        except (ValueError, TypeError):
            return key, value

    elif key.endswith("bool"):
        try:
            return key, bool(value)
        except (ValueError, TypeError):
            return key, value

    return key, value


class Preprocessor(Ecog):
    def __init__(self, config_file="", config={}, fs=None, line_freq=None, time_range=None, freq_range=None, *args, **kwargs):
        super(Preprocessor, self).__init__(*args, **kwargs)

        if config_file != "" and os.path.exists(config_file):
            with open(config_file) as fd:
                try:
                    self.config = xmltodict.parse(fd.read(), dict_constructor=dict, postprocessor=config_post)
                    self.config = self.config["root"]
                    self.config.update(config)
                except Exception:
                    self.config = config
        elif config != {}:
            self.config = config
        else:
            self.config["data_source"] = self.lp
            self.config["save_dir"] = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data',
                                                   'preprocessed_data', self.database_name)
            self.config["save_name"] = "ecog_preprocessed.hdf5"
            self.config["default_config_name"] = "CONFIG"
            self.config["create_validation_bool"] = False

        self.fs = fs  # sampling rate
        self.line_freq = line_freq  # line freq = 60 Hz
        self.time_range = time_range  # ignore first half of the measurement (before movement onset)
        self.freq_range = freq_range

    def get_train_test_data(self, subject):
        # return self.raw_data[subject]
        raise NotImplementedError

    def preprocess_with_label(self, x, y):

        """

        Parameters:
            x - numpy array with shape [timestep, ... ]
            y - numpy array with shape [timestep, ... ]

        Return:
            px - numpy array with shape [N, ...]
            py - numpy array with shape [N, 1]

        """

        # ACTUAL PREPROCESSING

        # Notch filtering to remove line frequency noise and its multiplicands, in the desired frequency range (note
        # that frequencies outside of this range will be excluded from the data anyways
        x = notch_filtering(x, self.fs, self.line_freq, self.freq_range)

        # Common Average Reference (car) filter to remove noise common to all channels
        x = car(x)

        # extract desired time range
        t_start = self.time_range[0]
        t_end = self.time_range[1]
        x = x[:, :, t_start:t_end]

        # Calculate spectra from 0 to 200 Hz
        px = get_spectra(x, self.fs, self.freq_range)
        py = y  # nothing to change

        ##

        px = np.asarray(px)
        py = np.asarray(py)

        return px, py

    def run(self):
        train_x = []
        train_y = []
        test_x = []
        test_y = []

        print('reading and preprocessing data...')

        for i, subject in enumerate(self.subject_ids):
            print("\t{}/{}:\tpatient '{}'".format(i + 1, len(self.subject_ids), subject))

            # get train and test data for current subject
            X_train, y_train, X_test, y_test = self.get_train_test_data(subject)

            # preprocess train data
            px, py = self.preprocess_with_label(X_train, y_train)
            train_x.append(px)
            train_y.append(py)

            # preprocess test data
            px, py = self.preprocess_with_label(X_test, y_test)
            test_x.append(px)
            test_y.append(py)

            # Sanity check
            if len(train_x) != len(train_y):
                raise ValueError("Train dataset first dimension mismatch: ", len(train_x), " and ", len(train_y))

            if len(test_x) != len(test_y):
                raise ValueError("Test dataset first dimension mismatch: ", len(train_x), " and ", len(train_y))

        '''if self.config["create_validation_bool"]:
            train_x, train_y, val_x, val_y = self.create_validation_from_train(train_x, train_y)'''

        # Create datasets
        print('saving datasets...')

        if not os.path.exists(self.config["save_dir"]):
            os.makedirs(self.config["save_dir"])

        with h5py.File(os.path.join(self.config["save_dir"], self.config["save_name"]), 'w') as hf:
            grp_train_x = hf.create_group("train_x")
            grp_train_y = hf.create_group("train_y")
            grp_test_x = hf.create_group("test_x")
            grp_test_y = hf.create_group("test_y")

            for i, px in enumerate(train_x):
                grp_train_x[self.subject_ids[i]] = px
            for i, py in enumerate(train_y):
                grp_train_y[self.subject_ids[i]] = py
            for i, px in enumerate(test_x):
                grp_test_x[self.subject_ids[i]] = px
            for i, py in enumerate(test_y):
                grp_test_y[self.subject_ids[i]] = py

            '''if self.config["create_validation_bool"]:
                hf.create_dataset("val_x", data=val_x)
                hf.create_dataset("val_y", data=val_y)'''

        with open(os.path.join(self.config["save_dir"], self.config["default_config_name"]), "w") as fd:
            self.config["time_range"] = str(self.time_range)
            root = {"root": self.config}
            fd.write(xmltodict.unparse(root, pretty=True))

        print("Preprocessing done")


class HtnetPreprocessor(Preprocessor):
    def __init__(self, *args, **kwargs):
        super(HtnetPreprocessor, self).__init__(*args, **kwargs)

        # define path to folder
        my_path = os.path.abspath(os.path.dirname(__file__))
        path_to_data = os.path.join(my_path, "../data/raw_data/naturalistic_move_v_rest/")
        self.lp = path_to_data + 'ecog_dataset/'  # data load path

        self.config["data_source"] = self.lp
        self.config["save_dir"] = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data',
                                               'preprocessed_data', 'naturalistic_move_v_rest')
        self.config["save_name"] = "ecog_preprocessed.hdf5"
        self.config["default_config_name"] = "CONFIG"
        self.config["create_validation_bool"] = False

        self.subject_ids = ['EC01', 'EC02', 'EC03', 'EC04', 'EC05', 'EC06', 'EC07', 'EC08', 'EC09', 'EC10', 'EC11',
                            'EC12']

        self.fs = 250  # sampling rate
        self.line_freq = 60  # line freq = 60 Hz
        self.time_range = (500, -1)  # ignore first half of the measurement (before movement onset)
        self.freq_range = (0, 125)  # range of Power Spectrum, min and max freq in a tuple eg.(0,200) gives
        # power spectrum from 0 to 199 Hz. Default: PSD_freq_range=(0,200). NOTE_: max_freq not included!

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


class StanfordPreprocessor(Preprocessor):
    def __init__(self, test_size=None, *args, **kwargs):
        super(StanfordPreprocessor, self).__init__(*args, **kwargs)

        # define path to folder
        my_path = os.path.abspath(os.path.dirname(__file__))
        path_to_data = os.path.join(my_path, '..', 'data', 'raw_data', 'stanford_motorbasic')
        self.lp = path_to_data  # data load path

        self.config["data_source"] = self.lp
        self.config["save_dir"] = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data',
                                               'preprocessed_data', 'stanford_motorbasic')
        self.config["save_name"] = "ecog_preprocessed.hdf5"
        self.config["default_config_name"] = 'CONFIG'
        self.config["create_validation_bool"] = False

        self.blocksize = 3000  # ignore trials shorter than blocksize
        if test_size is None:
            self.test_size = 0.20  # for train test split, in percentage

        self.fs = 1000  # sampling rate
        self.line_freq = 60  # line freq = 60 Hz
        self.time_range = (0, -1)  # set time range of trials to use in a tuple of (first data, last data),
        # eg (1000,-500) ignores first 1000 and last 500 datapoints. Default: time_range=(0,-1) to use whole range
        self.freq_range = (0, 200)  # range of Power Spectrum, min and max freq in a tuple eg.(0,200) gives
        # power spectrum from 0 to 199 Hz. Default: PSD_freq_range=(0,200). NOTE_: max_freq not included!
        self.subject_ids = ['bp', 'ca', 'cc', 'de', 'fp', 'gc', 'hh', 'hl', 'jc', 'jm', 'jt', 'rh', 'rr', 'ug',
                            'wc', 'zt']

    def get_train_test_data(self, subject):
        """
        Returns train and test data.
        :param subject: id of the patient that identifies them during reading of the data
        :return: X_train, y_train, X_test, y_test. Return shape: [(train_trials, channels), (train_trials, 1),
        (test_trials, channels), (test_trials, 1)], type is ndarray.
        """

        filename = os.path.join(self.lp, subject + '_mot_t_h.mat')

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
            if stim[n] != stim[n - 1] and n - trial_time[-1][0] >= self.blocksize:
                trial_time.append((n, stim[n]))  # save onset time and type of trial if block was longer than blocksize

        # split up data by trials
        data_temp = []
        for i, trial in enumerate(trial_time):  # loop through all trials
            # get current trial
            trial_onset = trial[0]
            trial_type = trial[1]
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


'''
def get_htnet_data_segmented():
    """
    Read method for the stanford motor hand vs tongue vs rest dataset.

    Returns
    ------------
    X_train: (channels, timestep) ndarray
        Train data.
    y_train: ()
        Train labels.
    X_test:
        Test data.
    y_test: ()
        Test labels.
    """
    
    subject_ids = ['EC01']#, 'EC02', 'EC03', 'EC04', 'EC05', 'EC06', 'EC07', 'EC08', 'EC09', 'EC10', 'EC11', 'EC12']
    data_raw = []

    for i, subject in enumerate(subject_ids):
        print(subject)
        # define path to folder
        my_path = os.path.abspath(os.path.dirname(__file__))
        path_to_data = os.path.join(my_path, '..', 'data', 'raw_data', 'naturalistic_move_v_rest')
        lp = os.path.join(path_to_data, 'ecog_dataset')  # data load path
        filename = os.path.join(lp, subject + '_ecog_data.nc')

        sp = os.path.join(my_path, '..', 'data', 'preprocessed_data', 'HTNet_data_segmented.hdf5')

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

        data_raw.append([X_train, y_train, X_test, y_test])

    with h5py.File(sp, 'w') as hf:
        grp_train_x = hf.create_group("train_x")
        grp_train_y = hf.create_group("train_y")
        grp_test_x = hf.create_group("test_x")
        grp_test_y = hf.create_group("test_y")

        for i, data_per_subject in enumerate(data_raw):
            grp_train_x[subject_ids[i]] = data_per_subject[0]
            grp_train_y[subject_ids[i]] = data_per_subject[1]
            grp_test_x[subject_ids[i]] = data_per_subject[2]
            grp_test_y[subject_ids[i]] = data_per_subject[3]

    # return data_raw

def get_stanford_data_segmented():
    '''
