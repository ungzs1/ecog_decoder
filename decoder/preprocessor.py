import os

import numpy as np
import xmltodict
import h5py
import json

from scipy.signal import butter, lfilter, iirnotch, filtfilt, welch


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


def get_spectra(x, fs, freq_range, nperseg, noverlap):
    """
    returns PSD as "all_psd" in form of [frequencies, channels, trials]. power spectrum densitiy for each channel
    for each trial
    """

    all_psd = []  # PSD for all trials and channels

    for curr_data in x:  # loop through all trials
        block_psd = []  # PSD for features in current 'block' ie all channels for this trial
        # get Power Spectrum Density with signal.welch
        for ch in range(x.shape[1]):  # loop through all channels
            [f, temp_psd] = welch(curr_data[ch, :], fs=fs, noverlap=noverlap, nperseg=nperseg)
            temp_psd = temp_psd[
                np.where((freq_range[0] <= f) & (f <= freq_range[1]))]  # to get the desired freq range only
            block_psd.append(temp_psd)

        block_psd = np.asarray(block_psd)
        all_psd.append(block_psd)

    all_psd = np.asarray(all_psd)

    return all_psd


''''def config_post(path, key, value):
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

    return key, value'''


class Preprocessor:
    def __init__(self, time_range=(0, -1), freq_range=(0, 200), fs=-1, line_freq=-1, subject_ids=None):

        self.save_dir = ""
        self.save_name = ""
        self.default_config_name = "config.xml"
        self.create_validation_bool = True
        self.data_source = ""

        self.time_range = time_range  # set time range of trials to use in a tuple of (first data, last data),
        # eg (1000,-500) ignores first 1000 and last 500 data points. Default: PSD_time_range=(0,-1) to use whole range
        self.freq_range = freq_range  # range of Power Spectrum, min and max freq in a tuple eg.(0,200)
        self.fs = fs  # sampling rate
        self.line_freq = line_freq  # line freq = 60 Hz

        self.subject_ids = subject_ids  # store subject ids to ease their access

    def get_train_test_data(self, subject):
        raise NotImplementedError

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
                raise ValueError("Test dataset first dimension mismatch: ", len(test_x), " and ", len(test_y))

        '''if self.config["create_validation_bool"]:
            train_x, train_y, val_x, val_y = self.create_validation_from_train(train_x, train_y)'''

        # *** Create datasets ***
        print('saving datasets...')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # save results
        with h5py.File(os.path.join(self.save_dir, self.save_name), 'w') as hf:
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

        # save config info
        with open(os.path.join(self.save_dir, self.default_config_name), "w") as fd:
            # save config settings
            config = self.__dict__
            json.dump(config, fd)
            '''self.config["time_range"] = str(self.time_range)
            root = {"root": self.config}
            fd.write(xmltodict.unparse(root, pretty=True))'''

        print("Preprocessing done")

    def preprocess_with_label(self, x, y):

        """

        Parameters:
            x - numpy array with shape [timestep, ... ]
            y - numpy array with shape [timestep, ... ]

        Return:
            px - numpy array with shape [N, ...]
            py - numpy array with shape [N, 1]

        """

        px = []
        py = []

        ## ACTUAL PREPROCESSING ##

        # Notch filtering to remove line frequency noise and its multiplicands, in the desired frequency range (note
        # that frequencies outside of this range will be excluded from the data anyways
        x = notch_filtering(x, self.fs, self.line_freq, self.freq_range)

        # Common Average Reference (car) filter to remove noise common to all channels
        x = car(x)

        # extract desired time range
        t_start = self.time_range[0]
        t_end = self.time_range[1]
        x = x[:, :, t_start:t_end]

        # Calculate power spectral density
        px = get_spectra(x, fs=self.fs, freq_range=self.freq_range, nperseg=self.nperseg, noverlap=self.noverlap)
        py = y  # nothing to change

        px = np.asarray(px)
        py = np.asarray(py)

        return px, py
