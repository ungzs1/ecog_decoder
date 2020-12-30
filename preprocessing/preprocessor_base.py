import os
import sys

import glob
import numpy as np
import xmltodict
import h5py

from scipy.signal import butter, lfilter, iirnotch, filtfilt, welch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

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

def notch_filtering(data, fs, line_freq, PSD_range):
    #removes line frequency noise and multiples from the signal

    multiples_linefreq = range(line_freq,PSD_range[1],line_freq) # line frequency and multiples eg. [60,120,180]

    for f0 in multiples_linefreq:
        w0 = f0 / (fs / 2 ) # Normalized Frequency
        # filtering
        b, a = iirnotch(w0, fs)  
        data = filtfilt(b, a, data, axis=0)
        
    return data

def car(data):
    #%this function calculates and returns the common avg reference of 2-d matrix "data" of shape [timestep, channel].
    data = np.double(data)        
    num_chans = data.shape[1]
    
    # create a CAR spatial filter
    spatfiltmatrix = -np.ones((num_chans,num_chans))
    for i in range(num_chans):
        spatfiltmatrix[i, i] = num_chans-1
    spatfiltmatrix = spatfiltmatrix/num_chans

    # perform spatial filtering
    data = np.dot(data, spatfiltmatrix)

    return data

def get_spectra(x, tr_tm, fs, time_range, freq_range):
    # returns PSD as "all_PSD" in form of [frequencies, channels, trials]. power spectrum densitiy for each channel for each trial
    num_chans = x.shape[1]
    num_trials = tr_tm.shape[1]

    is_firstloop = True

    #calculate PSD
    for cur_trial in range(num_trials): # loop through all trials
        # get actual trial
        if cur_trial == num_trials-1:
            curr_data = np.squeeze(x[tr_tm[0,cur_trial]:,:])
        else:
            curr_data = np.squeeze(x[tr_tm[0,cur_trial]:tr_tm[0,cur_trial+1],:])
        
        # ignore data outside the range of time_freq (transition between movement types)
        curr_data = curr_data[time_range[0]:time_range[1],:] 
        
        # set window size and offset for PSD
        noverlap = np.floor(fs*0.1); nperseg = np.floor(fs*0.25) 
        
        # get Power Spectrum Density with signal.welch
        for p in range(num_chans):
            [f, temp_PSD] = welch(curr_data[:,p], nfft=fs, fs=fs, noverlap=noverlap, nperseg=nperseg)
            temp_PSD = temp_PSD.reshape((-1,1))
            if p == 0:
                block_PSD = temp_PSD
            else:
                block_PSD = np.hstack((block_PSD, temp_PSD))
        block_PSD = block_PSD[freq_range[0]:freq_range[1],:] # downsample - we only want to get spectra of PSD_range
        
        if is_firstloop:
            all_PSD = block_PSD
            is_firstloop = False
        else:
            all_PSD = np.dstack((all_PSD, block_PSD))

    return(all_PSD)

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

class Preprocessor(object):

    fs = -1 # sampling rate
    line_freq = -1 # line freq = 60 Hz
    blocksize = np.floor(fs*0.25) # minimum block size of trials, shorter trials ignored. Default: blocksize=floor(fs*0.25) window length of PSD
    PSD_time_range = (0,-1) # set time range of trials to use in a tuple of (first data, last data), eg (1000,-500) ignores first 1000 and last 500 datapoints. Default: PSD_time_range=(0,-1) to use whole range
    PSD_freq_range = (0,200) # range of Power Spectrum, min and max freq in a tuple eg.(0,200) gives power spectrum from 0 to 199 Hz. Default: PSD_freq_range=(0,200)

    def __init__(self, config_file ="", config={}):

        if config_file != "" and path.exists(config_file):
            with open(config_file) as fd:
                try:
                    self.config = xmltodict.parse(fd.read(), dict_constructor=dict, postprocessor=config_post)
                    self.config = self.config["root"]
                    self.config.update(config)

                except Exception:
                    self.config = config
        else:
            self.config = config

        self.config.setdefault("save_dir", "")
        self.config.setdefault("save_name","") 
        self.config.setdefault("default_config_name", "config.xml")
        self.config.setdefault("create_validation_bool", True)
        self.config.setdefault("data_source", "")

    def load_data_and_labels(self, filename):
        # should return a pair of numpy arrays of dimensions ( [timestep, channels], [timestep, label] )
        raise NotImplementedError

    def train_files_from_dir(self):
        # return all the valid train files in a list
        raise NotImplementedError
    
    def test_files_from_dir(self):
        # return all the valid test files in a list
        raise NotImplementedError

    def run(self):
        train_x = []
        train_y = []
        test_x = []
        test_y = []

        for ftrain in self.train_files_from_dir():
            print(ftrain) # to see progress
            x,y = self.load_data_and_labels(ftrain)
            px,py = self.preprocess_with_label(x,y) 
            train_x.append(px)
            train_y.append(py)

        for ftest in self.test_files_from_dir():
            x,y = self.load_data_and_labels(ftest)
            px,py = self.preprocess_with_label(x,y)
            test_x.append(px)
            test_y.append(py)

        # Sanity check
        
        if len(train_x) != len(train_y):
            raise ValueError("Train dataset first dimension mismatch: ", len(train_x), " and ", len(train_y))

        if len(test_x) != len(test_y):
            raise ValueError("Test dataset first dimension mismatch: ", len(train_x), " and ", len(train_y))
        
        if self.config["create_validation_bool"]:
            train_x, train_y, val_x, val_y = self.create_validation_from_train(train_x, train_y)
        
        # Create datasets

        with h5py.File(os.path.join(self.config["save_dir"], self.config["save_name"]), 'w') as hf:
            grp_train_x = hf.create_group("train_x")
            grp_train_y = hf.create_group("train_y")
            #grp_test_x = hf.create_group("test_x")
            #grp_test_y = hf.create_group("test_y")

            for i, px in enumerate(train_x): grp_train_x[self.subject_ids[i]] = px
            for i, py in enumerate(train_y): grp_train_y[self.subject_ids[i]] = py
            #for i, px in enumerate(train_x): grp_test_x[self.subject_ids[i]] = px
            #for i, py in enumerate(train_y): grp_test_y[self.subject_ids[i]] = py

            if self.config["create_validation_bool"]:
                hf.create_dataset("val_x",  data=val_x)
                hf.create_dataset("val_y",  data=val_y)

        with open(os.path.join(self.config["save_dir"], self.config["default_config_name"]), "w") as fd:
            root = {"root":self.config}
            fd.write(xmltodict.unparse(root, pretty = True))

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

        # catalogue trials in "tr_tm": [trial onset, trial type]
        tr_tm = [[0, y[0]]] # initialize with [time=0, trial=y[0]]
        for n in range(1,len(y)):
            if y[n] != y[n-1] and n-tr_tm[-1][0] >= self.blocksize: # save onset and type of trial only if trial>=blocksize
                tr_tm.append([n, y[n]])

        tr_tm = np.transpose(np.asarray(tr_tm))
        py = tr_tm[1]

        # Notch filtering to remove line frequency noise and multiples in PSD_range
        x = notch_filtering(x, self.fs, self.line_freq, self.PSD_freq_range)

        # Common Average Reference (car) filter to remove noise common to all channels
        x = car(x)

        # Calculate spectra from 0 to 200 Hz
        px = get_spectra(x, tr_tm, self.fs, self.PSD_time_range, self.PSD_freq_range)

        ## 

        px = np.asarray(px)
        py = np.asarray(py)

        return px,py


