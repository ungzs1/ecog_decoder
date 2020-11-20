import os
import sys

import cv2
import glob
import numpy as np
import xmltodict

from scipy.signal import butter, lfilter
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

    def __init__(self, _data_dir, config_file ="", config={}):

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
        self.config.setdefault("save_name","") # 
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
            x,y = self.load_data_and_labels(ftrain)
            px,py = self.preprocess_with_label(x,y)
            train_x.append(px)
            train_y.append(py)

        for ftest in self.test_files_from_dir():
            x,y = self.load_data_and_labels(ftest)
            px,py = self.preprocess_with_label(x,y)
            test_x.append(px)
            test_y.append(py)

        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        test_x = np.asarray(test_x)
        test_y = np.asarray(test_y)

        # Sanity check

        if train_x.shape[0] != train_y.shape[0]:
            raise ValueError("Train dataset first dimension mismatch: "+str(train_x.shape[0])+" and "+str(train_y.shape[0]))

        if test_x.shape[0] != test_y.shape[0]:
            raise ValueError("Test dataset first dimension mismatch: "+str(test_x.shape[0])+" and "+str(test_y.shape[0]))

        if self.config["create_validation_bool"]:
            train_x, train_y, val_x, val_y = self.create_validation_from_train(train_x, train_y)

        with h5py.File(os.path.join(self.save_dir, self.save_name_base), 'w') as hf:
            hf.create_dataset("train_x",  data=train_x)
            hf.create_dataset("train_y",  data=train_y)
            hf.create_dataset("test_x",  data=test_x)
            hf.create_dataset("test_y",  data=test_y)

            if self.config["create_validation_bool"]:
                hf.create_dataset("val_x",  data=val_x)
                hf.create_dataset("val_y",  data=val_y)

        with open(os.path.join(self.config["save_dir"], self.config["default_config_name"]), "w") as fd:
            root = {"root":self.config}
            fd.write(xmltodict.unparse(root, pretty = True))

    def preprocess_with_label(self, x,y):

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


        ## 


        # Actual preprocessing 


        ## 

        px = np.asarray(px)
        py = np.asarray(py)

        return px,py


