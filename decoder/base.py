"""
base.py
-------------

Importing, exporting and doing operations on ECoG data.
"""



import os
import xarray as xr
import numpy as np


class Ecog:

    """
    `Ecog` is the parent class for all preprocessors and models.
    """

    def __init__(self, database_name='', lp=None, sp=None, config=None,
                 subject_ids=None, fs=-1, line_freq=-1, time_range=(0, -1), freq_range=(0, 200)):

        self.lp = lp  # load path of raw data
        self.sp = sp  # save path
        if lp is not None:
            self.database_name = lp.split(os.path.sep)[-1]  # unique name for the database, used during savings and loading
        else:
            self.database_name = database_name

        self.subject_ids = subject_ids

        self.fs = fs  # sampling rate
        self.line_freq = line_freq  # line freq = 60 Hz
        self.time_range = time_range  # set time range of trials to use in a tuple of (first data, last data),
        # eg (1000,-500) ignores first 1000 and last 500 data points. Default: PSD_time_range=(0,-1) to use whole range
        self.freq_range = freq_range  # range of Power Spectrum, min and max freq in a tuple eg.(0,200)
        # gives power spectrum from 0 to 199 Hz. Default: PSD_freq_range=(0,200)

        if config is None:
            self.config = {}
        else:
            self.config = config

        self.raw_data = None
        self.preprocessed_data = None

        self.save_model = False
        self.save_info = True

    def import_raw_data(self):
        raise NotImplementedError

    def export_raw_data(self):
        raise NotImplementedError

    def import_preprocessed_data(self):
        raise NotImplementedError

    def export_preprocessed_data(self):
        raise NotImplementedError