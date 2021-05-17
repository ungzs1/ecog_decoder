from svm_base import *

import os
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn import svm


class SvmHTdata(SvmClassifier):
    def __init__(self, *args, **kwargs):
        super(SvmHTdata, self).__init__(*args, **kwargs)

        # load settings
        root = os.path.abspath(os.path.dirname(__file__))
        lp = os.path.join(root, '..', 'data', 'preprocessed_data', 'HTNet_data_preprocesed.hdf5')
        self.data = h5py.File(lp, 'r')

        # save settings
        # self.sp = os.path.join(root, '..', 'trained_models', 'HTNet_data')
        self.sp = os.path.join(root, '..', 'trained_models', 'HTNet_data', 'dell')
        self.save_model = False
        self.save_info = True  # if True, results not only displayed in the terminal but also saved.

        # feature engineering settings
        self.trial_pairs = {  # set groups to classify between hand v. rest, tongue v. rest, hand v. tongue)
            'descriptions': ['move v rest'],
            'label_pairs': [(1, 2)]
        }
        self.label_pairs = [(1, 2)]
        self.ranges = [range(7, 13), range(10, 14), range(14, 26), range(26, 36), range(36, 70), range(76, 125)]
        # , range(150,170)] # set freq ranges: low Alpha, high alpha, beta, low gamma, high gamma, kszi
        self.same_scaler = False
        # True: fit scaler to train data, scales train and test data with the same scaler.
        # False: fit scaler to train data and test data, scales with the corresponding scaler
        self.do_correlation_analysis = False
        self.correlation_pairs = None
        self.multiple_rest = False
        self.corr_threshold = 1

        # model settings
        self.scaler = StandardScaler()
        self.clf = svm.SVC()
        self.evaluation = 'cross_val'  # 'simple_split' or 'cross_val'
        self.cv = 5
        self.test_size = 0.3
        self.model_types = ['baseline']  # , 'Nbest'],  # 'baseline', 'Nbest', 'greedy', 'rGreedy'
        self.greedy_max_features = -1
        self.reverse_greedy_min_features = -1

    def save_params(self):
        # save model settings
        if self.save_info:
            # create save directory
            if self.sp is None:
                raise ValueError('save directory not specified.')
            else:
                if not os.path.exists(self.sp):
                    os.makedirs(self.sp)

            # Save pickle file with dictionary of input parameters
            params_dict = {key: value for key, value in SvmHTdata.__dict__.items() if  # todo ez nem mukodik megfeleloen
                           not key.startswith('__') and not callable(key)}

            # save as .pkl
            with open(os.path.join(self.sp, 'param_file.pkl'), 'wb') as f:
                pickle.dump(params_dict, f)

            # save as .txt
            with open(os.path.join(self.sp, 'param_file.txt'), 'w') as f:
                f.write(str(params_dict))


if __name__ == '__main__':
    SvmHTdata().run()
