from base import Ecog
from preprocessor import Preprocessor, HtnetPreprocessor, StanfordPreprocessor
from svm import *

import os

#HtnetPreprocessor().run()

#StanfordPreprocessor().run()

ec = Ecog(lp='jj')

sv = SvmClassifier(ec)

sv.subject_ids = ['bp', 'ca', 'cc', 'de', 'fp', 'gc', 'hh', 'hl', 'jc', 'jm', 'jt', 'rh', 'rr', 'ug', 'wc', 'zt']

# load settings
root = os.path.abspath(os.path.dirname(__file__))
lp = os.path.join(root, '..', 'data', 'preprocessed_data', 'HTNet_data_preprocesed.hdf5')
sv.data = h5py.File(lp, 'r')

# save settings
sv.sp = os.path.join(root, '..', 'trained_models', 'HTNet_data')
sv.sp = os.path.join(root, '..', 'trained_models', 'HTNet_data', 'dell')
sv.save_model = False
sv.save_info = True  # if True, results not only displayed in the terminal but also saved.

# feature engineering settings
sv.trial_pairs = {  # set groups to classify between hand v. rest, tongue v. rest, hand v. tongue)
    'descriptions': ['move v rest'],
    'label_pairs': [(1, 2)]
}
sv.label_pairs = [(1, 2)]
sv.ranges = [range(7, 13), range(10, 14), range(14, 26), range(26, 36), range(36, 70), range(76, 125)]
# , range(150,170)] # set freq ranges: low Alpha, high alpha, beta, low gamma, high gamma, kszi
sv.same_scaler = False
# True: fit scaler to train data, scales train and test data with the same scaler.
# False: fit scaler to train data and test data, scales with the corresponding scaler
sv.do_correlation_analysis = False
sv.correlation_pairs = None
sv.multiple_rest = False
sv.corr_threshold = 1

# model settings
sv.scaler = StandardScaler()
sv.clf = svm.SVC()
sv.evaluation = 'cross_val'  # 'simple_split' or 'cross_val'
sv.cv = 5
sv.test_size = 0.3
sv.model_types = ['baseline']  # , 'Nbest'],  # 'baseline', 'Nbest', 'greedy', 'rGreedy'
sv.greedy_max_features = -1
sv.reverse_greedy_min_features = -1
