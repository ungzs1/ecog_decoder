from ecog.decoder.svm_classifier import *

class SvmHtnetData(SvmDecoder):
    def __init__(self):
        super(SvmHtnetData, self).__init__()

        # *** USER PARAMETERS ***
        # i/o settings
        root = os.path.abspath(os.path.dirname(__file__))
        lp = os.path.join(root, '..', '..', 'data', 'preprocessed_data', 'HTNet_data_preprocesed.hdf5')
        self.preprocessed_data = h5py.File(lp, 'r')
        self.subject_ids = ['EC01', 'EC02', 'EC03', 'EC04', 'EC05', 'EC06', 'EC07', 'EC08', 'EC09', 'EC10',
                            'EC11', 'EC12']
        self.sp = os.path.join(root, '..', '..', 'trained_models', 'svm', 'HTNet_data', 'del')
        self.save_model = False
        self.save_info = True  # if True, results not only displayed in the terminal but also saved.

        # feature engineering settings
        self.trial_pairs = {  # set groups to classify between hand v. rest, tongue v. rest, hand v. tongue)
            'descriptions': ['move v rest'],
            'label_pairs': [(1, 2)]
        }
        #self.label_pairs = label_pairs
        self.ranges = [range(7, 13), range(10, 14), range(14, 26), range(26, 36), range(36, 70), range(76, 125)]
        # , range(150,170)] # set freq ranges: low Alpha, high alpha, beta, low gamma, high gamma, kszi
        self.same_scaler = True
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
        self.model_types = ['baseline']  #, 'greedy']  # 'baseline', 'Nbest', 'greedy', 'rGreedy'
        self.greedy_max_features = -1
        self.reverse_greedy_min_features = -1


if __name__ == '__main__':
    SvmHtnetData().run()
