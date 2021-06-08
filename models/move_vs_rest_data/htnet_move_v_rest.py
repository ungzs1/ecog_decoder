from ecog.decoder.htnet_classifier import *
from ecog.decoder.utils import load_data


class HtnetOriginal(HtnetDecoder):
    def __init__(self, *args, **kwargs):
        super(HtnetOriginal, self).__init__(*args, **kwargs)

        my_path = os.path.abspath(os.path.dirname(__file__))

        # Where data will be saved: save_rootpath + dataset
        self.save_rootpath = os.path.join(my_path, '../../..', 'ecog', 'trained_models', 'HTNet', 'htnet_data')

        # Data load paths
        load_rootpath = os.path.join(my_path, '../../..', 'ecog', 'data', 'raw_data', 'naturalistic_move_v_rest')
        self.lp = os.path.join(load_rootpath, 'ecog_dataset')
        # self.ecog_roi_proj_lp = os.path.join(self.lp, 'proj_mat')  #

        # patient ids
        self.pats_ids_in = ['EC01', 'EC02', 'EC03', 'EC04', 'EC05', 'EC06', 'EC07', 'EC08', 'EC09', 'EC10', 'EC11',
                            'EC12']

        # model settings
        self.models = ['eegnet_hilb', 'eegnet']
        self.spec_meas_tail = ['power', 'power_log', 'relative_power', 'phase', 'freqslide']

    def set_tailored_hyperparameters(self):
        """ Tailored decoder parameters (within participant) """
        self.n_folds = 3
        self.hyps = {'F1': 20,
                     'dropoutRate': 0.693,
                     'kernLength': 64,
                     'kernLength_sep': 56,
                     'dropoutType': 'SpatialDropout2D',
                     'D': 2,
                     'n_estimators': 240,
                     'max_depth': 9}
        self.hyps['F2'] = self.hyps['F1'] * self.hyps['D']  # F2 = F1 * D
        self.epochs = 30
        self.patience = 30

    def load_data(self, patient, randomize_events=True):
        """
        Database specific load method for given patient. Load data from file.
        :param patient: current patient id.
        :param randomize_events: set 'True' to randomize events, set 'False' to keep event order. Default is 'True'.
        :return: X: train data in the form of (trial, channel, timestep)
        :return: y: train labels in the form of (trial, )
        :return: X_test: test data in the form of (trial, channel, timestep)
        :return: y_test: test labels in the form of (trial, )
        """

        tlim = [-1, 1]
        test_day = 'last'

        X, y, X_test, y_test, sbj_order, sbj_order_test = load_data(patient, self.lp,
                                                                    n_chans_all=self.n_chans_all,
                                                                    test_day=test_day, tlim=tlim)
        X[np.isnan(X)] = 0  # set all NaN's to 0

        # Randomize event order (random seed facilitates consistency)
        if randomize_events:
            order_inds = np.arange(len(y))
            np.random.shuffle(order_inds)
            X = X[order_inds, ...]
            y = y[order_inds]

            order_inds_test = np.arange(len(y_test))
            np.random.shuffle(order_inds_test)
            X_test = X_test[order_inds_test, ...]
            y_test = y_test[order_inds_test]

        return X, y, X_test, y_test


if __name__ == '__main__':
    HtnetOriginal().train_tailored_decoder()
