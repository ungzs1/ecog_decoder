from SVM_model import *
import os
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn import svm

### USER DEFINED PARAMETERS HERE ####

# save dir
root = os.path.abspath(os.path.dirname(__file__))
database_folder = '../trained_models/htnet_data/'
subfolder = 'plot_N_best/'
my_folder = os.path.join(root, database_folder, subfolder)

# clf = svm.NuSVC(decision_function_shape='ovo', class_weight='balanced', verbose=True)
# save model settings
modelSettings = {
    'scaler': StandardScaler(),
    'clf': svm.SVC(),
    'evaluation': 'cross_val',
    'cv': 5,
    'test_size': 0.3,
    'model_types': ['baseline', 'Nbest'],  # , 'greedy']
    'greedy_max_features': -1,
    'reverse_greedy_min_features': -1,
    'save_model': False,
    'save_info': True,
    'save_dir': my_folder
}
featureSettings = {
    'trial_pairs': {  # set groups to classify between hand v. rest, tongue v. rest, hand v. tongue)
        'descriptions': ['move v rest'],
        'label_pairs': [(1, 2)]
    },
    'ranges': [range(7, 13), range(10, 14), range(14, 26), range(26, 36), range(36, 70), range(76, 125)],
    # , range(150,170)] # set freq ranges: low Alpha, high alpha, beta, low gamma, high gamma, kszi
    'same_scaler': False,
    # True: fit scaler to train data, scales train and test data with the same scaler.
    # False: fit scaler to train data and test data, scales with the corresponding scaler
    'do_correlation_analysis': False,
    'correlation_pairs': None,
    'multiple_rest': False,
    'corr_threshold': 1
}

### build and save models ###

## Call model and run functions that builds models ###
# save settings

# load data
path_to_data = os.path.join(root, "../data/preprocessed_data/HTNet_data_preprocesed.hdf5")
all_data = h5py.File(path_to_data, 'r')

run(all_data, modelSettings, featureSettings)

'''
C = [5, 10, 50]
gamma = [0.01, 0.001, 0.0001]
c = 10
g = 0.001
for c in C:
    for g in gamma:
        print(c, g)
        # modelSettings['clf'] = svm.NuSVC(decision_function_shape='ovo', class_weight='balanced', verbose=True)

        modelSettings['clf'] = svm.SVC(C=c, gamma=g)
        # modelSettings['save_dir'] = 'c_'+str(c)+'_g_'+str(g)
        # run models
        run(all_data, modelSettings, featureSettings)'''
