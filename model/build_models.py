from SVM_model import *
import os
import h5py 
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from tabulate import tabulate

### USER DEFINED PARAMETERS HERE ####

root = os.path.abspath(os.path.dirname(__file__))

# load data
path_to_data = os.path.join(root, "../data/preprocessed_data/HTNet_data_preprocesed.hdf5")
all_data = h5py.File(path_to_data, 'r')

# save dir
#root = Path(".")
database_folder = '../trained_models/htnet_data/'
subfolder = 'rGreedy/'
my_folder = os.path.join(root, database_folder, subfolder)

#clf = svm.NuSVC(decision_function_shape='ovo', class_weight='balanced', verbose=True)
#save model settings
modelSettings={
    'scaler':StandardScaler(),
    'clf':svm.SVC(),
    'evaluation':'cross_val',
    'cv':5,
    'test_size':0.3,
    'greedy_max_features':-1,
    'reverse_greedy_min_features':-1,
    'save_model':False,
    'save_info':True,
    'save_dir':my_folder,
}
featureSettings={
    'trial_pairs':{        # set groups to classify between hand v. rest, tongue v. rest, hand v. tongue)
        'descriptions':['move v rest'],
        'label_pairs':[(1,2)]
    },
    'ranges':[range(7,13), range(10,14), range(14,26), range(26,36), range(36,70), range(76,125)],#, range(150,170)] # set freq ranges: low Alpha, high alpha, beta, low gamma, high gamma, kszi
    'same_scaler':False, # True: fit scaler to train data, scales train and test data with the same scaler. False: fit scaler to train data and test data, scales with the corresponding scaler
    'do_correlation_analysis':False,
    'correlation_pairs':None,
    'multiple_rest':False,
    'corr_treshold':1
}

### build and save models ###

### Call model and run functions that builds models ###
#save settings

if modelSettings['save_info']:
    # create save directory
    if modelSettings['save_dir']==None:
        raise ValueError('save directory not specified.')
    else:
        if not os.path.exists(modelSettings['save_dir']):
            os.makedirs(modelSettings['save_dir'])

    with open(modelSettings['save_dir']+'model_settings.pkl', 'wb') as f:
        pickle.dump({'modelSettings':modelSettings, 'featureSettings':featureSettings}, f)

# get data and labels
train_x = all_data['train_x']
train_y = all_data['train_y']
test_x = all_data['test_x']
test_y = all_data['test_y']

# set model settings as class variables
svmClassifier.set_class_vars(modelSettings)

# calculate classificatioin accuracy for each subject
for i, name in enumerate(list(train_x.keys())):
    #if not name=='EC02':continue
    print('########################################  subject: ', name, '  ################################################\n')
    # get train/test data
    # train
    px_base = np.asarray(train_x[name])
    px_base = np.transpose(px_base, axes=[2,1,0]) # reorder to ???(freq, channels, trials)
    py_base = np.asarray(train_y[name])
    # test
    px_base_test = np.asarray(test_x[name])
    px_base_test = np.transpose(px_base_test, axes=[2,1,0]) # reorder to ???(freq, channels, trials)
    py_base_test = np.asarray(test_y[name])

    # reshape feature vectors
    trial_pairs=featureSettings['trial_pairs']
    # train
    px_dict, py_dict = featurevectors_by_trial(px_base, py_base, trial_types=trial_pairs['label_pairs'], 
                                                do_correlation_analysis=featureSettings['do_correlation_analysis'],
                                                correlation_pairs=featureSettings['correlation_pairs'],
                                                ranges=featureSettings['ranges'],
                                                corr_treshold=featureSettings['corr_treshold'])
    # test
    px_dict_test, py_dict_test = featurevectors_by_trial(px_base_test, py_base_test, trial_types=trial_pairs['label_pairs'], 
                                                do_correlation_analysis=featureSettings['do_correlation_analysis'],
                                                correlation_pairs=featureSettings['correlation_pairs'],
                                                ranges=featureSettings['ranges'],
                                                corr_treshold=featureSettings['corr_treshold'])

    # loop through each featurevector for given patient
    for j, featurevector in enumerate(px_dict):
        # handle exception: when 0 valid channels or freq ranges found based on correlation values
        '''if featurevector.shape[1] == 0:
            for k in range(len(results)): results[k].append(-1)#?????????
            continue'''
        
        # assign class variables
        px = featurevector
        py = py_dict[j]
        px_test = px_dict_test[j]
        py_test = py_dict_test[j]

        #init class
        my_model = svmClassifier(px=px, py=py, id=name, px_test=px_test, py_test=py_test)

        # standardize data
        my_model.standardize_featurevectors(same_scaler=featureSettings['same_scaler'])

        ### BUILD AND SAVE MODELS ###
        
        # SINGLE FEATURE startegy
        #my_model.single_feature() # this list stores accuracy of each feature for the given classification task
        
        # BASELINE startegy
        print('**baseline')
        my_model.baseline()
        '''
        # N-BEST startegy
        my_model.N_best(evaluation)
        # GREEDY startegy
        print('**greedy')
        my_model.greedy()
        
        # reverse GREEDY startegy
        print('**greedy REVERSO')
        my_model.r_greedy()
        '''
# save results
if model['save_info']:
    with open(modelSettings['save_dir']+'accs_all.pkl', 'wb') as f:
        pickle.dump(svmClassifier.results,f)

# print results as table
for name in svmClassifier.results.keys():
    print(name)
    result = svmClassifier.results[name]
    train_row=['train']
    test_row=['test']
    for trial in result.keys():
        if not trial =='greedy':
            train_row.append(round(result[trial][0], 2))
            test_row.append(round(result[trial][1], 2))
        else:
            train_row.append(round(result[trial][0][-1], 2))
            test_row.append(round(result[trial][1][-1], 2))
    table = tabulate([train_row, test_row], headers=[' ', 'single', 'baseline', 'greedy'])
    print(table)

print('accuracies saved')
