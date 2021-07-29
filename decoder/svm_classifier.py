try:
    from .feature_transformation import *
except ImportError:
    from feature_transformation import *

import h5py
import os
import numpy as np

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import pickle
import json
from tabulate import tabulate
import matplotlib.pyplot as plt


def progress_bar(message, current, total, bar_length=20):
    percent = float(current) * 100 / (total - 1)
    arrow = '#' * int(percent / 100 * bar_length - 1) + '#'
    spaces = '.' * (bar_length - len(arrow))
    if not current == total - 1:
        print(message, ': [%s%s] %d %%' % (arrow, spaces, percent), end='\r')
    else:
        print(message, ': [%s%s] %d %%' % (arrow, spaces, percent))


"""
def run(all_data, modelSettings, featureSettings):
    # save model settings
    if modelSettings['save_info']:
        # create save directory
        if modelSettings['save_dir'] is None:
            raise ValueError('save directory not specified.')
        else:
            if not os.path.exists(modelSettings['save_dir']):
                os.makedirs(modelSettings['save_dir'])

        # save as .pkl
        with open(os.path.join(modelSettings['save_dir'], 'model_settings.pkl'), 'wb') as f:
            pickle.dump({'modelSettings': modelSettings, 'featureSettings': featureSettings}, f)

        # save as .txt
        with open(os.path.join(modelSettings['save_dir'], 'model_settings.txt'), 'w') as f:
            f.write(str({'modelSettings': modelSettings, 'featureSettings': featureSettings}))

    # get data and labels
    train_x = all_data['train_x']
    train_y = all_data['train_y']
    test_x = all_data['test_x']
    test_y = all_data['test_y']

    # set model settings as class variables
    SvmClassifier.set_class_vars(modelSettings)

    # calculate classification accuracy for each subject
    for i, name in enumerate(list(train_x.keys())):
        if not name == 'EC02': continue
        print('####  subject: ', name, '####  \n')
        # get train/test data
        # train
        px_base = np.asarray(train_x[name])
        px_base = np.transpose(px_base, axes=[2, 1, 0])  # reorder to ???(freq, channels, trials)
        py_base = np.asarray(train_y[name])
        # test
        px_base_test = np.asarray(test_x[name])
        px_base_test = np.transpose(px_base_test, axes=[2, 1, 0])  # reorder to ???(freq, channels, trials)
        py_base_test = np.asarray(test_y[name])

        # reshape feature vectors
        trial_pairs = featureSettings['trial_pairs']
        # train
        px_dict, py_dict = featurevectors_by_trial(px_base, py_base, trial_types=trial_pairs['label_pairs'],
                                                   do_correlation_analysis=featureSettings['do_correlation_analysis'],
                                                   correlation_pairs=featureSettings['correlation_pairs'],
                                                   ranges=featureSettings['ranges'],
                                                   corr_threshold=featureSettings['corr_threshold'])
        # test
        px_dict_test, py_dict_test = featurevectors_by_trial(px_base_test, py_base_test,
                                                             trial_types=trial_pairs['label_pairs'],
                                                             do_correlation_analysis=featureSettings[
                                                                 'do_correlation_analysis'],
                                                             correlation_pairs=featureSettings['correlation_pairs'],
                                                             ranges=featureSettings['ranges'],
                                                             corr_threshold=featureSettings['corr_threshold'])

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

            # init class
            my_model = SvmClassifier(px=px, py=py, patient_id=name, px_test=px_test, py_test=py_test)

            # standardize data
            my_model.standardize_feature_vectors(same_scaler=featureSettings['same_scaler'])

            # BUILD AND SAVE MODELS ###
            for model_type in modelSettings['model_types']:
                # SINGLE FEATURE strategy
                # my_model.single_feature()# this list stores accuracy of each feature for the given classification task
                if model_type == 'baseline':
                    # BASELINE strategy
                    print('**baseline')
                    my_model.baseline()
                elif model_type == 'Nbest':
                    # N-BEST strategy
                    print('**N best')
                    my_model.n_best(plot=True)
                elif model_type == 'greedy':
                    # GREEDY strategy
                    print('**greedy')
                    my_model.greedy()
                elif model_type == 'rGreedy':
                    # reverse GREEDY strategy
                    print('**greedy REVERSO')
                    my_model.r_greedy()

    # print results as table
    results = SvmClassifier.results
    tables_all, table2 = SvmClassifier.print_results(results)

    # save results
    if modelSettings['save_info']:
        # save as .pkl
        with open(os.path.join(modelSettings['save_dir'], 'accs_all.pkl'), 'wb') as f:
            pickle.dump(SvmClassifier.results, f)

        # save as .txt
        with open(os.path.join(modelSettings['save_dir'], 'accs_all.txt'), 'w') as f:
            f.write(str(SvmClassifier.results))

        # save summary as .txt
        with open(os.path.join(modelSettings['save_dir'], 'accs_summary.txt'), 'w') as f:
            for result in tables_all:
                f.write(result['name'] + '\n')
                f.write(result['table'] + '\n\n')
            f.write('mean\n')
            f.write(table2)

    return results
"""


class SvmDecoder:
    """
    docstring
    """
    # SVM parameters
    '''scaler = StandardScaler()
    clf = svm.SVC()
    evaluation = 'cross_val'
    save_model = False
    save_dir = None
    cv = 5
    test_size = 0.2
    greedy_max_features = -1
    r_greedy_min_features = -1'''

    # to collect results
    results = {}

    def __init__(self, preprocessed_data=None, subject_ids=None, sp=None, save_model=None, save_info=None,
                 trial_pairs=None, ranges=None, same_scaler=False, do_correlation_analysis=False,
                 correlation_pairs=None, multiple_rest=False, corr_threshold=None,
                 scaler=None, clf=None, evaluation=None, cv=None, test_size=None, model_types=None,
                 greedy_max_features=None, reverse_greedy_min_features=None, *args, **kwargs):  # todo default erteketk beallitasa ahol kell

        # parameters defined by the script
        self.px = None
        self.py = None
        self.px_test = None
        self.py_test = None
        self.patient_id = None

        # *** USER PARAMETERS ***
        # i/o settings
        self.root = os.path.abspath(os.path.dirname(__file__))
        self.preprocessed_data = preprocessed_data
        if subject_ids is not None:
            self.subject_ids = subject_ids
        else:
            self.subject_ids = []
        self.sp = sp
        self.save_model = save_model
        self.save_info = save_info  # if True, results not only displayed in the terminal but also saved.

        # feature engineering settings
        self.trial_pairs = trial_pairs  # set groups to classify between hand v. rest, tongue v. rest, hand v. tongue)
        self.ranges = ranges
        # , range(150,170)] # set freq ranges: low Alpha, high alpha, beta, low gamma, high gamma, kszi
        self.same_scaler = same_scaler
        # True: fit scaler to train data, scales train and test data with the same scaler.
        # False: fit scaler to train data and test data, scales with the corresponding scaler
        self.do_correlation_analysis = do_correlation_analysis
        self.correlation_pairs = correlation_pairs
        self.multiple_rest = multiple_rest
        self.corr_threshold = corr_threshold

        # model settings
        self.scaler = scaler
        self.clf = clf
        self.evaluation = evaluation  # 'simple_split' or 'cross_val'
        self.cv = cv
        self.test_size = test_size
        self.model_types = model_types
        self.greedy_max_features = greedy_max_features
        self.reverse_greedy_min_features = reverse_greedy_min_features

        # save config settings
        config = self.__dict__
        del config['preprocessed_data']
        self.config = config

    def standardize_feature_vectors(self, same_scaler=True):
        # reshape(flatten) training data
        self.px = np.asarray(self.px, dtype='double')  # "Avoiding data copy: For SVC, SVR, NuSVC and NuSVR, if the data
        # passed to certain methods is not C-ordered contiguous and double precision, it will be copied before calling
        # the underlying C implementation."
        px_temp = np.transpose(self.px.reshape(-1, self.px.shape[-1]))

        # scale train data
        scaler_train = self.scaler.fit(px_temp)  # fit scaler to train data
        self.px = np.transpose(scaler_train.transform(px_temp)).reshape(self.px.shape)  # scale data with fitted scaler

        # scale test data (with the same scaler!!)
        if not self.px_test == []:
            self.px_test = np.asarray(self.px_test, dtype='double')
            px_test_temp = np.transpose(self.px_test.reshape(-1, self.px_test.shape[-1]))
            if same_scaler:  # scale test data with the same scaler as train data
                self.px_test = np.transpose(scaler_train.transform(px_test_temp)).reshape(self.px_test.shape)
            else:  # scale data with scaler fitted to it
                scaler_test = self.scaler.fit(px_test_temp)
                self.px_test = np.transpose(scaler_test.transform(px_test_temp)).reshape(self.px_test.shape)
        else:
            print('test set not scaled because not defined')

    def score_model(self, px, py):
        # build model
        if self.evaluation == 'cross_val':
            scores = cross_val_score(self.clf, px, py, cv=self.cv)
            res = np.mean(scores)
            # print(scores.mean(),scores.std())
            return res
        elif self.evaluation == 'simple_split':
            X_train, X_test, y_train, y_test = train_test_split(px, py, test_size=self.test_size, shuffle=False)
            clf = self.clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            res = score
            return res
        else:
            import warnings
            warnings.warn("invalid evaluation mode '" + self.evaluation + "'")

    '''def single_feature(self):    ### SINGLE FEATURE IS GREEDY WITH SELECTED FEATURES SET TO 1!!!
        print('**single feature')
        res = 0 # stores accuracy values for each channel and each frequency range

        num_ch = self.px.shape[1]
        num_ranges = self.px.shape[0]

        ## build model
        for ch in range(num_ch):
            for freq_range in range(num_ranges):
                px_temp = self.px[freq_range,ch,:].reshape(-1,1)

                if self.evaluation == 'cross_val':
                    scores = cross_val_score(self.clf, px_temp, self.py, cv=self.cv)
                    res_temp = scores.mean()#, scores.std()*2)
                elif self.evaluation == 'simple_split':
                    X_train, X_test, y_train, y_test = train_test_split(px_temp, self.py,
                    test_size=self.test_size, shuffle=False)
                    clf = self.clf.fit(X_train, y_train)
                    score = clf.score(X_test, y_test)
                    res_temp = score

                if res_temp>res: 
                    res=res_temp

        ## save result
        if self.id not in svmClassifier.results:
            svmClassifier.results[self.id] = {'single':[],'baseline':[],'n_best':[], 'greedy':[]}
        svmClassifier.results[self.id]['single'].append(res)
        print('\taccuracy: ', res)'''

    def evaluate_single_features(self):
        """
        Evaluates single features. Saves results to self.results.
        :return:
        """

        num_ch = self.px.shape[1]
        num_ranges = self.px.shape[0]
        freq_range = np.arange(num_ranges)  # to use all ranges, i.e to only select features

        # these two variables store selected feature set and accuracy in each iteration
        best_features_in_trial = {'result': [], 'channel': [], 'freq_range': []}
        best_features_in_trial_test = {'result': [], 'channel': [], 'freq_range': []}

        # loop through all frequency bands
        for freq in [freq_range]:  # TODO!!! minden freq range egybe
            # loop through all channels bands
            for ch in range(num_ch):
                # fit px (featurevector) to required model input shape
                px_temp = np.transpose(self.px[freq, ch, :])
                px_temp_test = np.transpose(self.px_test[freq, ch, :])

                # build and evaluate model, returns accuracy result
                res = self.score_model(px_temp, self.py)

                # build model on (final) px_temp
                model = self.clf.fit(px_temp, self.py)

                # evaluate model on test set
                res_test = model.score(px_temp_test, self.py_test)

                best_features_in_trial['result'].append(res)
                best_features_in_trial['channel'].append(ch)
                best_features_in_trial['freq_range'].append(freq)

                best_features_in_trial_test['result'].append(res_test)
                best_features_in_trial_test['channel'].append(ch)
                best_features_in_trial_test['freq_range'].append(freq)


        # add result to global class variable
        if self.id not in SvmDecoder.results:
            SvmDecoder.results[self.id] = {}

        SvmDecoder.results[self.id]['all_features'] = {'train': best_features_in_trial,
                                                       'test': best_features_in_trial_test}

        # print results
        print('All features are evaluated. channels: {}, frequency bands: {}'.format(num_ch, num_ranges))

    def baseline(self):
        # fit px (featurevector) to required model input shape
        px_temp = np.transpose(self.px.reshape(-1, self.py.shape[0]))  # train
        px_temp_test = np.transpose(self.px_test.reshape(-1, self.py_test.shape[0]))  # test

        # build and evaluate model, returns accuracy result
        res = self.score_model(px_temp, self.py)

        # build model on (final) px_temp
        model = self.clf.fit(px_temp, self.py)

        # evaluate model on test set
        res_test = model.score(px_temp_test, self.py_test)

        # add result to global class variable
        if self.id not in SvmDecoder.results:
            SvmDecoder.results[self.id] = {}

        SvmDecoder.results[self.id]['baseline'] = [res, res_test]

        # save the model to disk
        if self.save_model:
            filename = self.id + '_baseline_SVM.sav'
            my_path = os.path.join(self.save_dir, filename)

            with open(my_path, 'wb') as f:
                pickle.dump(model, f)

        # print results
        print('\taccuracy: ', round(res, 2), 'test accuracy: ', round(res_test, 2))

    def n_best(self, plot=False):
        # init some variables
        num_ch = self.px.shape[1]
        num_ranges = self.px.shape[0]

        feature_dict = []

        # get accuracy for each channel, store in a dict of {rank, acc, ch, freq, feature}
        for ch in range(num_ch):
            # print progress bar
            progress_bar('calculate accuracy', ch, num_ch)

            for freq_range in range(num_ranges):
                # get current feature
                px_temp = np.transpose(self.px[freq_range, ch, :]).reshape(-1, 1)
                px_temp_test = np.transpose(self.px_test[freq_range, ch, :]).reshape(-1, 1)

                # evaluate feature
                res_temp = self.score_model(px_temp, self.py)

                # test accuracy
                model = self.clf.fit(px_temp, self.py)
                res_temp_test = model.score(px_temp_test, self.py_test)

                # locally save result to dict
                feature_dict.append({'accuracy': res_temp, 'test_accuracy': res_temp_test,
                                     'channel': ch, 'freq_range': freq_range,
                                     'px_temp': px_temp, 'px_temp_test': px_temp_test})

        # build model from best 1,2,3...N channels and add rank to list
        for i, feature in enumerate(sorted(feature_dict, key=lambda j: j['accuracy'], reverse=True)):
            progress_bar('creating models', i, num_ch * num_ranges)
            feature['rank'] = i  # add rank key to dict for further usage

            # concatenate  first N features
            if i == 0:
                px_temp = feature['px_temp']
                px_temp_test = feature['px_temp_test']
            else:
                px_temp = np.concatenate((px_temp, feature['px_temp']), axis=1)
                px_temp_test = np.concatenate((px_temp_test, feature['px_temp_test']), axis=1)

            # evaluate feature set acc
            res_temp = self.score_model(px_temp, self.py)

            # test accuracy
            model = self.clf.fit(px_temp, self.py)
            res_temp_test = model.score(px_temp_test, self.py_test)

            # add result to dict
            feature['n_best_acc'] = res_temp
            feature['n_best_acc_test'] = res_temp_test

        # optionally plot result
        if plot:
            plt.close()
            plt.plot([feature['accuracy'] for feature in sorted(feature_dict, key=lambda j: j['rank'])],
                     label='single acc')
            '''plt.scatter([feature['test_accuracy'] for feature in sorted(feature_dict, key=lambda j: j['rank'])],
                     label='single acc test')'''
            plt.plot([feature['n_best_acc'] for feature in sorted(feature_dict, key=lambda j: j['rank'])],
                     label='n best acc')
            plt.plot([feature['n_best_acc_test'] for feature in sorted(feature_dict, key=lambda j: j['rank'])],
                     label='n best acc test')
            plt.xlabel('feature no.')
            plt.ylabel('accuracy')
            plt.grid()
            plt.legend()
            plt.title(self.id)
            # plt.show()

            # save fig
            filename = self.id + '_Nbest.png'
            my_path = os.path.join(self.save_dir, filename)
            plt.savefig(my_path)

        # add result to global class variable
        if self.id not in SvmDecoder.results:
            SvmDecoder.results[self.id] = {}
        SvmDecoder.results[self.id]['Nbest'] = [{key: feature[key] for key in ['rank', 'accuracy', 'n_best_acc',
                                                                                  'n_best_acc_test', 'channel',
                                                                                  'freq_range']} for feature in
                                                sorted(feature_dict, key=lambda j: j['rank'])]

        # save the model and selected parameters to disk

    def n_best_old(self):
        # get accuracies for each featurevector
        accuracy = self.single_feature()

        # define frequently used parameters
        num_ch = self.px.shape[1]
        num_ranges = self.px.shape[0]
        N = num_ranges * num_ch

        res = []  # stores accuracy of feature vector of 1,2,...,N features
        Nbest = []  # N channels ordered by accuracy

        # create feaurevectors of 1 vector, 2 vectors, 3 vectors, ..., N vectors
        for n in range(N):
            max_acc = 0  # maximum accuracy achieved (init from 0)
            max_acc_channel = -1  # maximum accuracy achieved in this channel (init from -1)
            max_acc_range = -1  # maximum accuracy achieved in this range (init from -1)

            # go through each individual feature accuracies
            for ch in range(num_ch):
                for freq in range(num_ranges):
                    curr_acc = accuracy[freq, ch]
                    if curr_acc > max_acc:
                        max_acc = curr_acc
                        max_acc_channel = ch
                        max_acc_range = freq

            # append n-th best feature and its parameters
            Nbest.append((max_acc_range, max_acc_channel))
            accuracy[max_acc_range, max_acc_channel] = -1  # to not use this feature again

            # create the n-th featurevector that consists of best n features
            for i, nbest in enumerate(Nbest):
                freq = nbest[0]
                ch = nbest[1]
                if i == 0:
                    px_temp = self.px[freq, ch, :].reshape(-1, 1)
                else:
                    px_temp = np.concatenate((self.px[freq, ch, :].reshape(-1, 1), px_temp), axis=1)

            # SVM on featurevector
            if self.evaluation == 'cross_val':
                scores = cross_val_score(self.clf, px_temp, self.py, cv=self.cv)
                res.append(np.mean(scores))
            elif self.evaluation == 'simple_split':
                X_train, X_test, y_train, y_test = train_test_split(px_temp, self.py,
                                                                    test_size=self.test_size, shuffle=False)
                clf = self.clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                res.append(score)

        if self.id not in self.results:
            self.results[self.id] = {'single': [], 'baseline': [], 'n_best': [], 'greedy': []}
        self.results['n_best'].append(res)

    def greedy(self):
        num_ch = self.px.shape[1]
        num_ranges = self.px.shape[0]
        freq_range = np.arange(num_ranges)  # to use all ranges, i.e to only select features

        # these two variables store selected feature set and accuracy in each iteration
        best_features_in_trial = {'freq_range': [], 'channel': [], 'result': [], 'px': []}
        best_features_in_trial_test = {'result': [], 'px': []}

        max_found = False
        curr_features = 0  # to count maximum features to be selected

        # Find best feature set
        while not max_found:
            # find next best feature
            res = 0
            for ch in range(num_ch):
                # fit px (featurevector) to required model input shape
                px_temp = np.transpose(self.px[freq_range, ch, :])
                px_temp_test = np.transpose(self.px_test[freq_range, ch, :])

                # concatenate current (n-th) features with best n-1 to get feature vector
                if len(best_features_in_trial['px']) != 0:
                    px_temp = np.concatenate((best_features_in_trial['px'][-1], px_temp), axis=1)
                    px_temp_test = np.concatenate((best_features_in_trial_test['px'][-1], px_temp_test), axis=1)

                # build and evaluate model, returns accuracy result
                res_temp = self.score_model(px_temp, self.py)

                # compare current px_temp accuracy with best px_temp accuracy, if better, store it
                if res_temp > res:
                    res = res_temp
                    best_feature = {'freq_range': freq_range, 'channel': ch, 'result': res, 'px': px_temp}
                    best_feature_test = px_temp_test

            # add next best feature to preciously selected set of features (if conditions stand)
            # and evaluate on test set
            if (len(best_features_in_trial["result"]) > 1) and (not res > best_features_in_trial['result'][-1]):
                max_found = True
            else:
                # build model on final feature set
                model = self.clf.fit(best_feature['px'], self.py)

                # store best feature set with n features, and its parameters
                best_features_in_trial['freq_range'].append(best_feature['freq_range'])
                best_features_in_trial['channel'].append(best_feature['channel'])
                best_features_in_trial['result'].append(best_feature['result'])
                best_features_in_trial['px'].append(best_feature['px'])

                res_test = model.score(best_feature_test, self.py_test)  # to store test set accuracy
                best_features_in_trial_test['result'].append(res_test)
                best_features_in_trial_test['px'].append(best_feature_test)

                # increase current feature count
                curr_features += 1

                # check if max number of features is reached
                if curr_features == self.greedy_max_features:
                    max_found = True

                # print progress
                print('\tfeatures selected: ', curr_features, ', accuracy: ', round(best_feature['result'], 2),
                      'test accuracy: ', round(res_test, 2))

        # add result to global class variable
        if self.id not in SvmDecoder.results:
            SvmDecoder.results[self.id] = {}

        SvmDecoder.results[self.id]['single'] = [best_features_in_trial['result'][0],
                                                 best_features_in_trial_test['result'][0]]
        SvmDecoder.results[self.id]['greedy'] = [best_features_in_trial['result'],
                                                 best_features_in_trial_test['result'],
                                                 best_features_in_trial['channel'],
                                                 best_features_in_trial[
                                                        'freq_range']]  # TODO ezt dictionary alakba kell irni, de ehhez a kiiratast is modositani kell, meg mindenhol ahol ebbol olvas

        # save the model and selected parameters to disk
        if self.save_model:
            # save model
            filename = self.id + '_greedy_SVM.sav'
            my_path = os.path.join(self.save_dir, filename)
            with open(my_path, 'wb') as f:
                pickle.dump(model, f)

            # save selected features
            params = {'freq_range': best_features_in_trial['freq_range'], 'channel': best_features_in_trial['channel'],
                      'train_set_result': best_features_in_trial['result'],
                      'test_set_result': best_features_in_trial_test['result']}

            filename = self.id + '_greedy_params.pkl'
            my_path = os.path.join(self.save_dir, filename)
            with open(my_path, 'wb') as f:
                pickle.dump(params, f)

    def r_greedy(self):
        '''# from [freq, ch, trial] to [(freq x ch), trial]
        px_return = np.transpose(self.px.reshape(-1, self.py.shape[0]))
        px_return_test = np.transpose(self.px_test.reshape(-1, self.py_test.shape[0]))

        # these two variables store selected feature set and accuracy in each iteration
        worst_features_in_trial = {'feature_id': [], 'result': [], 'px': []}
        worst_features_in_trial_test = {'result': [], 'px': []}

        max_found = False
        curr_features = px_return.shape[1]  # to count maximum features to be excluded

        # Find best feature set
        while not max_found:
            res = 0

            for i in range(px_return.shape[1]):
                # fit px (featurevector) to required model input shape
                px_temp = np.delete(px_return, i, axis=1)

                # build and evaluate model, returns accuracy result
                res_temp = self.score_model(px_temp, self.py)

                # compare current px_temp accuracy with best px_temp accuracy, if better, store it
                if res_temp > res:
                    res = res_temp
                    worst_feature = {'feature_id': i, 'result': res, 'px': px_temp}

            if (len(worst_features_in_trial["result"]) > 1) and (not res > worst_features_in_trial['result'][-1]):
                max_found = True
            else:
                # get best feature set in current iteration
                px_return = worst_feature['px']
                px_return_test = np.delete(px_return_test, worst_feature['feature_id'], axis=1)

                # build model on final feature set
                model = self.clf.fit(px_return, self.py)

                # store best feature set with n features, and its parameters
                worst_features_in_trial['feature_id'].append(worst_feature['feature_id'])
                worst_features_in_trial['result'].append(worst_feature['result'])
                worst_features_in_trial['px'].append(worst_feature['px'])

                res_test = model.score(px_return_test, self.py_test)  # to store test set accuracy
                worst_features_in_trial_test['result'].append(res_test)
                worst_features_in_trial_test['px'].append(px_return)

                # increase current feature count
                curr_features -= 1

                # check if max number of features is reached
                if curr_features == self.r_greedy_min_features:
                    max_found = True

                # print progress
                print('\tfeatures remaining: ', curr_features, ', accuracy: ', round(worst_feature['result'], 2),
                      'test accuracy: ', round(res_test, 2))

        # add result to global class variable
        if self.id not in SvmClassifier.results:
            SvmClassifier.results[self.id] = {}
        SvmClassifier.results[self.id]['greedy Reverso'] = [worst_features_in_trial['result'],
                                                            worst_features_in_trial_test['result']]
    
        # save the model and selected parameters to disk
        if self.save_model:
            # save model
            filename =  self.id+'_greedy_SVM.sav'
            my_path =  self.save_dir + filename
            with open(my_path, 'wb') as f:
                pickle.dump(model, f)
            
            # save selected features
            params = {'freq_range':[],'channel':[],'train_set_result':[],'test_set_results':[]} 
            params['freq_range'] = best_features_in_trial['freq_range']
            params['channel'] = best_features_in_trial['channel']
            params['train_set_result'] = best_features_in_trial['result']
            params['test_set_result'] = best_features_in_trial_test['result']

            filename = self.id+'_greedy_params.pkl'
            my_path = self.save_dir + filename
            with open(my_path, 'wb') as f:
                pickle.dump(params, f)
    '''
        raise NotImplementedError

    def run(self):
        # get data and labels
        train_x = self.preprocessed_data['train_x']
        train_y = self.preprocessed_data['train_y']
        test_x = self.preprocessed_data['test_x']
        test_y = self.preprocessed_data['test_y']

        # calculate classification accuracy for each subject
        for i, name in enumerate(self.subject_ids):
            print('####  subject: ', name, '####  \n')
            # get train/test data
            # train
            px_base = np.asarray(train_x[name])
            px_base = np.transpose(px_base, axes=[2, 1, 0])  # reorder to (freq, channels, trials)
            py_base = np.asarray(train_y[name])
            # test
            px_base_test = np.asarray(test_x[name])
            px_base_test = np.transpose(px_base_test, axes=[2, 1, 0])  # reorder to ???(freq, channels, trials)
            py_base_test = np.asarray(test_y[name])

            # reshape feature vectors
            trial_pairs = self.trial_pairs

            px_dict, py_dict = featurevectors_by_trial(px_base, py_base, trial_types=trial_pairs['label_pairs'],
                                                       do_correlation_analysis=self.do_correlation_analysis,
                                                       correlation_pairs=self.correlation_pairs,
                                                       ranges=self.ranges,
                                                       corr_threshold=self.corr_threshold)

            px_dict_test, py_dict_test = featurevectors_by_trial(px_base_test, py_base_test,
                                                                 trial_types=trial_pairs['label_pairs'],
                                                                 do_correlation_analysis=self.do_correlation_analysis,
                                                                 correlation_pairs=self.correlation_pairs,
                                                                 ranges=self.ranges,
                                                                 corr_threshold=self.corr_threshold)

            # loop through each featurevector
            for j, featurevector in enumerate(px_dict):
                # handle exception: when 0 valid channels or freq ranges found based on correlation values
                '''if featurevector.shape[1] == 0:
                    for k in range(len(results)): results[k].append(-1)#?????????
                    continue'''

                # assign class variables
                self.px = featurevector
                self.py = py_dict[j]
                self.px_test = px_dict_test[j]
                self.py_test = py_dict_test[j]
                self.id = name

                # init class
                # my_model = SvmClassifier(px=px, py=py, patient_id=name, px_test=px_test, py_test=py_test)
                '''self.px = px
                self.py = py
                self.id = name
                self.px_test = px_test
                self.py_test = py_test'''
                '''if isinstance(px_test, np.ndarray):
                            self.px_test = px_test
                        else:
                            self.px_test = []
                        if isinstance(px_test, np.ndarray):
                            self.py_test = py_test
                        else:
                            self.py_test = []
                        self.id = patient_id'''

                # standardize data
                self.standardize_feature_vectors(same_scaler=self.same_scaler)

                # BUILD AND SAVE MODELS ###
                for model_type in self.model_types:
                    # SINGLE FEATURE strategy
                    # my_model.single_feature()# this list stores accuracy of each feature for the given classification task
                    if model_type == 'eval_all':
                        # evaluate all features individually
                        print('**evalutaing all features')
                        self.evaluate_single_features()
                    elif model_type == 'baseline':
                        # BASELINE strategy
                        print('**baseline')
                        self.baseline()
                    elif model_type == 'Nbest':
                        # N-BEST strategy
                        print('**N best')
                        self.n_best(plot=True)
                    elif model_type == 'greedy':
                        # GREEDY strategy
                        print('**greedy')
                        self.greedy()
                    elif model_type == 'rGreedy':
                        # reverse GREEDY strategy
                        print('**greedy REVERSO')
                        self.r_greedy()

        # print results as table
        results = self.results
        tables_all, table2 = self.print_results(results)  # todo ezt osszevonni

        # save parameters and resulting accuracies
        if self.save_info:
            if not os.path.exists(self.sp):
                os.makedirs(self.sp)

            # save model settings as class variables
            with open(os.path.join(self.sp, 'svm_settings.json'), 'w') as fp:
                json.dump(self.config, fp)

            # save accuracies as .pkl
            with open(os.path.join(self.sp, 'accs_all.pkl'), 'wb') as f:
                pickle.dump(SvmDecoder.results, f)

            # save accuracies as .txt
            with open(os.path.join(self.sp, 'accs_all.txt'), 'w') as f:
                f.write(str(SvmDecoder.results))

            # save summary as .txt
            with open(os.path.join(self.sp, 'accs_summary.txt'), 'w') as f:
                for result in tables_all:
                    f.write(result['name'] + '\n')
                    f.write(result['table'] + '\n\n')
                f.write('mean\n')
                f.write(table2)

        print('results and models saved at:\n{}'.format(self.sp))
        return results

    @classmethod
    def set_class_vars(cls, modelSettings):
        cls.scaler = modelSettings['scaler']
        cls.clf = modelSettings['clf']
        cls.evaluation = modelSettings['evaluation']
        cls.save_model = modelSettings['save_model']
        cls.save_dir = modelSettings['save_dir']
        cls.greedy_max_features = modelSettings['greedy_max_features']
        cls.r_greedy_min_features = modelSettings['reverse_greedy_min_features']
        cls.cv = modelSettings['cv']
        cls.test_size = modelSettings['test_size']

        # feature engineering parameters
        # cls.trial_pairs=modelSettings['trial_pairs']
        # cls.ranges = modelSettings['ranges']
        # cls.do_correlation_analysis = modelSettings['do_correlation_analysis']
        # cls.correlation_pairs = modelSettings['correlation_pairs']
        # cls.multiple_rest = modelSettings['multiple_rest']
        # cls.corr_threshold = modelSettings['corr_threshold']

    @staticmethod
    def print_results(results):
        train_all = []
        test_all = []
        tables_all = []
        for name in results.keys():
            print(name)
            result = results[name]
            headers = []
            train_row = ['train']
            test_row = ['test']
            for trial in result.keys():
                if trial == 'eval_all': continue  # todo erre is raigazitani
                elif trial == 'greedy':
                    res_train = round(result[trial][0][-1], 2)
                    res_test = round(result[trial][1][-1], 2)

                    train_row.append(res_train)
                    test_row.append(res_test)
                    train_all.append(res_train)
                    test_all.append(res_test)

                    headers.append(trial)
                elif trial == 'Nbest':
                    continue  # TODO megirni ezt
                else:
                    res_train = round(result[trial][0], 2)
                    res_test = round(result[trial][1], 2)

                    train_row.append(res_train)
                    test_row.append(res_test)
                    train_all.append(res_train)
                    test_all.append(res_test)

                    headers.append(trial)

            table = tabulate([train_row, test_row], headers=headers)
            tables_all.append({'name': name, 'table': table})
            print(table, '\n')

        print('mean')
        train_mean = np.mean(np.reshape(train_all, (-1, len(headers))), axis=0)
        train_std = np.std(np.reshape(train_all, (-1, len(headers))), axis=0)
        test_mean = np.mean(np.reshape(test_all, (-1, len(headers))), axis=0)
        test_std = np.std(np.reshape(test_all, (-1, len(headers))), axis=0)

        train_out = ['train']
        test_out = ['test']
        for i, element in enumerate(train_mean):
            train_out.append(str(round(element, 2)) + '+-' + str(round(train_std[i], 2)))
            test_out.append(str(round(test_mean[i], 2)) + '+-' + str(round(test_std[i], 2)))

        table2 = tabulate([train_out, test_out], headers=headers)
        print(table2)

        return tables_all, table2
