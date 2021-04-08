from feature_transformation import *
import os

import numpy as np
#from numpy import squeeze, power, mean, absolute, var, concatenate
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
#from scipy.stats import ttest_ind
#import matplotlib.pyplot as plt
#import h5py
import pickle
#from pathlib import Path
#from tabulate import tabulate
#import csv

class svmClassifier(object):
    """
    docstring
    """
    # SVM parameters
    scaler=StandardScaler()
    clf=svm.SVC()
    evaluation='cross_val'
    save_model=False
    save_dir=None
    cv=5
    test_size=0.2
    greedy_max_features=-1
    r_greedy_min_features=-1

    # to collect results
    results={}

    def __init__(self, id, px, py, px_test=None, py_test=None, *args, **kwargs):
        self.px=px
        self.py=py
        if isinstance(px_test, np.ndarray):
            self.px_test=px_test
        else: 
            self.px_test=[]
        if  isinstance(px_test, np.ndarray):
            self.py_test=py_test
        else: 
            self.py_test=[]
        self.id=id

    def standardize_featurevectors(self, same_scaler=True):
        #reshape(flatten) training data
        self.px = np.asarray(self.px, dtype='double') # "Avoiding data copy: For SVC, SVR, NuSVC and NuSVR, if the data passed to certain methods is not C-ordered contiguous and double precision, it will be copied before calling the underlying C implementation."
        px_temp = np.transpose(self.px.reshape(-1, self.px.shape[-1]))

        # fit scaler to train data
        scaler = self.scaler.fit(px_temp)    

        # scale training data
        self.px = np.transpose(scaler.transform(px_temp)).reshape(self.px.shape)

        # scale test data (with the same scaler!!)
        if not self.px_test==[]:
            self.px_test = np.asarray(self.px_test, dtype='double') # "Avoiding data copy: For SVC, SVR, NuSVC and NuSVR, if the data passed to certain methods is not C-ordered contiguous and double precision, it will be copied before calling the underlying C implementation."
            px_test_temp = np.transpose(self.px_test.reshape(-1, self.px_test.shape[-1]))
            if same_scaler:
                self.px_test = np.transpose(scaler.transform(px_test_temp)).reshape(self.px_test.shape)
            else:
                scaler_test = self.scaler.fit(px_test_temp)
                self.px_test = np.transpose(scaler_test.transform(px_test_temp)).reshape(self.px_test.shape)

    def scoreModel(self, px, py):
        ## build model
        if self.evaluation == 'cross_val':
            scores = cross_val_score(self.clf, px, py, cv=self.cv)
            res = np.mean(scores)
            #print(scores.mean(),scores.std())
        elif self.evaluation == 'simple_split':
            X_train, X_test, y_train, y_test = train_test_split(px, py, test_size=self.test_size, shuffle=False)
            clf = self.clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            res = score
        else:
            import warnings
            warnings.warn("invalid evaluatuion mode '" + self.evaluation+"'")
        
        return res

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
                    X_train, X_test, y_train, y_test = train_test_split(px_temp, self.py, test_size=self.test_size, shuffle=False)
                    clf = self.clf.fit(X_train, y_train)
                    score = clf.score(X_test, y_test)
                    res_temp = score

                if res_temp>res: 
                    res=res_temp

        ## save result
        if self.id not in svmClassifier.results:
            svmClassifier.results[self.id] = {'single':[],'baseline':[],'N_best':[], 'greedy':[]}
        svmClassifier.results[self.id]['single'].append(res)
        print('\taccuracy: ', res)'''

    def baseline(self):
        # fit px (featurevector) to required model input shape
        # train
        px_temp = np.transpose(self.px.reshape(-1,self.py.shape[0]))
        # test
        px_temp_test = np.transpose(self.px_test.reshape(-1,self.py_test.shape[0]))
        
        # build and evaluate model, returns accuracy result
        res = self.scoreModel(px_temp, self.py)
        
        # build model on (final) px_temp
        model = self.clf.fit(px_temp, self.py)

        # test accuracy
        res_test = model.score(px_temp_test, self.py_test)

        # add result to global class variable
        if self.id not in svmClassifier.results:
            svmClassifier.results[self.id] = {}
        svmClassifier.results[self.id]['baseline'] = [res, res_test]

        # save the model to disk
        if self.save_model:            
            filename = self.id+'_baseline_SVM.sav'
            my_path =  self.save_dir + filename
            
            with open(my_path, 'wb') as f:
                pickle.dump(model, f)
        
        # print results
        print('\taccuracy: ', round(res, 2), 'test accuracy: ', round(res_test, 2))

    '''def N_best(self, plot=False):
        print('**N_best')
        # get accuracies for each featurevector
        accuracy = self.single_feature()

        # define frequently used parameters
        num_ch = self.px.shape[1]
        num_ranges = self.px.shape[0]
        N = num_ranges*num_ch

        res = [] # stores accuracy of feature vector of 1,2,...,N features
        Nbest = [] # N channels ordered by accuracy

        # create feaurevectors of 1 vector, 2 vectors, 3 vectors, ..., N vectors
        for n in range(N):
            max_acc = 0 # maximum accuracy achieved (init from 0)
            max_acc_channel = -1 # maximum accuracy achieved in this channel (init from -1)
            max_acc_range = -1 # maximum accuracy achieved in this range (init from -1)

            # go through each individual featre accuracies
            for ch in range(num_ch):
                for freq in range(num_ranges):
                    curr_acc = accuracy[freq,ch]
                    if curr_acc > max_acc:
                        max_acc = curr_acc
                        max_acc_channel = ch
                        max_acc_range = freq

            # append n-th best feature and its parameters
            Nbest.append((max_acc_range, max_acc_channel))
            accuracy[max_acc_range, max_acc_channel] = -1 # to not use this feature again

            # create the n-th featurevector that consists of best n features
            for i, nbest in enumerate(Nbest):
                freq = nbest[0]
                ch = nbest[1]
                if i == 0:
                    px_temp = self.px[freq, ch, :].reshape(-1,1)
                else:
                    px_temp = np.concatenate((self.px[freq, ch, :].reshape(-1,1), px_temp), axis=1)

            # SVM on featurevector
            if self.evaluation == 'cross_val':
                scores = cross_val_score(self.clf, px_temp, self.py, cv=self.cv)
                res.append(np.mean(scores))
            elif self.evaluation == 'simple_split':
                X_train, X_test, y_train, y_test = train_test_split(px_temp, self.py, test_size=self.test_size, shuffle=False)
                clf = self.clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                res.append(score)

        if self.id not in self.results:
            self.results[self.id] = {'single':[],'baseline':[],'N_best':[], 'greedy':[]}  
        self.results['N_best'].append(res)'''

    def greedy(self):
        num_ch = self.px.shape[1]
        num_ranges = self.px.shape[0]
        
        #these two variables store selected feature set and accuracy in each iteration
        best_features_in_trial = {'freq_range':[],'channel':[],'result':[],'px':[]} 
        best_features_in_trial_test={'result':[], 'px':[]}

        max_found = False
        curr_features = 0   # to count maximum features to be selected
        
        ## Find best feature set
        while not max_found:
            #best_feature = {} # store the best feature in given trial_pair,channel
            res = 0

            for ch in range(num_ch):
                for freq_range in range(num_ranges):
                    # fit px (featurevector) to required model input shape
                    px_temp = np.transpose(self.px[freq_range,ch,:]).reshape(-1,1)
                    px_temp_test = np.transpose(self.px_test[freq_range,ch,:]).reshape(-1,1)
                    
                    # concatenate current (n-th) features with best n-1 to get feature vector
                    if len(best_features_in_trial['px'])!=0:
                        px_temp = np.concatenate((best_features_in_trial['px'][-1],px_temp), axis=1)
                        px_temp_test = np.concatenate((best_features_in_trial_test['px'][-1], px_temp_test), axis=1)
                    
                    # build and evaluate model, returns accuracy result
                    res_temp = self.scoreModel(px_temp, self.py)

                    # compare current px_temp accuracy with best px_temp accuracy, if better, store it
                    if res_temp > res:
                        res = res_temp
                        best_feature = {'freq_range':freq_range,'channel':ch,'result':res,'px':px_temp}
                        best_feature_test = px_temp_test
                        # concatenate current (n-th) features with best n-1 to get feature vector
            
            if (len(best_features_in_trial["result"]) > 1) and (not res > best_features_in_trial['result'][-1]):
                max_found = True
            else: 
                # build model on final feature set
                model = self.clf.fit(best_feature['px'], self.py)

                #store best feature set with n features, and its parameters
                best_features_in_trial['freq_range'].append(best_feature['freq_range'])
                best_features_in_trial['channel'].append(best_feature['channel'])
                best_features_in_trial['result'].append(best_feature['result'])
                best_features_in_trial['px'].append(best_feature['px'])
                
                res_test = model.score(best_feature_test, self.py_test) #to store test set accuracy
                best_features_in_trial_test['result'].append(res_test)
                best_features_in_trial_test['px'].append(best_feature_test)
                
                # increase current feature count
                curr_features +=1
                
                #check if max number of features is reached
                if curr_features == self.greedy_max_features:
                    max_found = True
                
                #print progress
                print('\tfeatures selected: ', curr_features, ', accuracy: ', round(best_feature['result'],2), 'test accuracy: ', round(res_test ,2))

        ## add result to global class variable
        res = best_features_in_trial['result'][-1]
        if self.id not in svmClassifier.results:
            svmClassifier.results[self.id] = {}
        svmClassifier.results[self.id]['single'] = [best_features_in_trial['result'][0], best_features_in_trial_test['result'][0]]
        svmClassifier.results[self.id]['greedy'] = [best_features_in_trial['result'], best_features_in_trial_test['result']]     

        ## save the model and selected parameters to disk
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

    def r_greedy(self):
        # from [freq, ch, trial] to [(frek x ch), trial]
        px_return = np.transpose(self.px.reshape(-1,self.py.shape[0]))
        px_return_test = np.transpose(self.px_test.reshape(-1,self.py_test.shape[0]))

        #these two variables store selected feature set and accuracy in each iteration
        worst_features_in_trial = {'feature_id':[],'result':[],'px':[]} 
        worst_features_in_trial_test={'result':[], 'px':[]}

        max_found = False
        curr_features = px_return.shape[1]   # to count maximum features to be excluded
       
        ## Find best feature set
        while not max_found:
            res = 0

            for i in range(px_return.shape[1]):
                # fit px (featurevector) to required model input shape
                px_temp = np.delete(px_return, i, axis=1)
                
                # build and evaluate model, returns accuracy result
                res_temp = self.scoreModel(px_temp, self.py)

                # compare current px_temp accuracy with best px_temp accuracy, if better, store it
                if res_temp > res:
                    res = res_temp
                    worst_feature = {'feature_id':i,'result':res, 'px':px_temp}
            
            if (len(worst_features_in_trial["result"]) > 1) and (not res > worst_features_in_trial['result'][-1]):
                max_found = True
            else: 
                # get best featureset in current iteration
                px_return = worst_feature['px']
                px_return_test =  np.delete(px_return_test, worst_feature['feature_id'], axis=1)

                # build model on final feature set
                model = self.clf.fit(px_return, self.py)

                #store best feature set with n features, and its parameters
                worst_features_in_trial['feature_id'].append(worst_feature['feature_id'])
                worst_features_in_trial['result'].append(worst_feature['result'])
                worst_features_in_trial['px'].append(worst_feature['px'])
                
                res_test = model.score(px_return_test, self.py_test) #to store test set accuracy
                worst_features_in_trial_test['result'].append(res_test)
                worst_features_in_trial_test['px'].append(px_return)
                
                # increase current feature count
                curr_features -=1
                
                #check if max number of features is reached
                if curr_features == self.r_greedy_min_features:
                    max_found = True
                
                #print progress
                print('\tfeatures remaining: ', curr_features, ', accuracy: ', round(worst_feature['result'],2), 'test accuracy: ', round(res_test ,2))
        
        ## add result to global class variable
        res = worst_features_in_trial['result'][-1]
        if self.id not in svmClassifier.results:
            svmClassifier.results[self.id] = {}
        svmClassifier.results[self.id]['greedy Reverso'] = [worst_features_in_trial['result'], worst_features_in_trial_test['result']]     
        '''
        ## save the model and selected parameters to disk
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
    @classmethod
    def set_class_vars(cls, modelSettings):
        cls.scaler=modelSettings['scaler']
        cls.clf=modelSettings['clf']
        cls.evaluation=modelSettings['evaluation']
        cls.save_model=modelSettings['save_model']
        cls.save_dir=modelSettings['save_dir']
        cls.greedy_max_features=modelSettings['greedy_max_features']
        cls.r_greedy_min_features
        cls.cv=modelSettings['cv']
        cls.test_size=modelSettings['test_size']
        '''
        # feature engineering parameters
        cls.trial_pairs=modelSettings['trial_pairs']
        cls.ranges=modelSettings['ranges']
        cls.do_correlation_analysis=modelSettings['do_correlation_analysis']
        cls.correlation_pairs=modelSettings['correlation_pairs']
        cls.multiple_rest=modelSettings['multiple_rest']
        cls.corr_treshold=modelSettings['corr_treshold']'''

