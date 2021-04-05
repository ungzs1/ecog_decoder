import numpy as np
from numpy import squeeze, power, mean, absolute, var, concatenate
from scipy.stats import ttest_ind
import copy

def rsa(d1,d2):
    '''this function calculates the signed r-squared cross-correlation for vectors d1 and d2.  it is signed to reflect d1>d2'''
    np.reshape(d1, (1,-1))
    np.reshape(d2, (1,-1))

    outrsa = (power((mean(d1)-mean(d2)), 3) 
    /absolute(mean(d1)-mean(d2)) 
    /var(concatenate([d1,d2], 1), ddof=1) #Delta Degrees of Freedom: the divisor used in the calculation is N - ddof, where N represents the number of elements. By default, ddof is zero. 
    *(d1.shape[1]*d2.shape[1]) 
    /power(concatenate([d1,d2], 1).shape[1], 2))

    return outrsa

def get_correlation(px, py):
    ''' Takes a feature vector and corresponding labels as input. 
    Returns r-value and p-value for Low Frequency Band and High Freqvency Band (r-value from signed r-squared cross-correlation and p-value from unpaired t-test)
    '''

    # Normalize data & isolate HFB and LFB
    num_chans = px.shape[1]
    num_trials = px.shape[2]

    px_mean = np.mean(px, axis=2)

    trials = np.unique(py)

    # Isolate HFB and LFB
    px_temp = np.divide(px, np.dstack([px_mean]*num_trials)) #normalize data:every datapoint is divided by the average power of the same freq at the same channel
    LFB_trials = squeeze(sum(px_temp[7:32,:,:],0))
    HFB_trials= squeeze(sum(px_temp[75:100,:,:],0))

    # Comparisons - signed r-squared and unpaired t-test hand=hand v rest, tongue=tongue v rest; r = signed square cross correlation, p=p-value from paired t-test
    r_LFB = []; r_HFB = []
    p_LFB = []; p_HFB = []

    for ch in range(num_chans):
        # extract trial type 1 data
        LFB_t1 = LFB_trials[ch, np.where(py==trials[0])]
        HFB_t1 = HFB_trials[ch, np.where(py==trials[0])]
        # extract trial type 2 data
        LFB_t2 = LFB_trials[ch, np.where(py==trials[1])]
        HFB_t2 = HFB_trials[ch, np.where(py==trials[1])]
        
        # calculate r-squared cross correlation 
        r_LFB.append(rsa(LFB_t1, LFB_t2))
        r_HFB.append(rsa(HFB_t1, HFB_t2))

        # calculate p-value of t-test
        p_LFB.append(float(ttest_ind(LFB_t1, LFB_t2, 1)[1]))
        p_HFB.append(float(ttest_ind(HFB_t1, HFB_t2, 1)[1]))
        
    r_all = [np.asarray(r_LFB), np.asarray(r_HFB)]
    p_all = [np.asarray(p_LFB), np.asarray(p_HFB)]
            
    return r_all, p_all

def featurevectors_by_trial(px_base, py_base, trial_types, do_correlation_analysis=False, correlation_pairs=None, ranges=None, corr_treshold=1):
    '''This function reshapes the feature vector and reduces its size. Returns a list of N feature vectors, where N is the number of given trial pairs. 
    It calculates the mean power over the given frequency ranges. It     also ignores channels with p-value over 'corr_treshold' for the given trial type, 
    based on unpaired t-test (default is 'corr_treshold=1', where all channels are included)'''

    # list of veature vectors for each trial type
    px = []
    py = []
    
    for i, trial_type in enumerate(trial_types):
        #remove uncorrelationg channels from px
        if do_correlation_analysis == True:
             # to collect valid channels
            valid_channels = []  

            # if correlation pairs are not given, use current trial pair instead
            if correlation_pairs == None:
                correlation_pairs = [trial_type]

            # getting valid channels for each given correlation pair
            for correlation_pair in correlation_pairs:
                curr_trials = np.where((py_base == correlation_pair[0])|(py_base == correlation_pair[1]))
                px_temp = np.squeeze(px_base[:,:,curr_trials])
                py_temp = np.squeeze(py_base[curr_trials])

                # get correlation values
                r_all, p_all = get_correlation(px_temp, py_temp)
                p_LFB = p_all[0]
                p_HFB = p_all[1]

                # get valid channels based on correlation and corr_treshold
                valid_channels_temp = np.where(p_LFB<corr_treshold)+np.where(p_HFB<corr_treshold)
                valid_channels += valid_channels_temp

            valid_channels = np.concatenate(valid_channels, axis=0)
            valid_channels = list(np.unique(valid_channels))
            
            px_temp = np.squeeze(px_base[:,valid_channels,:])
        else:
            px_temp = np.squeeze(px_base[:,:,:])
        
        # get current trials
        condition = (py_base==trial_type[0])
        for lable in trial_type:
            condition = condition|(py_base==lable)
        curr_trials = np.where(condition)[0]

        px_temp = np.squeeze(px_temp[:,:,curr_trials])
        py_temp = np.squeeze(py_base[curr_trials])

        # avg ranges
        if ranges != None:
            featurevector = [] # initialize featurevector
            for freq_range in ranges:
                featurevector.append(np.mean(px_temp[freq_range,:,:], axis = 0))
            px_temp = featurevector
            
        px_temp = np.asarray(px_temp)

        px.append(px_temp)
        py.append(py_temp)
    
    return px, py

