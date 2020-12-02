try:
    from .preprocessor_stanford_motor import *
except ImportError:
    from preprocessor_stanford_motor import *

from sklearn import svm

subjects=['bp','ca','cc','de','fp','gc','hh','hl','jc',
          'jm','jt','rh','rr','ug','wc','zt']

for subject in subjects:
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, "../data/raw_data/stanford_motorbasic/")
    filename = path + subject + '_mot_t_h.mat'

    preproc = StanfordPreprocessor('')    

    X, y = preproc.load_data_and_labels(filename)
    print(subject)
    print(X.shape, y.shape)

    X, y = preproc.preprocess_with_label(X, y)

    print(X.shape, y.shape, y)

    ##### SVM #####
    id_hand_v_rest = np.where(np.logical_or(np.equal(y,12),np.equal(y,120)))
    id_tongue_v_rest = np.where(np.logical_or(np.equal(y,11),np.equal(y,110)))
    id_hand_v_tongue = np.where(np.logical_or(np.equal(y,12),np.equal(y,110)))

    X = np.squeeze(X[:,:,id_hand_v_rest])
    y = np.squeeze(y[id_hand_v_rest])

    # freq ranges: low Alpha, high alpha, beta, low gamma, high gamma, kszi
    ranges = [range(7,13), range(10,14), range(14,26), range(26,36), range(36,70), range(76,151)]

    # SVM for each range
    for freq_range in ranges:
        X_temp = np.transpose(sum(X[freq_range,:,:]))
        
        X_train , X_test , y_train, y_test = train_test_split(X_temp, y)
        clf = svm.SVC()
        clf.fit(X_train, y_train)
        print(freq_range, clf.score(X_test, y_test))