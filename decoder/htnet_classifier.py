# Custom imports
from .htnet_model import htnet

# import warnings
import os, pickle, time
import numpy as np

import tensorflow as tf
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt


class HtnetDecoder:
    def __init__(self, save_rootpath=None, lp=None, pats_ids_in=None, test_day=None, combined_sbjs=False, n_folds=3,
                 hyps=None, epochs=30, patience=30, spec_meas_tail=None, do_log=False, randomize_events=False,
                 n_evs_per_sbj=500, n_chans_all=140, dipole_dens_thresh=0.2, rem_bad_chans=True, models=None,
                 save_suffix='', n_estimators=150, max_depth=8, overwrite=True, rand_seed=1337,
                 loss='categorical_crossentropy', optimizer='adam', early_stop_monitor='val_loss', n_test=1, n_val=4,
                 custom_rois=True, n_train=7, compute_val='power', ecog_srate=500, half_n_evs_test='nopad',
                 trim_n_chans=True):

        # database information
        self.save_rootpath = save_rootpath  # path to save directory (subfolders will be created here)
        self.lp = lp  # load path
        self.pats_ids_in = pats_ids_in
        self.test_day = test_day

        # Tailored decoder parameters
        self.combined_sbjs = combined_sbjs
        self.n_folds = n_folds
        if hyps is not None:
            self.hyps = hyps
        else:
            self.hyps = {'F1': 8,
                         'dropoutRate': 0.25,
                         'kernLength': 16,
                         'kernLength_sep': 56,
                         'dropoutType': 'Dropout',
                         'D': 2,
                         'n_estimators': 240,
                         'max_depth': 9,
                         'F2': 16}
        self.epochs = epochs
        self.patience = patience
        if spec_meas_tail is not None:
            self.spec_meas_tail = spec_meas_tail
        else:
            self.spec_meas_tail = ['power', 'power_log', 'relative_power', 'phase', 'freqslide']

        # decoder training parameters
        self.do_log = do_log
        self.compute_val = compute_val
        self.randomize_events = randomize_events

        # NN model parameters with default values
        self.n_evs_per_sbj = n_evs_per_sbj
        self.n_chans_all = n_chans_all
        self.dipole_dens_thresh = dipole_dens_thresh
        self.rem_bad_chans = rem_bad_chans
        if models is not None:
            self.models = models
        else:
            self.models = ['eegnet_hilb', 'eegnet', 'rf', 'riemann']
        self.save_suffix = save_suffix
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.overwrite = overwrite
        self.rand_seed = rand_seed
        self.loss = loss
        self.optimizer = optimizer
        self.early_stop_monitor = early_stop_monitor
        self.n_test = n_test
        self.n_val = n_val
        self.custom_rois = custom_rois
        self.n_train = n_train
        self.ecog_srate = ecog_srate
        self.half_n_evs_test = half_n_evs_test
        self.trim_n_chans = trim_n_chans

        # HTnet parameters
        self.nROIs = 100  # not yet used
        self.proj_mat_out = None  # not yet used
        self.sbj_order_train = None  # not yet used
        self.sbj_order_validate = None  # not yet used
        self.sbj_order_test = None  # not yet used

    def save_res_as_plt(self):
        # %%
        accs_all = {}
        for folder in self.spec_meas_tail:
            lp = os.path.join(self.save_rootpath, 'single_sbjs_' + folder)
            acc_temp = []
            for model in self.models:
                acc_per_model = []
                for patient in self.pats_ids_in:
                    fname = 'acc_{}_{}.npy'.format(model, patient)
                    acc = np.load(os.path.join(lp, fname))
                    acc_mean_train = np.mean(acc[:, 0])
                    acc_mean_val = np.mean(acc[:, 1])
                    acc_mean_test = np.mean(acc[:, 2])
                    acc_per_model.append([acc_mean_train, acc_mean_val, acc_mean_test])
                acc_temp.append(acc_per_model)
            accs_all[folder] = acc_temp

        # create plots
        fsize = 5
        fig, ax = plt.subplots(len(self.spec_meas_tail), 3, figsize=(3 * fsize, len(self.spec_meas_tail) * fsize),
                               facecolor='white')

        if len(self.spec_meas_tail) == 1:
            for j, set in enumerate(['train', 'val', 'test']):
                ax[j].scatter(self.pats_ids_in, [acc[j] for acc in accs_all[self.spec_meas_tail[0]][0]],
                              label=self.models[0])
                ax[j].scatter(self.pats_ids_in, [acc[j] for acc in accs_all[self.spec_meas_tail[0]][1]],
                              label=self.models[1])
                ax[j].title.set_text(folder + ' / ' + set)
                ax[j].set_ylim([0.3, 1.1])
                ax[j].legend()
                ax[j].grid()
        else:
            for i, folder in enumerate(self.spec_meas_tail):
                for j, set in enumerate(['train', 'val', 'test']):
                    ax[i][j].scatter(self.pats_ids_in, [acc[j] for acc in accs_all[folder][0]], label=self.models[0])
                    ax[i][j].scatter(self.pats_ids_in, [acc[j] for acc in accs_all[folder][1]], label=self.models[1])
                    ax[i][j].title.set_text(folder + ' / ' + set)
                    ax[i][j].set_ylim([0.3, 1.1])
                    ax[i][j].legend()
                    ax[i][j].grid()

        fig.suptitle('HTnet accuracies')
        plt.savefig(os.path.join(self.save_rootpath, 'accs_plot'), facecolor=fig.get_facecolor())

    def set_tailored_hyperparameters(self):
        raise NotImplementedError

    def load_data(self, patient, randomize_events=True):
        raise NotImplementedError

    def rf_model(self):
        raise NotImplementedError

    def riemann_model(self):
        raise NotImplementedError

    def cnn_model(self, X_train, y_train, X_validate, y_validate, X_test, y_test, chckpt_path, modeltype):
        """
        Perform NN model fitting based on specified prarameters.
        """
        # Logic to determine how to run model
        useHilbert = True if modeltype == 'eegnet_hilb' else False  # True if want to use Hilbert transform layer

        # number of unique classes
        nb_classes = len(np.unique(y_train, axis=0))

        # Load NN model
        model = htnet(nb_classes, Chans=X_train.shape[1], Samples=X_train.shape[2],
                      dropoutRate=self.hyps['dropoutRate'], kernLength=self.hyps['kernLength'],
                      F1=self.hyps['F1'], D=self.hyps['D'], F2=self.hyps['F2'],
                      dropoutType=self.hyps['dropoutType'], kernLength_sep=self.hyps['kernLength_sep'],
                      ROIs=self.nROIs, useHilbert=useHilbert, do_log=self.do_log,
                      compute_val=self.compute_val, data_srate=self.ecog_srate)

        # Set up compiler, checkpointer, and early stopping during model fitting
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])

        # numParams = model.count_params() # count number of parameters in the model
        checkpointer = ModelCheckpoint(filepath=chckpt_path, verbose=1, save_best_only=True)
        early_stop = EarlyStopping(monitor=self.early_stop_monitor, mode='min',
                                   patience=self.patience,
                                   verbose=0)  # stop if val_loss doesn't improve after certain # of epochs

        # Perform model fitting in Keras (model inputs differ depending on whether or not to project to roi's
        t_start_fit = time.time()
        fittedModel = model.fit(X_train, y_train, batch_size=16, epochs=self.epochs, verbose=2,
                                validation_data=(X_validate, y_validate),
                                callbacks=[checkpointer, early_stop])
        t_fit_total = time.time() - t_start_fit

        # Get the last epoch for training
        last_epoch = len(fittedModel.history['loss'])
        if last_epoch < self.epochs:
            last_epoch -= self.patience  # revert to epoch where best model was found
        print("Last epoch was: ", last_epoch)

        # Load model weights from best model and compute train/val/test accuracies
        model.load_weights(chckpt_path)

        accs_lst = []

        preds_train = model.predict(X_train).argmax(axis=-1)
        preds_validate = model.predict(X_validate).argmax(axis=-1)
        preds_test = model.predict(X_test).argmax(axis=-1)

        accs_lst.append(np.mean(preds_train == y_train.argmax(axis=-1)))
        accs_lst.append(np.mean(preds_validate == y_validate.argmax(axis=-1)))
        accs_lst.append(np.mean(preds_test == y_test.argmax(axis=-1)))

        tf.keras.backend.clear_session()  # avoids slowdowns when running fits for many folds

        return accs_lst, np.array([last_epoch, t_fit_total])  # TODO A MODELT NEM MENTI EL!!!!!!!!

    def run_nn_models(self):
        """
        Main function that prepares data and aggregates accuracy values from model fitting.
        Note that overwrite variable no longer does anything??????????.
        Also note that ecog_srate is only needed for frequency sliding computation in
        neural net (if compute_val=='freqslide')
        """

        # Ensure pats_ids_in and models variables are lists
        if self.pats_ids_in is None:
            raise ValueError("ValueError 'pats_ids_in' not specified")
        if not isinstance(self.pats_ids_in, list):
            raise ValueError("ValueError 'pats_ids_in' not a list but it must be")
        if not isinstance(self.models, list):
            raise ValueError("ValueError 'models' not a list but it must be")

        # Save pickle file with dictionary of input parameters ,useful for reproducible dataset splits and model fitting
        params_dict = {key: value for key, value in self.__dict__.items() if
                       not key.startswith('__') and not callable(key)}
        f = open(os.path.join(self.sp, 'param_file.pkl'), 'wb')
        pickle.dump(params_dict, f)
        f.close()

        # Set random seed
        np.random.seed(self.rand_seed)

        for pat_id_curr in self.pats_ids_in:
            # Load data
            X, y, X_test, y_test = self.load_data(patient=pat_id_curr, randomize_events=self.randomize_events)

            # Iterate across all model types specified
            for modeltype in self.models:
                # Reformat data based on model
                if modeltype == 'rf' or modeltype == 'riemann':
                    y2 = y.copy()
                    y_test2 = y_test.copy()
                    X2 = X.copy()
                    X_test2 = X_test.copy()
                else:
                    y2 = np_utils.to_categorical(y - 1)
                    y_test2 = np_utils.to_categorical(y_test - 1)
                    X2 = np.expand_dims(X, 1)
                    X_test2 = np.expand_dims(X_test, 1)

                # Create splits for train/val and fit model
                split_len = X2.shape[0] // self.n_folds
                accs = np.zeros([self.n_folds, 3])
                last_epochs = np.zeros([self.n_folds, 2])

                for frodo in range(self.n_folds):
                    val_inds = np.arange(0, split_len) + (frodo * split_len)
                    train_inds = np.setdiff1d(np.arange(X2.shape[0]), val_inds)  # take all events not in val set

                    # Split data and labels into train/val sets
                    X_train = X2[train_inds, ...]
                    Y_train = y2[train_inds]
                    X_validate = X2[val_inds, ...]
                    Y_validate = y2[val_inds]

                    # Create and fit model, evaluate it
                    if modeltype == 'rf':
                        accs_lst, last_epoch_tmp = self.rf_model()  # todo not implemented
                    if modeltype == 'riemann':
                        accs_lst, last_epoch_tmp = self.riemann_model()  # todo not implemented
                    else:
                        # Fit NN model and store accuracies
                        chckpt_path = os.path.join(self.sp,
                                                   'checkpoint_{}_{}_fold{}{}.h5'.format(modeltype, pat_id_curr,
                                                                                         str(frodo), self.save_suffix))

                        # rearrange axis to fit on htnet model, correct form is (trial, channel, timestep, 1)
                        X_train = np.moveaxis(X_train, 1, -1)
                        X_validate = np.moveaxis(X_validate, 1, -1)
                        X_test3 = np.moveaxis(X_test2, 1, -1)

                        # build and evaulate model
                        accs_lst, last_epoch_tmp = self.cnn_model(X_train=X_train, y_train=Y_train,
                                                                  X_validate=X_validate, y_validate=Y_validate,
                                                                  X_test=X_test3, y_test=y_test2,
                                                                  chckpt_path=chckpt_path, modeltype=modeltype)

                        # save accuracies in variable, 0:train, 1:validate, 2:test
                        for ss in range(3):
                            accs[frodo, ss] = accs_lst[ss]

                        last_epochs[frodo, :] = last_epoch_tmp

                # Save accuracies (train/val/test)
                np.save(os.path.join(self.sp, 'acc_{}_{}{}.npy'.format(modeltype, pat_id_curr,
                                                                       self.save_suffix)),
                        accs)
                # save last epoch data ??
                np.save(os.path.join(self.sp, 'last_training_epoch_gen_tf_{}_{}{}.npy'.format(modeltype,
                                                                                              pat_id_curr,
                                                                                              self.save_suffix)),
                        last_epochs)

        # Return validation accuracy for hyperparameter tuning (assumes only 1 model and 1 subject)
        return accs[:, 1].mean()

    def train_tailored_decoder(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify GPU to use

        t_start = time.time()

        # call subcalss function to set tailored decoder hyperparameters as class variables - will be useful if we add
        # more decoder types, sincee these call the same variables with different values
        self.set_tailored_hyperparameters()

        # build models for each measurement type
        for s, val in enumerate(self.spec_meas_tail):
            self.do_log = True if val == 'power_log' else False
            self.compute_val = 'power' if val == 'power_log' else val
            self.sp = os.path.join(self.save_rootpath, 'single_sbjs_' + val)

            if not os.path.exists(self.sp):
                os.makedirs(self.sp)
            if not s == 0:  # avoid fitting non-HTNet models multiple times
                if ['eegnet_hib'] in self.models:
                    self.models = ['eegnet_hilb']

            self.run_nn_models()

        print('Elapsed time: ' + str(time.time() - t_start))

        # create plots
        self.save_res_as_plt()
