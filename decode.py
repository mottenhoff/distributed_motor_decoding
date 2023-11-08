'''
TODO: Save and plot mean covariance matrix. Maybe also time the calculation of the mean?
TODO: Because I have performance issues, take the most efficient riemannian distance/mean 
      calculations. Chevallier 2020 shows no significant difference in performance on all
      riemannian methods. So: Kullback-Leibner or Jeffreys. Jeffreys is a symmetrized 
      version of Kullback-leibner, which seems to be implemented in PyRiemann

'''
from itertools import product
from pathlib import Path
import pickle
import sys

import pyriemann
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy import fftpack
from scipy.stats import mode
from mne.filter import filter_data
from mne.filter import notch_filter
from mne.decoding import CSP
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score



# LOCAL
path = r'./libs'
sys.path.insert(0, path)
from data_quality.check_quality import QualityChecker
from loader import get_all_files, load_grasp_seeg
from locations import load_all_electrode_locations, load_csv
import central_locations_list as cll

from plotting import plot_pca, print_results

DECODE_CSP_LDA = True

def print_experiment_metrics(ppts, locations):
    n_ppts = len(set([ppt[0].parts[1] for ppt in ppts]))
    n_contacts = [len(loc.keys())-1 for loc in locations.values()]
    n_electrodes = [len(set([l.rstrip('0123456789') for l in loc.keys()]))-1
                    for loc in locations.values()]

    unique_locs = []
    for val in locations.values():
        for v in val.values():
            if type(v) != list:
                unique_locs += [v]
    unique_locs = set(unique_locs)

    print(f'''
        n contacts: {n_contacts}
                avg: {np.mean(n_contacts)}
                min: {min(n_contacts)}
                max: {max(n_contacts)}
        n electrodes: {n_electrodes}
                avg: {np.mean(n_electrodes)}
                min: {min(n_electrodes)}
                max: {max(n_electrodes)}
        n unique locs: {len(unique_locs)}        
                ''')
        
def remove_locations(seeg):
    if not seeg['locations']:
        print('No location data available')
        return [], seeg

    to_remove = cll.import_list()

    to_remove_names = [contact for contact, label in seeg['locations'].items()
                        if label in to_remove]
    to_remove_idc = np.array([i for i, name in enumerate(seeg['channel_names'])\
                                if name in to_remove_names])

    for contact in to_remove_names:
        print(f'Removing {contact}: {seeg["locations"][contact]}')

    with open('removed_locs.txt', 'a+') as f:
        print(f"{seeg['subject']}: {len(to_remove_names)} \t {[seeg['locations'][c] for c in to_remove_names]}", file=f)

    return to_remove_idc, seeg

def clean_data(seeg):
    flagged_channels = np.array([])

    qc = QualityChecker()

    if any(qc.consistent_timestamps(seeg['eeg_ts'], seeg['fs'])):
        return 'invalid', seeg
  
    irrelevant_channels = np.concatenate([
        qc.get_marker_channels(seeg['eeg'], 
                               channel_names=seeg['channel_names']),
        qc.get_ekg_channel(seeg['eeg'], 
                           channel_names=seeg['channel_names']),
        qc.get_disconnected_channels(seeg['eeg'], 
                                     channel_names=seeg['channel_names'])
        ]).astype(int)

    noisy_channels = np.concatenate([
        qc.flat_signal(seeg['eeg']),
        qc.excessive_line_noise(seeg['eeg'],
                                seeg['fs'],
                                freq_line=50,
                                plot=0),
        qc.abnormal_amplitude(seeg['eeg'], plot=0)
        ]).astype(int)
    
    print(f'Removed channels due to noise: {noisy_channels.size}')

    flagged_channels = np.union1d(irrelevant_channels, noisy_channels)

    return flagged_channels.astype(int), seeg

def preprocess(data, fs, bands):
    params = []
    hilbert3 = lambda x: scipy.signal.hilbert(x, fftpack.next_fast_len(len(x)), 
                                              axis=0)[:len(x)]

    # Expects a [samples x channels] matrix
    data = scipy.signal.detrend(data, axis=0)
    data -= data.mean(axis=0)
    data = notch_filter(data.T, fs, np.arange(50, 201, 50)).T

    filtered_bands = []
    for band in bands:

        if band == 'beta':
            filtered_bands += [filter_data(data.T,
                            sfreq=fs,
                            l_freq=12,
                            h_freq=30).T]
            params += ['beta_12_30']

        if band == 'high_gamma':
            filtered_bands += [filter_data(data.T,
                        sfreq=fs,
                        l_freq=55,
                        h_freq=90).T]
            params += ['high_gamma_55_90']

    # Convert to power
    bands = np.concatenate(filtered_bands, axis=1) 
    bands = abs(hilbert3(bands))
    return bands, params

def split_per_trial(eeg, trial_labels):
    # Extracts start and stop indices of each trial
    trials_idc = []
    labels = []
    start_label = None
    start_idx = None
    prev = None
    for curr_idx, curr in enumerate(trial_labels):
        
        if start_label == None:
            start_label = curr
            start_index = curr_idx

        if curr != start_label and prev == start_label:
            # Detected change of labels
            current_idc = [start_idx, curr_idx]
            if None not in current_idc:
                trials_idc += [current_idc]
                labels += [start_label]
            start_idx = curr_idx
            start_label = curr
        
        prev = curr

    # Throw them in a 3d matrix
    window_sizes = [np.diff(idc) for idc in trials_idc]
    min_samples = min(window_sizes)[0]
    max_samples = max(window_sizes)[0]
    print('Reduced trial size to {:d} samples. (max seen window size: {:d})'\
           .format(min_samples, max_samples))

    trials = []
    for idc in trials_idc:
        trial = eeg[idc[0]:idc[1], :]
        trials += [trial[0:min_samples, :]]

    return np.array(trials), np.array(labels)

def get_train_test(x, y, fold, folds, continuous=False):
    y = np.expand_dims(y, axis=1) if len(y.shape) == 1 else y
    y = mode(y, axis=1)[0] if continuous else y

    samples_per_fold = int(x.shape[0]/folds)
    test_idc = np.arange(fold*samples_per_fold,
                         fold*samples_per_fold + samples_per_fold)

    mask = np.ones(x.shape[0], bool)
    mask[test_idc] = False

    test_x = x[~mask, :, :]
    test_y = y[~mask, :]
    train_x = x[mask, :, :]
    train_y = y[mask, :]

    return train_x, train_y.ravel(), test_x, test_y.ravel()

def get_classifier():
    
    return make_pipeline(
                pyriemann.estimation.Covariances(estimator='lwf'),
                pyriemann.classification.MDM(metric='kullback_sym')
        )

def run_pipeline(train_x, train_y, test_x, n_components):

    unsplit = lambda arr: np.vstack(arr.transpose(0, 2, 1))
    split = lambda arr, dim_size: np.array(np.vsplit(arr, dim_size)).transpose(0, 2, 1)

    dim_trials_train, dim_trials_test = train_x.shape[0], test_x.shape[0]

    train_x = unsplit(train_x)
    test_x = unsplit(test_x)

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    pca = PCA(n_components=n_components)
    pca = pca.fit(train_x)

    train_pcs = split(pca.transform(train_x), dim_trials_train)
    test_pcs = split(pca.transform(test_x), dim_trials_test)

    clf = get_classifier()
    clf.fit(train_pcs, train_y)

    train_y_hat = clf.predict_proba(train_pcs)
    test_y_hat = clf.predict_proba(test_pcs)
    
    return pca, clf, train_y_hat, test_y_hat

def run_pipeline_csp_lda(train_x, train_y, test_x, n_components):

    unsplit = lambda arr: np.vstack(arr.transpose(0, 2, 1))
    split = lambda arr, dim_size: np.array(np.vsplit(arr, dim_size)).transpose(0, 2, 1)

    dim_trials_train, dim_trials_test = train_x.shape[0], test_x.shape[0]

    csp  = CSP(n_components=n_components, reg='ledoit_wolf')
    csp.fit(train_x, train_y)

    train_x_csp, test_x_csp = csp.transform(train_x), csp.transform(test_x)

    lda = LinearDiscriminantAnalysis()
    lda.fit(train_x_csp, train_y)
    
    train_y_hat = lda.predict_proba(train_x_csp)
    test_y_hat = lda.predict_proba(test_x_csp)
    
    return csp, lda, train_y_hat, test_y_hat

def decode(x, y, n_components):

    results = []
    params = []

    folds = 10

    clf = get_classifier()
    x = x.transpose((0, 2, 1))

    for fold in range(folds):
        train_x, train_y, test_x, test_y = get_train_test(x, y, fold, folds)

        if DECODE_CSP_LDA:

            csp, lda, train_y_hat, test_y_hat = run_pipeline_csp_lda(train_x, train_y, test_x, n_components)
            print(f'{fold} | Train: {roc_auc_score(train_y, train_y_hat[:, 1]):5.2f} | Test: {roc_auc_score(test_y, test_y_hat[:, 1]):5.2f}')
            
            results.append({
                "pca": csp, # Keeping the name the same for simplicity. Key should be csp 
                "clf": lda,
                "train_x": train_x,
                "train_y": train_y,
                "test_x": test_x,
                "test_y": test_y,
                "train_y_hat": train_y_hat, 
                "test_y_hat": test_y_hat,
                })
        else:

            pca, clf, train_y_hat, test_y_hat = run_pipeline(train_x, train_y, test_x, n_components)

            results.append({
                "pca": pca,
                "clf": clf,
                "train_x": train_x,
                "train_y": train_y,
                "test_x": test_x,
                "test_y": test_y,
                "train_y_hat": train_y_hat, 
                "test_y_hat": test_y_hat,
                })
            print('.', end='', flush=True)

    params += [{'principle_components': n_components}]

    if type(clf) == Pipeline:
        params += [clf.named_steps]

    return results, params

def explore(seeg, ppt_id, session_id, name=''):
    fig, axes = plot_pca(seeg, ppt_id, session_id, name=name, normalize=True)

def evaluate(results, ppt_id, session_id, params, name):
    tmp = print_results(results, ppt_id, session_id, params, name)


def run(ppt, exp, band):
    n_components = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    remove_motor_locs = True
    session_id = 0
    ppt_id = ppt


    data_path = Path('./data')
    filename = data_path/f'{ppt}'/f'{"execute" if exp == "grasp" else "imagine"}.xdf'

    name = f"{exp}_{''.join(band)}"
    path_save = Path(f"./results/{name}")
    path_save.mkdir(parents=True, exist_ok=True)


    # print_experiment_metrics(ppts, locations)
    results = {}
    params = {}

    # Load data
    seeg = load_grasp_seeg(filename)
    seeg['locations'] = load_csv(data_path/f'{ppt}'/'electrode_locations.csv')
    original_chs = seeg['channel_names']

    # Remove locations around the central sulcus
    if remove_motor_locs:
        removed_locs, seeg = remove_locations(seeg)
    else:
        removed_locs = np.empty(0)

    # Check channels for noise
    flagged_noise, seeg = clean_data(seeg)
    if type(flagged_noise)==str and flagged_noise == 'invalid':
        print('{} has invalid data! Skipping...'.format(ppt_id))
        return

    # Remove the channels with excessive noise
    flagged = np.union1d(removed_locs, flagged_noise).astype(int)
    removed_chs_name = np.array(seeg['channel_names'])[flagged]
    removed_chs_idc = flagged

    seeg['eeg'] = np.delete(seeg['eeg'], flagged, axis=1)
    seeg['channel_names'] = np.delete(seeg['channel_names'], flagged)

    # Preprocess the data, including filter + hilbert for beta, high-gamma and both
    seeg['eeg'], params_pp = preprocess(seeg['eeg'], seeg['fs'], band)
    params.update({'n_channels': seeg['eeg'].shape[1],
                    'bands': params_pp})

    # Split into trials
    seeg['eeg'], seeg['trial_labels'] = split_per_trial(seeg['eeg'], seeg['trial_labels'])

    # Only train for move vs rest
    seeg['trial_labels'] = np.where(seeg['trial_labels']=='0', 0, 1)

    # Train and evaluate decoder for each number of component
    for components in n_components:
        if seeg['eeg'].shape[2] < components:
            continue

        results, learner_params = decode(seeg['eeg'], seeg['trial_labels'], n_components=components)

        params.update({'learner': learner_params})
        # evaluate(results, ppt_id, session_id, params, name)
        print('')
        plt.close('all')

    data = {
        'original_chs': original_chs,
        'removed_chs_name': removed_chs_name,
        'removed_chs_idc': removed_chs_idc }

    with open(f'{path_save}/data_{ppt_id}_{exp}_{"_".join(band)}.pkl', 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    run()
