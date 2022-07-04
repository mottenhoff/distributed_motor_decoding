'''
TODO: Save and plot mean covariance matrix. Maybe also time the calculation of the mean?
TODO: Because I have performance issues, take the most efficient riemannian distance/mean 
      calculations. Chevallier 2020 shows no significant difference in performance on all
      riemannian methods. So: Kullback-Leibner or Jeffreys. Jeffreys is a symmetrized 
      version of Kullback-leibner, which seems to be implemented in PyRiemann

'''
import pickle
import sys
from pathlib import Path

import pyriemann
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy import fftpack
from mne.filter import filter_data
from mne.filter import notch_filter
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# LOCAL
path = r'./libs'
sys.path.insert(0, path)
from data_quality.check_quality import QualityChecker
from loader import get_all_files, load_grasp_seeg
from locations import load_all_electrode_locations
import central_locations_list as cll

from plotting import plot_pca, print_results


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

    flagged_channels = np.append(flagged_channels, irrelevant_channels).astype(int)

    noisy_channels = np.concatenate([
        qc.flat_signal(seeg['eeg']),
        qc.excessive_line_noise(seeg['eeg'],
                                seeg['fs'],
                                freq_line=50,
                                plot=0),
        qc.abnormal_amplitude(seeg['eeg'], plot=0)
        ]).astype(int)
    
    flagged_channels = np.append(flagged_channels, noisy_channels).astype(int)

    return flagged_channels, seeg

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

def split_per_trial(seeg):
    # Extracts start and stop indices of each trial
    trials_idc = []
    labels = []
    start_label = None
    start_idx = None
    prev = None
    for curr_idx, curr in enumerate(seeg['trial_labels']):
        
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
        trial = seeg['eeg'][idc[0]:idc[1], :]
        trials += [trial[0:min_samples, :]]

    return np.array(trials), np.array(labels)

def get_classifier():
    return make_pipeline(
                pyriemann.estimation.Covariances(estimator='lwf'), # Sample covmat + shrinkage
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

def decode(train_x, train_y, test_x, test_y, n_components):
    results = []
    params = []

    clf = get_classifier()
    train_x = train_x.transpose((0, 2, 1))
    test_x = test_x.transpose((0, 2, 1))

    pca, clf, train_y_hat, test_y_hat = run_pipeline(train_x, train_y, test_x, n_components)

    results.append({
        "clf": clf,
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
        "train_y_hat": train_y_hat, 
        "test_y_hat": test_y_hat,
        "pca": pca,
        })
    print('.', end='', flush=True)

    params += [{'principle_components': n_components}]

    if type(clf) == Pipeline:
        params += [clf.named_steps]

    return results, params
    
def explore(seeg, ppt_id, session_id, name=''):
    fig, axes = plot_pca(seeg, ppt_id, session_id, name=name, normalize=True)

def evaluate(results, ppt_id, session_id, params, name):
    print_results(results, ppt_id, session_id, params, name)

def run():
   
    exp = 'cross'
    n_components = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    bands = [['beta'], ['high_gamma'], ['beta', 'high_gamma']]

    remove_motor_locs = True
    session_id = 0

    for band in bands:
        name = f"{exp}_{''.join(band)}"
        path_save = Path(f"./results/{name}")
        path_save.mkdir(parents=True, exist_ok=True)

        path = Path('./data')
        exec_ = get_all_files(path, 'xdf', keywords=['execute'])
        imag = get_all_files(path, 'xdf', keywords=['imagine'])
        ppts = list(zip([e[0] for e in exec_], [i[0] for i in imag]))
        
        all_locations = load_all_electrode_locations(path)

        for ppt in ppts:
            params = {}
            ppt_id = ppt[0].parts[1]

            # Load data
            locations = all_locations.get(ppt_id, None)

            exec_ = load_grasp_seeg(ppt[0])
            exec_['locations'] = locations

            imag  = load_grasp_seeg(ppt[1])
            imag['locations'] = locations

            if remove_motor_locs:
                removed_locs_exec, exec_ = remove_locations(exec_)
                removed_locs_imag, imag = remove_locations(imag)

            flagged_e, exec_ = clean_data(exec_)
            if flagged_e == 'invalid':
                print('{} has invalid data! Skipping...'.format(ppt_id))
                continue

            flagged_i, imag = clean_data(imag)
            if flagged_i == 'invalid':
                print('{} has invalid data! Skipping...'.format(ppt_id))
                continue
            
            channels_to_delete = np.union1d(flagged_e, flagged_i)
            print(channels_to_delete)
            with open('./desc.txt', 'a+') as f:
                print(f"{ppt_id} | {len(locations)} | {len(set([l.strip('1234567890') for l in locations if 'order' not in l]))} | {exec_['eeg'].shape[1]} | {len(channels_to_delete)} | {exec_['fs']}",
                      file=f)

            channels_to_delete_names = np.array(exec_['channel_names'])[channels_to_delete]
            exec_['eeg'] = np.delete(exec_['eeg'], channels_to_delete, axis=1)
            imag['eeg'] = np.delete(imag['eeg'], channels_to_delete, axis=1)
            exec_['channel_names'] = np.delete(exec_['channel_names'], channels_to_delete)
            imag['channel_names'] = np.delete(imag['channel_names'], channels_to_delete)

            exec_['eeg'], params_pp = preprocess(exec_['eeg'], exec_['fs'], band)
            imag['eeg'], params_pp = preprocess(imag['eeg'], imag['fs'], band)
            
            params.update({'n_channels': exec_['eeg'].shape[1],
                           'bands': params_pp,
                           'cross_task': True})

            # explore(exec_, ppt_id, session_id, name=f'exec_{band}')
            # explore(imag, ppt_id, session_id, name=f'imag_{band}')

            ch_info = {
                'original_chs': exec_['channel_names'],
                'removed_chs_name': channels_to_delete_names,
                'removed_chs_idc': channels_to_delete,
                'removed_motor_locs': removed_locs_exec
            }

            with open(f'{path_save}/ch_info_{ppt_id}_{session_id}_{exp}.pkl', 'wb') as f:
                pickle.dump(ch_info, f) 


            exec_['eeg'], exec_['trial_labels'] = split_per_trial(exec_)
            imag['eeg'], imag['trial_labels'] = split_per_trial(imag)

            exec_['trial_labels'] = np.where(exec_['trial_labels']=='0', 0, 1)
            imag['trial_labels'] = np.where(imag['trial_labels']=='0', 0, 1)

            for components in n_components:
                
                if exec_['eeg'].shape[2] < components \
                    or imag['eeg'].shape[2] < components:
                    continue

                results, learner_params = decode(exec_['eeg'], exec_['trial_labels'],
                                                imag['eeg'], imag['trial_labels'],
                                                components)

                params.update({'learner': learner_params})
                evaluate(results, ppt_id, session_id, params, name)
                
                if components == max(n_components):
                    data = {
                        'pca': results[0]['pca'],
                        'original_chs': exec_['channel_names'],
                        'removed_chs_name': channels_to_delete_names,
                        'removed_chs_idc': channels_to_delete 
                    }

                    with open(f'{path_save}/data_{ppt_id}_{exp}_{"_".join(band)}.pkl', 'wb') as f:
                        pickle.dump(data, f) 

                plt.close('all')

    print('done')


if __name__ == '__main__':
    run()
