import pickle
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


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

def print_results(results, ppt_id, session_id, params, name, save=True):
    with open('./results/{}/{}_{}.txt'.format(name, ppt_id, session_id), 'a+') as f:
        print('{:<5s}| N_channels: {} | {} | learner: {}'\
                .format(datetime.now().strftime("%d-%m-%Y %H:%M"),
                        params['n_channels'], params['bands'],
                        params['learner']), file=f)
        print("{:<5s} | {:<12s}  {:<12s} | {:<12s}  {:<12s}"\
              .format("AUC", "Move vs Rest", "", "Left vs Right", ""),
              file=f)
        print("{:<5s} | {:<12s} | {:<12s} | {:<12s} | {:<12s}"\
              .format("", "TRAIN", "TEST", "TRAIN", "TEST"),
              file=f)
        
        scores = []
        for fold, result in enumerate(results):
            mvr_train_y = np.where(result['train_y']==0, 0, 1)
            mvr_test_y = np.where(result['test_y']==0, 0, 1)
            mvr_train_y_hat = np.vstack([result['train_y_hat'][:, 0],
                                         result['train_y_hat'][:, 1:].sum(axis=1)]).T
            mvr_test_y_hat = np.vstack([result['test_y_hat'][:, 0],
                                        result['test_y_hat'][:, 1:].sum(axis=1)]).T
            mvr_train_auc = metrics.roc_auc_score(mvr_train_y, mvr_train_y_hat[:, 1])
            try:
                mvr_test_auc = metrics.roc_auc_score(mvr_test_y, mvr_test_y_hat[:, 1])
            except Exception:
                print('No AUC, setting mvr_test_auc to -1')
                mvr_test_auc = -1

            print("{:<5d} | {:<12.2f} | {:<12.2f}"\
                   .format(fold, mvr_train_auc, mvr_test_auc),
                  file=f)

            scores.append({
                # Move vs Rest
                'mvr_train_y': mvr_train_y, 'mvr_test_y': mvr_test_y,'mvr_train_y_hat': mvr_train_y_hat,
                'mvr_test_y_hat': mvr_test_y_hat, 'mvr_train_auc': mvr_train_auc, 'mvr_test_auc': mvr_test_auc,
            })

        print("{:<5s} | {:<12.2f} | {:<12.2f}"\
              .format("MEAN", 
                    np.mean([score['mvr_train_auc'] for score in scores]),
                    np.mean([score['mvr_test_auc'] for score in scores \
                                if score['mvr_train_auc'] != -1])), 
              file=f)
              
        print('\n', file=f)
    
    return scores

def plot_pca(seeg, ppt_id, session_id, name=''):
    seeg = seeg.copy()

    path = f'./figures/pca/{ppt_id}_{session_id}'
    Path(path).mkdir(parents=True, exist_ok=True)

    colors = ['tab:blue', 'tab:orange', 'tab:green']
    trans_dict = {'0': 'Rest',
                  'Links': 'Left',
                  'Rechts': 'Right'}
    
    seeg['eeg'] = scale(seeg['eeg'], axis=0)

    n_components = 50
    pca = PCA(n_components=n_components).fit(seeg['eeg'])

    # Transform data
    seeg['eeg'] = pca.transform(seeg['eeg'])
    seeg, labels = split_per_trial(seeg)
    seeg = seeg - seeg[:, :1, :]  # Normalize
        
    # Complete fig
    fig = plt.figure()
    for j, subplot in enumerate([221, 222, 223, 224]):
        pc_dims = list(range(0+j, 3+j))
        ax = fig.add_subplot(subplot, projection='3d')
        for i, label in enumerate(trans_dict.keys()):
            class_idc = np.where(labels==label)[0]
            ax.scatter(seeg[class_idc, :, pc_dims[0]].mean(axis=0),
                       seeg[class_idc, :, pc_dims[1]].mean(axis=0),
                       seeg[class_idc, :, pc_dims[2]].mean(axis=0), 
                       label=trans_dict[label],
                       color=colors[i],
                       s=3)
            # print(seeg[class_idc, :, pc_dims[0]].mean(axis=0).shape)
        ax.set_xlabel('PCA {}'.format(pc_dims[0]))
        ax.set_ylabel('PCA {}'.format(pc_dims[1]))
        ax.set_ylabel('PCA {}'.format(pc_dims[2]))
    plt.legend()
    plt.tight_layout()
    fig.savefig(f'{path}/pca_transformed_{name}_normalized.png')

    results = {'pca': pca,
               'transformed_seeg': seeg,
               'labels': labels,
               'best_features': best_features}
    
    with open(f'{path}/data_{name}_normalized.pkl', 'wb') as f_obj:
        pickle.dump(results, f_obj)

    return None, None

