import os
from datetime import datetime
# from itertools import combinations
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
# from matplotlib.colors import DivergingNorm
import numpy as np
# import seaborn as sns
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
# import pyriemann
# from pyriemann.embedding import Embedding
# from scipy.stats import ttest_ind

# import sys

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

# def multiclass_auc(y, y_hat, classes, protocol='1vr'):
#     '''
#     protocol options:
#         1vr: Computes the problem as a 1 versus rest
#                 generating n_class aucs
#         1v1: Computes the AUC of all possible pairwise
#                 combinations, generating 
#                 n_classes*(nclasses - 1) / 2  aucs  
#     TODO: Create option to compare specific labels  
#             Specifically left vs right                      
#     '''
#     if protocol not in ['1vr', '1v1']:
#         print("Multiclass_auc: Unknown protocol: {}. Using 1vr."\
#                 .format(protocol))
            
#     aucs = {}
#     if protocol in ['1vr', 'all']:
#         for i, class_ in enumerate(classes):
#             auc_1vr = []
#             name = '{} vs rest'.format(class_)
#             y_binary = [1 if label == class_ else 0 for label in y]
#             auc = metrics.roc_auc_score(y_binary, y_hat[:, i])
#             auc_1vr += [auc]
#             aucs[name] = auc
#         aucs['avg 1vr'] = sum(aucs.values()) / len(aucs.values())

#     if protocol in ['1v1', 'all']:
#         for combination in combinations(classes, 2):
#             auc_1v1 = []
#             name = '{} vs {}'.format(combination[0], combination[1])
#             y_binary = np.array([0 if label==combination[0] else \
#                                     1 if label==combination[1] else -1 \
#                                     for label in y])
#             trial_idc = np.where(y_binary!=-1)[0]

#             i = np.where(classes==combination[1])[0]
#             if len(np.unique(y_binary[trial_idc]))<=1:
#                 print("\nWARNING: Only a single class in this fold. AUC is not defined. Skipping...")
#                 # TODO: calculate a score anyway
#             else:
#                 auc = metrics.roc_auc_score(y_binary[trial_idc], y_hat[trial_idc, i])
#                 auc_1v1 += [auc]
#                 aucs[name] = auc
#         aucs['Avg 1v1'] = sum(aucs.values()) / len(aucs.values())

#     # Move vs Rest
#     y_binary = np.array([0 if label in ['Links', 'Rechts'] else 1 \
#                         for label in y])
#     auc = metrics.roc_auc_score(y_binary, y_hat[:, i])
#     aucs['Move vs Rest'] = auc
        
#     # TODO: Double check if this labelling is correct
#     # Links = 0
#     # Rechts = 1
#     # Rest = 2
#     return aucs

# def auc(train_y, train_y_hat, test_y, test_y_hat):
#     # Move vs Rest
#     mvr_train_y = np.where(train_y==0, 0, 1)
#     mvr_test_y = np.where(test_y==0, 0, 1)
#     mvr_train_y_hat = np.vstack([train_y_hat[:, 0],
#                                     train_y_hat[:, 1:].sum(axis=1)]).T
#     mvr_test_y_hat = np.vstack([test_y_hat[:, 0],
#                                 test_y_hat[:, 1:].sum(axis=1)]).T
#     mvr_train_auc = metrics.roc_auc_score(mvr_train_y, mvr_train_y_hat[:, 1])
#     try:
#         mvr_test_auc = metrics.roc_auc_score(mvr_test_y, mvr_test_y_hat[:, 1])
#     except Exception:
#         print('No AUC, setting mvr_test_auc to -1')
#         mvr_test_auc = -1

#     # Left vs Right
#     train_mask = np.where(train_y != 0)[0]
#     test_mask = np.where(test_y != 0)[0]
#     lvr_train_y = train_y[train_mask]
#     lvr_test_y = test_y[test_mask]
#     lvr_train_y_hat = train_y_hat[train_mask, 1:]
#     lvr_test_y_hat = test_y_hat[test_mask, 1:]
#     lvr_train_auc = metrics.roc_auc_score(lvr_train_y, lvr_train_y_hat[:, 1])
#     try:
#         lvr_test_auc = metrics.roc_auc_score(lvr_test_y, lvr_test_y_hat[:, 1])
#     except Exception:
#         print('No AUC, setting lvr_test_auc to -1')
#         lvr_test_auc = -1

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

# def plot_predictions(results, ppt_id, session_id, params, save=True):
#     fig, axes = plt.subplots(nrows=len(results), ncols=2,
#                              sharey=True,
#                              gridspec_kw={'width_ratios':[3,1]}, 
#                              figsize=(16, 16))
#     fig.suptitle('Predictions per trial')

#     hline_alpha = 0.1
#     hline_color = 'k'
#     for i, result in enumerate(results):
#         axes[i, 0].axhline(0, alpha=hline_alpha, color=hline_color)
#         axes[i, 0].axhline(1, alpha=hline_alpha, color=hline_color)
#         axes[i, 0].axhline(2, alpha=hline_alpha, color=hline_color)
#         mask = np.where(result['train_y']==np.argmax(result['train_y_hat'], axis=1), 'tab:green', 'tab:red')
#         axes[i, 0].scatter(range(len(result['train_y'])), result['train_y']-0.1, marker='s', color='darkblue', label='True')
#         axes[i, 0].scatter(range(len(result['train_y_hat'])), np.argmax(result['train_y_hat'], axis=1)+0.1, marker='s', c=mask, label='Pred')
#         axes[i, 0].set_yticks([0, 1, 2])
#         axes[i, 0].set_yticklabels(['Rest', 'Left', 'Right'])

#         axes[i, 1].axhline(0, alpha=hline_alpha, color=hline_color)
#         axes[i, 1].axhline(1, alpha=hline_alpha, color=hline_color)
#         axes[i, 1].axhline(2, alpha=hline_alpha, color=hline_color)
#         mask = np.where(result['test_y']==np.argmax(result['test_y_hat'], axis=1), 'tab:green', 'tab:red')
#         axes[i, 1].scatter(range(len(result['test_y'])), result['test_y']-0.1, marker='s', color='darkblue', label='True')
#         axes[i, 1].scatter(range(len(result['test_y_hat'])), np.argmax(result['test_y_hat'], axis=1)+0.1, marker='s', c=mask, label='Pred')
        
#         axes[i, 1].set_ylim(-0.5, 2.5)
#     axes[0, 0].set_title('TRAIN')
#     axes[0, 1].set_title('TEST')
#     axes[0, 1].legend(bbox_to_anchor=(1, 1))

#     if save:
#         fig.savefig('./figures/{}_{}_predictions_per_trial.svg'.format(ppt_id, session_id))

#     return fig, axes

# def plot_spectral_embeddings_3d(results, ppt_id, session_id, save=True):

#     def scatter3d(ax, emb, y, colors):
#         # TODO: Color points by correct/incorrect
#         for label in set(y):
#             idx = np.where(y==label)[0]
#             ax.scatter(emb[idx, 0], emb[idx, 1], emb[idx, 2],
#                        s=24, color=colors[label], label=label)
#         return ax

#     def decorate_ax(ax):
#         ax.set_xticks([-1, -.5, 0, .5, 1])
#         ax.set_yticks([-1, -.5, 0, .5, 1])
#         ax.set_zticks([-1, -.5, 0, .5, 1])

#         ax.set_xlabel(r'$\varphi_1$', fontsize=16)
#         ax.set_ylabel(r'$\varphi_2$', fontsize=16)
#         ax.set_zlabel(r'$\varphi_3$', fontsize=16)
#         return ax

#     colors = dict(zip([0, 1, 2], ['tab:blue', 'tab:red', 'tab:green']))

#     fig = plt.figure(figsize=plt.figaspect(0.5))
#     ax_tr = fig.add_subplot(1, 2, 1, projection='3d')
#     ax_te = fig.add_subplot(1, 2, 2, projection='3d')

#     fig.suptitle('Spectral embedding', fontsize=16)
#     for result in results:
#         result['clf'][0].estimator = 'corr'
#         cov_tr = result['clf'][:-1].transform(result['train_x'])
#         cov_te = result['clf'][:-1].transform(result['test_x'])
#         emb_tr = Embedding(n_components=3, metric='kullback_sym').fit_transform(cov_tr)
#         emb_te = Embedding(n_components=3, metric='kullback_sym').fit_transform(cov_te)

#         ax_tr = scatter3d(ax_tr, emb_tr, result['train_y'], colors)
#         ax_te = scatter3d(ax_te, emb_te, result['test_y'], colors)

#     ax_tr = decorate_ax(ax_tr)
#     ax_te = decorate_ax(ax_te)

#     ax_tr.set_title('Train')
#     ax_te.set_title('Test')
#     ax_te.legend(labels=['Rest', 'Left', 'Right'], bbox_to_anchor=[1, 1])
#     if save:
#         fig.savefig('./figures/{}_{}_spectral_embedding_3d.svg'.format(ppt_id, session_id))
#     axes = [ax_te, ax_te]
#     # plt.show()
#     return fig, axes

# def plot_spectral_embeddings_2d(results, ppt_id, session_id, save=True):
#     from time import time
#     colors = dict(zip([0, 1, 2], ['tab:blue', 'tab:red', 'tab:green']))

#     fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
#     fig.suptitle('Spectral embedding', fontsize=16)
#     for i, result in enumerate(results):
#         result['clf'][0].estimator = 'corr'

#         start = time()
#         cov_tr = result['clf'][:-1].transform(result['train_x'])
#         cov_te = result['clf'][:-1].transform(result['test_x'])
        
#         print('{:d} - COVS: {:.5f}'.format(i, time()-start))
#         start = time()
#         emb_tr = Embedding(n_components=2, metric='kullback_sym').fit_transform(cov_tr)
#         emb_te = Embedding(n_components=2, metric='kullback_sym').fit_transform(cov_te)
#         print('{:d} - EMBS: {:.5f}'.format(i, time()-start))

#         # TODO: Doesnt seem to fit with the AUC?
#         # correct = np.where(result['test_y']==np.argmax(result['test_y_hat'], axis=1))[0]
#         # incorrect = np.where(result['test_y']!=np.argmax(result['test_y_hat'], axis=1))[0]

#         for label in set(result['train_y']):
#             idx = np.where(result['train_y']==label)[0]
#             axs[0].scatter(emb_tr[idx, 0], emb_tr[idx, 1], 
#                            s=24, color=colors[label], label=label)  # TODO: CHECK IF THIS IS CORRECT

#         for label in set(result['test_y']):
#             idx = np.where(result['test_y']==label)[0]
#             axs[1].scatter(emb_te[idx, 0], emb_te[idx, 1],
#                            s=24, color=colors[label], label=label)
#     for i in [0, 1]:
#         axs[i].set_xticks([-1.0, -0.5, 0.0, +0.5, 1.0])
#         axs[i].set_yticks([-1.0, -0.5, 0.00, +0.5, 1.0])

#     axs[0].set_aspect('equal', 'box')
#     axs[1].set_aspect('equal', 'box')
#     axs[0].set_xlabel(r'$\varphi_1$', fontsize=16)
#     axs[0].set_ylabel(r'$\varphi_2$', fontsize=16)
#     axs[0].set_title('Train')
#     axs[1].set_title('Test')
#     axs[1].legend(labels=['Rest', 'Left', 'Right'], bbox_to_anchor=[1, 1])

#     if save:
#         start = time()
#         fig.savefig('./figures/{}_{}_spectral_embedding_2d.svg'.format(ppt_id, session_id))
#         print('SAVE {:.5f}'.format(time()-start))
#     return fig, axs

# def plot_spectral_embeddings(results, ppt_id, session_id, type='2d', save=True):
#     fig = None
#     axes = None
#     if type=='2d':
#         fig, axes = plot_spectral_embeddings_2d(results, ppt_id, session_id, save=save)
#     elif type=='3d':
#         fig, axes = plot_spectral_embeddings_3d(results, ppt_id, session_id, save=save)
#     else:
#         print('WARNING: Invalid type for spectral embedding! Skipping...')
#     return fig, axes

# def plot_train_test_fit_bal_acc(results, ppt_id, session_id, save=True):

#     fig, axes = plt.subplots(nrows=len(results), ncols=2, figsize=(8, 16))
#     for i, result in enumerate(results):
#         if i==0:
#             axes[0, 0].set_title('TRAINING FIT')
#             axes[0, 1].set_title('TEST FIT')

#         # axes[i, 0].set_title('Bal Acc {:.3f}'.format(metrics.balanced_accuracy_score(result['train_y'], result['train_y_hat'])))
#         sns.heatmap(metrics.confusion_matrix(result['train_y'], result['train_y_hat']), annot=True, cbar=False, ax=axes[i, 0])

#         # axes[i, 1].set_title('Bal Acc: {:.3f}'.format(metrics.balanced_accuracy_score(result['test_y'], result['test_y_hat'])))
#         sns.heatmap(metrics.confusion_matrix(result['test_y'], result['test_y_hat']), annot=True, cbar=False, ax=axes[i, 1])
#         plt.tight_layout()

#     if save:
#         fig.savefig('./figures/{}_{}_train_test_fit_bal_acc.svg'.format(ppt_id, session_id))

#     return fig, axes

# def plot_train_test_fit_roc_auc(results, ppt_id, session_id, save=True):

#     fig, axes = plt.subplots(nrows=len(results), ncols=2, figsize=(8, 16))
#     for i, result in enumerate(results):
#         if i==0:
#             axes[0, 0].set_title('TRAINING FIT')
#             axes[0, 1].set_title('TEST FIT')

#         try:
#             axes[i, 0].set_title('AUC {:.3f}'.format(metrics.roc_auc_score(result['train_y'], 
#                                                                 result['train_y_hat'],
#                                                                 multi_class='ovo')))
#             axes[i, 1].set_title('AUC {:.3f}'.format(metrics.roc_auc_score(result['test_y'], 
#                                                                 result['test_y_hat'],
#                                                                 multi_class='ovo')))
#         except Exception:
#             pass

#         classes = np.unique(result['train_y']) 
#         for class_ in classes:
#             y_bin = np.where(result['train_y']==class_, 1, 0)
#             y_hat_bin = result['train_y_hat'][:, class_]
#             fpr, tpr, thresholds = metrics.roc_curve(y_bin, y_hat_bin)

#             axes[i, 0].plot(fpr, tpr, label="{} vs rest".format(class_))
#             axes[i, 0].plot([0, 1], [0, 1])

#             y_bin = np.where(result['test_y']==class_, 1, 0)
#             y_hat_bin = result['test_y_hat'][:, class_]
#             fpr, tpr, thresholds = metrics.roc_curve(y_bin, y_hat_bin)

#             axes[i, 1].plot(fpr, tpr, label="{} vs rest".format(class_))
#             axes[i, 1].plot([0, 1], [0, 1])


#     axes[i, 0].set_xlabel('FPR')
#     axes[i, 0].set_ylabel('TPR')
#     axes[0, 1].legend()

#     plt.tight_layout()

#     if save:
#         fig.savefig('./figures/{}_{}_train_test_fit_roc_auc.svg'.format(ppt_id, session_id))

#     return fig, axes

# def plot_covariance_mean(data, ppt_id, session_id, channel_locations=None, save=True):

#     show_corr = False  # NOTE: Still not sure if plotting corrs makes sense... 

#     def annot(ax, data):
#         for i in range(data.shape[0]):
#             for j in range(data.shape[1]):
#                 ax.text(j-.5, i-.5, "{:.1f}".format(data[i, j], ha='center', va='center'))

#     tmp = data.copy()
#     seeg = data['eeg']
#     labels = data['trial_labels']

#     seeg = seeg.transpose((0, 2, 1))

#     cov_mats = pyriemann.estimation.Covariances(estimator='lwf').transform(seeg)

#     trans_dict = {'0': 'Rest',
#                   'Links': 'Left',
#                   'Rechts': 'Right'}
#     # Calculate Centroids
#     centroids = []
#     cov_labels = []
#     for class_ in ['0', 'Links', 'Rechts']:
#         cov_label = cov_mats[np.where(labels==class_)[0], :, :]
#         cov_centroid = pyriemann.utils.mean.mean_covariance(cov_label, metric='kullback_sym')

#         if show_corr:
#         # Cov to correlation
#             varxy = np.sqrt(np.diag(cov_centroid))
#             varxy_outer = np.outer(varxy, varxy)
#             corr = cov_centroid / varxy_outer
#             corr[cov_centroid==0] = 0
#             cov_centroid = corr

#         cov_labels += [cov_label]
#         centroids += [cov_centroid]

#     # Plot Centroids
#     vmin = min([c.min() for c in centroids])
#     vmin = -1 if vmin >= 0 else vmin
#     vmax = max([c.max() for c in centroids])
#     vmax = 1 if vmax <= 0 else vmax
#     fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 16))
#     for i in range(3):
#         im = axes[0, i].imshow(centroids[i], cmap='seismic',
#                                norm=DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax))
#         # annot(axes[0, i], centroids[i])
#         axes[0, i].set_title(list(trans_dict.values())[i])
#     fig.colorbar(im, ax=axes[0, -1], orientation='vertical')

#     # Calculate differences
#     lvre = centroids[1] - centroids[0]
#     rvre = centroids[2] - centroids[0]
#     lvr = centroids[2] - centroids[1]
#     vmin = min([lvre.min(), rvre.min(), lvr.min()])
#     vmax = max([lvre.max(), rvre.max(), lvr.max()])

#     # Plot differences
#     axes[1, 0].imshow(lvre, cmap='seismic', norm=DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax))
#     axes[1, 0].set_title('Left - Rest | d={:3.3f}'.format(pyriemann.utils.distance.distance(centroids[1], centroids[0], metric='kullback_sym')))

#     axes[1, 1].imshow(rvre, cmap='seismic', norm=DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax))
#     axes[1, 1].set_title('Right - Rest | d={:3.3f}'.format(pyriemann.utils.distance.distance(centroids[2], centroids[0], metric='kullback_sym')))
    
#     im = axes[1, 2].imshow(lvr, cmap='seismic', norm=DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax))
#     axes[1, 2].set_title('Right - Left | d={:3.3f}'.format(pyriemann.utils.distance.distance(centroids[2], centroids[1], metric='kullback_sym')))
#     fig.colorbar(im, ax=axes[1, -1], orientation='vertical')
    
#     # Calculate significant differences
#     p_sig = 0.05
#     p_mask = lambda a, b, p: np.where(ttest_ind(a, b)[1] < p/a[0,:,:].size, 1, 0)
#     sig_lvre = p_mask(cov_labels[0], cov_labels[1], p_sig)
#     sig_rvre = p_mask(cov_labels[0], cov_labels[2], p_sig)
#     sig_lvr = p_mask(cov_labels[1], cov_labels[2], p_sig)
    
#     axes[2, 0].imshow(sig_lvre, cmap='gray')
#     axes[2, 0].set_title('Left - Rest')
#     axes[2, 1].imshow(sig_rvre, cmap='gray')
#     axes[2, 1].set_title('Right - Rest')
#     im = axes[2, 2].imshow(sig_lvr, cmap='gray')
#     axes[2, 2].set_title('Right - Left')
    
#     fig.colorbar(im, ax=axes[2, -1], orientation='vertical')

#     # Decorate
#     axes[1, 0].annotate('Values calculates as "d = b - a", thus high values mean that there is higher covariance in b than a.', 
#                          xy=(0, 0), xycoords='figure fraction', xytext=(20, 20), textcoords='offset points', ha='left', va='bottom')
    
#     if show_corr:
#         fig.suptitle('''Correlation matrices
#                      Top row: Centroid covariance matrix per class
#                      Middle row: Differences between centroids
#                      Bottom row: Significant differences between centroids''')
#     else:
#         fig.suptitle('''Covariance matrices
#                      Top row: Centroid covariance matrix per class
#                      Middle row: Differences between centroids
#                      Bottom row: Significant differences between centroids''')
    

#     # TODO: Might want to normalize (== Correlation, read link) 
#     # https://support.minitab.com/en-us/minitab-express/1/help-and-how-to/modeling-statistics/regression/how-to/covariance/interpret-the-results/

#     # if save:
#     #     fig.savefig('./figures/{}_{}_covariance_centroids.svg'.format(ppt_id, session_id))

#     bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']
#     contacts = [channel_locations.get(ch, ch) for ch in data['channel_names']]
#     locs = ['{:s}_{:s}'.format(band, contact) for band in bands for contact in contacts]

#     names = ['Left_v_rest', 'Right_v_rest', 'Left_v_right']
#     for i, sigs in enumerate([sig_lvre, sig_rvre, sig_lvr]):
#         x, y = np.where(sigs)

#         coords = []
#         for coord in list(zip(x, y)):
#             coord = sorted(coord)
#             if coord not in coords:
#                 coords += [coord]
        
#         coords = [[[locs[c] for c in coord], coord] for coord in coords]
#         with open('./results/{:s}_{:d}_significant_locs_{:s}.txt'\
#                     .format(ppt_id, session_id, names[i]), 'w+') as f:
#             for coord in coords:
#                 print(coord, file=f)

#     return fig, axes

# def plot_pre_post_covariance(data, ppt_id, session_id, channel_locations=None, save=True):

#     show_corr = False  # NOTE: Still not sure if plotting corrs makes sense... 

#     def annot(ax, data):
#         for i in range(data.shape[0]):
#             for j in range(data.shape[1]):
#                 ax.text(j-.5, i-.5, "{:.1f}".format(data[i, j], ha='center', va='center'))

#     tmp = data.copy()
#     seeg = data['eeg']
#     labels = data['trial_labels']

#     seeg = seeg.transpose((0, 2, 1))

#     cov_mats = pyriemann.estimation.Covariances(estimator='lwf').transform(seeg)

#     trans_dict = {'0': 'Rest',
#                   'Links': 'Left',
#                   'Rechts': 'Right'}
#     # Calculate Centroids
#     centroids = []
#     cov_labels = []
#     for class_ in ['0', 'Links', 'Rechts']:
#         cov_label = cov_mats[np.where(labels==class_)[0], :, :]
#         cov_centroid = pyriemann.utils.mean.mean_covariance(cov_label, metric='kullback_sym')

#         if show_corr:
#         # Cov to correlation
#             varxy = np.sqrt(np.diag(cov_centroid))
#             varxy_outer = np.outer(varxy, varxy)
#             corr = cov_centroid / varxy_outer
#             corr[cov_centroid==0] = 0
#             cov_centroid = corr

#         cov_labels += [cov_label]
#         centroids += [cov_centroid]

#     # Plot Centroids
#     vmin = min([c.min() for c in centroids])
#     vmin = -1 if vmin >= 0 else vmin
#     vmax = max([c.max() for c in centroids])
#     vmax = 1 if vmax <= 0 else vmax
#     fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 16))
#     for i in range(3):
#         im = axes[0, i].imshow(centroids[i], cmap='seismic',
#                                norm=DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax))
#         # annot(axes[0, i], centroids[i])
#         axes[0, i].set_title(list(trans_dict.values())[i])
#     fig.colorbar(im, ax=axes[0, -1], orientation='vertical')

#     # Whitening
#     wn = pyriemann.preprocessing.Whitening(metric='kullback_sym',
#                                       dim_red={'n_components': 50},
#                                       wse=True)
#     cov_mats_wn = wn.fit_transform(cov_mats)

#     centroids_wn = []
#     cov_labels_wn = []
#     for class_ in ['0', 'Links', 'Rechts']:
#         cov_label = cov_mats_wn[np.where(labels==class_)[0], :, :]
#         cov_centroid = pyriemann.utils.mean.mean_covariance(cov_label, metric='kullback_sym')
#         cov_labels_wn += [cov_label]
#         centroids_wn += [cov_centroid]

#     # Plot whitened centroids
#     vmin = min([c.min() for c in centroids_wn])
#     vmin = -1 if vmin >= 0 else vmin
#     vmax = max([c.max() for c in centroids_wn])
#     vmax = 1 if vmax <= 0 else vmax
#     for i in range(3):
#         im = axes[1, i].imshow(centroids_wn[i], cmap='seismic',
#                                norm=DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax))
#         # annot(axes[0, i], centroids[i])
#         axes[1, i].set_title(list(trans_dict.values())[i])
#     fig.colorbar(im, ax=axes[1, -1], orientation='vertical')

#     axes[2, 0].plot(wn.filters_)
#     axes[2, 0].set_title('Spatial Filters | chs x red_dims')

#     axes[2, 1].plot(wn.filters_.T)
#     axes[2, 1].set_title('red_dims x chs')

#     return fig, axes

# def plot_explained_variance(pca, n_components, seeg, save, ppt_id, session_id, normalize=False):
#     # TODO: Subplot_adjust offsets title from subplots
#     fig = plt.figure(figsize=(16, 12))
#     gs = fig.add_gridspec(2, 2)
#     fig.suptitle('Explained variance')
    
#     ax0 = fig.add_subplot(gs[0, 0])
#     ax0.set_title('Ratio per component')
#     ax0.plot(pca.explained_variance_ratio_)
#     ax0.set_xlabel('Component [#]')
#     ax0.set_ylabel('Ratio [%]')
#     ax0.set_xticks(range(n_components))
#     ax0.set_ylim(0, ax0.get_ylim()[1])
#     ax0.grid(visible=True, axis='y', alpha=.5)

#     ax1 = fig.add_subplot(gs[0, 1])
#     ax1.set_title('Cumulative ratio')
#     ax1.plot(np.cumsum(pca.explained_variance_ratio_))
#     ax1.set_xlabel('Number of components')
#     ax1.set_ylabel('Ratio [%]')
#     ax1.set_xticks(range(n_components))
#     ax1.set_ylim(0, 1.1)        
#     ax1.grid(visible=True, axis='y', alpha=.5)
    
#     ax2 = fig.add_subplot(gs[1, :])
#     # Most contributing features:
#     # TODO: Bands should be dynamically selected
#     # TODO: Add top k features
#     get_loc_name = lambda contact: seeg['locations'].get(contact, "NaN") if seeg['locations'] else ''
#     channel_names = np.array([f'{contact:>5} {band:6<} {get_loc_name(contact).strip():>28}'\
#                                 # for band in ['delta', 'theta', 'alpha', 'beta', 'gamma', 'gamma+']\
#                                 for band in ['beta', 'gamma+']\
#                                 for contact in seeg['channel_names']])
#     contributing_features = pca.components_.argmax(axis=1)
#     contributing_values = pca.components_.max(axis=1)
#     contributing_values = contributing_values / np.abs(pca.components_).sum(axis=1)

#     print(f'Most contributing feature per component (in descending order):\n{channel_names[contributing_features]}')
#     print()

#     n = 5
#     best_features = []
#     with open(f'./figures/pca/{ppt_id}_{session_id}/component_importance{"_normalized" if normalize else ""}.txt', 'w+') as fobj:
#         print(f'Most contributing features per component (in descending importance):', file=fobj)
#         for i, component in enumerate(pca.components_):
#             best_n_args = np.flip(np.argsort(component)[-n:])
#             print(f'\nComponent {i}', file=fobj)
#             for idx in best_n_args:
#                 best_features += [channel_names[idx]]
#                 print(f'\t{channel_names[idx]:>35}: {component[idx]:>8.3f}{component[idx]/(np.abs(component).sum()):>8.3f}', file=fobj)

#     # TODO: Plot the principle components
#     ax2.barh(np.arange(n_components), contributing_values)
#     ax2.set_yticks(np.arange(n_components))
#     ax2.set_yticklabels([f'{i:>1}: {loc}' for i, loc in enumerate(channel_names[contributing_features])])
#     ax2.invert_yaxis()
#     ax2.set_ylabel(f'Feature')
#     ax2.set_xlabel(f'Max component weight (normalized by sum of absolute weights per component)')
#     ax2.set_title('Highest weighted features')
#     ax2.grid(visible=True, axis='x', alpha=.5)
    
#     plt.subplots_adjust(left=0.22)

#     if save:
#         fig.savefig(f'./figures/pca/{ppt_id}_{session_id}/pca_explained_variance{"_normalized" if normalize else ""}.svg')
#     return best_features

# def plot_transformation_vector(pca, n_components, seeg, save, ppt_id, session_id, normalize=False):
#     # NOTE: I don't think this makes sense.

#     eeg = pca.transform(seeg['eeg'])

#     fig, ax = plt.subplots()
#     for class_ in set(seeg['trial_labels']):
#         idc = np.where(seeg['trial_labels']==class_)[0]
#         ax.scatter(eeg[idc, 0], eeg[idc, 1])
#     n_max = [np.flip(np.argsort(component)[-2:]) for component in pca.components_]
#     for i, i_max in enumerate(n_max):
#         ax.arrow(0, 0, pca.components_[i, i_max[0]]*pca.explained_variance_[i],
#                        pca.components_[i, i_max[1]]*pca.explained_variance_[i],
#                        color='r')
#     fig.savefig(f'./figures/pca/{ppt_id}_{session_id}/biplot{"_normalized" if normalize else ""}.png')

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
    # return fig, ax

# def convert_animation(ppt_id, session_id, normalize=True):
#     # https://trac.ffmpeg.org/wiki/Slideshow
#     print('Converting images to mpeg...')
#     import subprocess
#     path = f'/mnt/c/Users/p70066129/Projects/Riemannian/figures/pca/{ppt_id}_{session_id}/animation'
#     powershell = r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
#     command = f'{powershell} wsl ffmpeg -y -framerate 5 -i {path}/%03d.png {path}/animation{"_normalized" if normalize else ""}.mp4'.split(' ')
#     subprocess.Popen(command)


