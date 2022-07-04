from pathlib import Path
from os import listdir

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import ttest_1samp

ERP = 0
POW = 1
MVR_TE = 1
LVR_TE = 3
MEAN = 0
STD = 1

OUTPUT_N_LINES = 29
OUTPUT_CROSS_LINES = 11
FONTSIZE = 25

def autolabel(ax, rects, error):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if height == 0:
            continue
        ax.annotate('{:0.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize='medium')

def retrieve_single_test_cross(data, start_idx):
    results = {}
    len_scores = 1

    classes = [class_.strip() for class_ in data[1].split('|')[1:]]
    get_score = lambda data, i, score: float(data[i].split('|')[score].strip())

    results[classes[0]] = {
        'train': [[get_score(data, fold, 1) for fold in range(start_idx, start_idx+len_scores)],
                get_score(data, start_idx+len_scores, 1)],
        'test': [[get_score(data, fold, 2) for fold in range(start_idx, start_idx+len_scores)],
                get_score(data, start_idx+len_scores, 2)]}
    try:
        results[classes[1]] = {
            'train': [[get_score(data, fold, 3) for fold in range(start_idx, start_idx+len_scores)],
                    get_score(data, start_idx+len_scores, 3)],
            'test': [[get_score(data, fold, 4) for fold in range(start_idx, start_idx+len_scores)],
                    get_score(data, start_idx+len_scores, 4)]}
    except Exception:
        pass

    return results

def retrieve_single_test(data, start_idx):
    results = {}
    len_scores = 10

    classes = [class_.strip() for class_ in data[1].split('|')[1:]]
    get_score = lambda data, i, score: float(data[i].split('|')[score].strip())

    results[classes[0]] = {
        'train': [[get_score(data, fold, 1) for fold in range(start_idx, start_idx+len_scores)],
                get_score(data, start_idx+len_scores, 1)],
        'test': [[get_score(data, fold, 2) for fold in range(start_idx, start_idx+len_scores)],
                get_score(data, start_idx+len_scores, 2)]}
    try:
        results[classes[1]] = {
            'train': [[get_score(data, fold, 3) for fold in range(start_idx, start_idx+len_scores)],
                    get_score(data, start_idx+len_scores, 3)],
            'test': [[get_score(data, fold, 4) for fold in range(start_idx, start_idx+len_scores)],
                    get_score(data, start_idx+len_scores, 4)]}
    except Exception:
        pass

    return results

def process_file(data, type_=None):
    results = {}

    # INFO
    line = data[0]
    info = {
        'datetime': line.split(' - ')[0],
        'n_channels': int(line.split('|')[1]\
                              .split(':')[1]\
                              .strip()),
        'bands': line.split('|')[2].strip()[2:-2],
        'windows': line.split('|')[3][4:14],
        'clf': line.split('learner:')[1][:-1]}

    if type_=='cross':
        for i, line in enumerate(data[::OUTPUT_CROSS_LINES]):
            components = int(data[i*OUTPUT_CROSS_LINES].split("principle_components':")[1].split('}')[0])
            results[components] = {'info': info,
                                'riemann': retrieve_single_test_cross(data, i*OUTPUT_CROSS_LINES+3),
                                'lda': retrieve_single_test_cross(data, i*OUTPUT_CROSS_LINES+7)}
    else:
        for i, line in enumerate(data[::OUTPUT_N_LINES]):
            components = int(data[i*OUTPUT_N_LINES].split("principle_components':")[1].split('}')[0])
            results[components] = {'info': info,
                                'riemann': retrieve_single_test(data, i*OUTPUT_N_LINES+3),
                                'lda': retrieve_single_test(data, i*OUTPUT_N_LINES+16)}
    return results

def retrieve_results(main_path):
    files = listdir(main_path)

    results = {}
    for file in files:
        if ('kh' in file) and ('significant_locs' not in file) \
                          and ('.pkl' not in file):
            ppt_id = file.split('_')[0]
            
            session_id = file.split('_')[1][:-4]
            if session_id != '0':
                continue

            with open(f'{main_path}/{file}', 'r') as fid:
                data = fid.readlines()
                results['{}_{}'.format(ppt_id, session_id)] = process_file(data, type_='cross' if 'cross' in str(main_path.parent) else None)
    return results

def line_plot(data, name, ax=None, fig=None):
    pcs = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    means_dict = {}
    for ppt, scores in data.items():
        means_dict[ppt] = [scores[pc]['riemann']['Move vs Rest']['test'][1] \
                              for pc in scores.keys()]
    
    for ppt, scores in means_dict.items():
        if len(scores) <= 10:
            new = [np.nan] * 11
            new[:len(scores)] = scores
            means_dict[ppt] = new
        assert len(means_dict[ppt]) == 11

    means = np.vstack([mean for mean in means_dict.values()])
    mean = np.nanmean(means, axis=0)
    sems = sem(means, axis=0, nan_policy='omit')
    stds = np.nanstd(means, axis=0)
    
    t, p = ttest_1samp(means, .5, axis=0, nan_policy='omit')
    p = p*means.shape[1]  # Bon

    print(f'''
    Significance:
        n.s.: {np.where(p>=0.05)[0]} 
        *:    {np.where((p<0.05) & (p>0.01))[0]}
        **:   {np.where((p<0.01) & (p>0.001))[0]}
        ***:  {np.where(p<0.001)[0]}
    ''')


    if not ax:
        fig, ax = plt.subplots()
    
    for i, scores in enumerate(means):
        if i==0:
            ax.plot(pcs, scores, color='grey', alpha=0.2, label='Individual scores')
            ax.scatter(1, -1, facecolors='lightgrey', edgecolors='black', s=25,
                       label='Mean score')
        else:    
            ax.plot(pcs, scores, color='grey', alpha=0.2)

    ax.plot(pcs, mean, color='k', zorder=1)

    colors = ['black'] * mean.shape[0]
    edgecolors = np.where(p<0.05, 'black', 'black')
    facecolors = np.where(p<0.05, 'black', 'lightgrey')
    ax.scatter(pcs, mean, facecolors=facecolors, edgecolors=edgecolors, s=25, zorder=2)
    ax.scatter([], [], facecolor='black', edgecolor='black', label='Mean score [p<0.05]')

    for i, txt in enumerate(mean):
        if txt in [min(mean), max(mean)]:
            ax.annotate(np.round(txt, 2), (pcs[i], txt), xytext=(pcs[i]-3, txt+0.05), fontsize=FONTSIZE/2)

    err = stds
    ax.fill_between(pcs, mean-err, mean+err,
                    alpha=0.15, color='k', label='Standard Deviation')
    
    ax.set_ylim(0, 1)
    ax.set_xticks(pcs)
    ax.axhline(0.5, alpha=0.3, linestyle='dotted', color='k', label='Chance level')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(f'{name}')
           
    return fig, ax, means

def line_results(path):
    path_save = Path(r'./figures/')
    path_save.mkdir(parents=True, exist_ok=True)

    paths = [p for p in path.glob('**/*') if p.is_dir()]
    
    translate = {'grasp_beta': {'name': ['Exec', 'Beta'],
                                'idx': [0, 0]},
                 'grasp_betahigh_gamma': {'name': ['Exec', 'Beta - High Gamma'],
                                          'idx': [0, 2]},
                 'grasp_high_gamma': {'name': ['Exec', 'High Gamma'],
                                      'idx': [0, 1]},
                 'imagine_beta': {'name': ['Imag', 'Beta'],
                                  'idx': [1, 0]},
                 'imagine_betahigh_gamma': {'name': ['Imag', 'Beta - High Gamma'],
                                            'idx': [1, 2]},
                 'imagine_high_gamma': {'name': ['Imag', 'High Gamma'],
                                        'idx': [1, 1]}}
    
    nrows = 2
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 8), dpi=300)
    means = []
    for i, path in enumerate(paths):
        # print(list(translate.keys())[i])
        if path.name not in translate:
            continue
        results = retrieve_results(path)

        idx = translate[path.name]['idx']
        _, _, m = line_plot(results, '', ax=axs[idx[0], idx[1]])
        means += [m]


    print(f'''
        Averages:
            Overall: {np.nanmean(np.vstack(means)):.3f} + {np.nanstd(np.vstack(means)):.3f}
            Execute: {np.nanmean(np.vstack(means[:3])):.3f} + {np.nanstd(np.vstack(means[:3])):.3f}
            Imagine: {np.nanmean(np.vstack(means[3:])):.3f} + {np.nanstd(np.vstack(means[3:])):.3f}
    ''')

    axs[0, 0].set_title('Beta', fontsize=FONTSIZE)
    axs[0, 1].set_title('High Gamma', fontsize=FONTSIZE)
    axs[0, 2].set_title('Beta +\nHigh Gamma', fontsize=FONTSIZE)

    axs[0, 0].annotate('Execute', xy=(-0.40, .5), xycoords='axes fraction', rotation='vertical', fontsize=FONTSIZE,
                                  verticalalignment='center')
    axs[1, 0].annotate('Imagine', xy=(-0.40, .5), xycoords='axes fraction', rotation='vertical', fontsize=FONTSIZE,
                                  verticalalignment='center')
    
    # axs[0, -1].legend(frameon=False, bbox_to_anchor=(1, 1.05))

    for i in range(ncols):
        axs[1, i].set_xlabel('Principal components', fontsize=FONTSIZE-9)
    for i in range(nrows):
        axs[i, 0].set_ylabel('Area under the curve', fontsize=FONTSIZE-9)

    # plt.tight_layout()
    fig.savefig(path_save/'main_results.png')
    return fig, axs

def line_results_cross_task(path):
    paths = [p for p in path.glob('**/*') if p.is_dir()]
    
    translate = {'cross_beta': {'name': ['Cross', 'Beta'],
                                'idx': [0]},
                 'cross_betahigh_gamma': {'name': ['Cross', 'Beta - High Gamma'],
                                          'idx': [1]},
                 'cross_high_gamma': {'name': ['Cross', 'High Gamma'],
                                      'idx': [2]}}
    
    nrows = 1
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 8))
    means = []
    for i, path in enumerate(paths):

        if path.name not in translate:
            continue

        results = retrieve_results(path)

        idx = translate[path.name]['idx']
        _, _, m = line_plot(results, '', ax=axs[idx[0]])
        means += [m]
    
    
    axs[0].set_title('Beta', fontsize=FONTSIZE)
    axs[1].set_title('High Gamma', fontsize=FONTSIZE)
    axs[2].set_title('Beta +\nHigh Gamma', fontsize=FONTSIZE)

    for i in range(ncols):
        axs[i].set_xlabel('Principal components', fontsize=FONTSIZE-9)
    axs[0].set_ylabel('Area under the curve', fontsize=FONTSIZE-9)

    plt.tight_layout()

    return fig, axs


if __name__=='__main__':

    path = Path(r"")
    fig, axs = line_results(path)

    path = Path(r"")
    fig, axs = line_results_cross_task(path)
