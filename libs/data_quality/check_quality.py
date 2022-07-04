'''
Reasons to flag channels:

1) Flat signal
2) Marker channels or other irrelevant channels (EKG+ etc)
3) Abnormal amplitude distribution
4) Excessive (line) noise
5) Outliers

'''

import re
from typing import Callable, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore, norm
from mne.filter import filter_data

class QualityChecker:

    def __init__(self) -> None:
        self.results = {}

    def _has_n_unique_values(self, 
                            eeg: np.array,
                            n_values: int) -> list:
        if n_values==1:
            # Vectorized method for efficiency.    
            flagged_channels = np.where(np.all(eeg == eeg[0], axis=0))[0]
        else:
            flagged_channels = np.where(np.array([len(np.unique(eeg[:, ch])) 
                                                  for ch in range(eeg.shape[1])]) <= n_values)[0]
        return flagged_channels

    def get_marker_channels(self, 
                            eeg: np.array,
                            channel_names: list=None,
                            plot: bool=False) -> list:

        if channel_names:
            flagged_channels = [i for i, ch in enumerate(channel_names) if 'MKR' in ch]
        else:
            # NOTE: Assumes that the marker channels oscillate between two values
            flagged_channels = self._has_n_unique_values(eeg, 2)
        
        self.results.update({'marker_channels': {
                                'flagged_channels': flagged_channels}
                            })
        if plot and len(flagged_channels) > 0:
            fig, ax = plt.subplots(nrows=len(flagged_channels), ncols=1)
            for i, ch in enumerate(flagged_channels):
                ax[i].plot(eeg[:, ch])
                ax[i].set_title('Channel #{}'.format(channel_names[ch] \
                                                     if channel_names else ch))
            fig.suptitle('Channels flagged as Marker channel')

        return flagged_channels
    
    def get_ekg_channel(self,
                        eeg: np.array,
                        channel_names: list=None,
                        plot: bool=False) -> list:
        
        if channel_names:
            flagged_channel = [i for i, ch in enumerate(channel_names) if 'EKG' in ch]
        else:
            # On the assumption that there are many negative peaks in the same channel:
            # Calculates the <search_in_n_max_values> maximum values and retrieves the
            # channels per timestamps in which these maximum values appear. The channel 
            # with the most maximum values per timestamp is flagged.
            # TODO: Check some sort of normalizing measure (Comparing Z-Scores?)
            # TODO: ERROR: WILL ALWAYS FLAG A CHANNEL
            # NOTE: Assumption: EKG has many large negative spikes outside of the range
            #                   of regular EEG channels
            search_in_n_max_values = 10000
            sorted_values = np.sort(np.abs(eeg), axis=0)
            channel_with_max_values = np.argmax(sorted_values[-search_in_n_max_values:, :],
                                                axis=1)
            unique, counts = np.unique(channel_with_max_values, return_counts=True)
            flagged_channel = [unique[np.argmax(counts)]]
        
        self.results.update({'ekg_channels': {
                                'flagged_channels': flagged_channel}
                            })

        if plot and len(flagged_channel) > 0:
            fig, ax = plt.subplots(nrows=len(flagged_channel), ncols=1)
            ax.plot(eeg[:, flagged_channel][0])
            ax.set_title('Channel #{}'.format(channel_names[flagged_channel[0]] \
                                              if channel_names else flagged_channel[0]))
            fig.suptitle('Channels flagged as EKG channel')

        return flagged_channel

    def get_disconnected_channels(self,
                                  eeg: np.array,
                                  channel_names: list=None,
                                  plot: bool=False) -> list:
        if channel_names:
            pattern = '(?<![A-Za-z])[Ee][l\d]'
            flagged_channels = [i for i, name in enumerate(channel_names) \
                                       if re.search(pattern, name)]
        else:
            # Flag if the channels average log(power) over all frequencies
            #   is higher than the average log(power) over all channels.
            # TODO: Check if marker_channels and ekg_channel are already found,
            #       and remove them from the flagged channels array.
            # TODO: ERROR! WILL ALWAYS FLAG VALUES
            ps_log_mean = np.log(np.abs(np.fft.rfft(eeg.T-eeg.T.mean(axis=0)))**2).T.mean(axis=0)
            flagged_channels = np.where(ps_log_mean > ps_log_mean.mean())[0]

        self.results.update({'disconnected_channels': {
                                'flagged_channels': flagged_channels}
                            })

        if plot and len(flagged_channels) > 0:
            fig, ax = plt.subplots(nrows=len(flagged_channels), ncols=1)
            for i, ch in enumerate(flagged_channels):
                ax[i].plot(eeg[:, ch])
                ax[i].set_title('Channel #{}'.format(channel_names[ch] \
                                                     if channel_names else ch))
            fig.suptitle('Channels flagged as disconnected channels')


        return flagged_channels

    def excessive_line_noise(self, 
                             data: np.array, 
                             fs: int,
                             freq_line: int=50,
                             distance_metric: Callable[[np.array, tuple], float]=None,
                             plot: bool=False) -> list:
        '''
        To define a distance metric:
            Supply a function that takes as input the mean filtered line power (2d np.array)
            and optional keywords arguments. The function should return a boundary value,
            where the channels with values larger than the returned value are flagged.        
        '''
        
        # 1) Filter power line
        line_noise = filter_data(data.T.astype(np.float64), fs, \
                                 freq_line-1, freq_line+1, method='fir').T
        
        # 2) Calculate power
        line_noise = np.mean(np.abs(line_noise), axis=0)
        
        # 3) determine the distance metric
        if not distance_metric:
            def distance_metric(line_noise: np.array, **kwargs) -> float:
                # Finds the interquartile distance (75% - 25%)
                quartile_75 = np.percentile(line_noise, 75, interpolation='linear')
                quartile_25 = np.percentile(line_noise, 25, interpolation='linear')
                interquartile_dist = quartile_75 - quartile_25
                return np.percentile(line_noise, 75) + 2*interquartile_dist

        decision_boundary = distance_metric(line_noise)

        # 4) Flag the channels
        flagged_channels = np.where(line_noise > distance_metric(line_noise))[0]

        # 5) Save the results
        self.results.update({'excessive_line_noise': {
                                'line_frequency': freq_line,
                                'fs': fs,
                                'line_power': line_noise,
                                'quartile_25': np.percentile(line_noise, 75, interpolation='linear'),
                                'quartile_75': np.percentile(line_noise, 25, interpolation='linear'),
                                'decision_boundary': decision_boundary,
                                'flagged_channels': flagged_channels}})

        # 6) Plot if necessary
        if plot and len(flagged_channels) > 0:
            plt.figure()
            plt.plot(line_noise, label='Line activity')
            xlim_left, xlim_right = plt.xlim()
            plt.hlines(decision_boundary, xmin=xlim_left, xmax=xlim_right, color='r', label='Decision boundary')
            plt.title('Line power')
            plt.xlabel('Channels')
            plt.ylabel('Power')
            plt.legend()
            plt.tight_layout()
            plt.savefig('line_noise.png')

        # 7) Return the indices of bad channels
        return flagged_channels

    def consistent_timestamps(self, 
                              timestamps: np.array,
                              expected_freq: int,  # Frequency provided by the amp
                              max_allowed_freq_diff: float=0.1,  # Allowed offset of expected_freq
                              max_allowed_diff: int=1, # [ms] Allowed diff in interstep interval
                              plot: bool=False) -> list:
        '''
        Invalid if:
           a) difference in intertimestep interval is larger than <max_allowed_diff>
           b) the observed frequency is not equal to <expected_freq>        
        '''

        timesteps = np.diff(timestamps)
        observed_frequency = 1/timesteps
        diff_timesteps = np.diff(timesteps)
        invalid_timesteps = np.where(abs(diff_timesteps) > max_allowed_diff/1000)[0]

        mean_observed_frequency = np.mean(observed_frequency)
        has_invalid_frequency = np.round(mean_observed_frequency, 1) < expected_freq - max_allowed_freq_diff \
                                or np.round(mean_observed_frequency, 1) > expected_freq + max_allowed_freq_diff

        self.results.update({'consistent_timestamps':{
                                'invalid_timesteps': invalid_timesteps,
                                'has_invalid_timesteps': any(invalid_timesteps),
                                'mean_observed_frequency': mean_observed_frequency,
                                'has_invalid_frequency': has_invalid_frequency}})

        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].plot(diff_timesteps, color='b', label='Difference between timesteps')
            xlim_left, xlim_right = ax[0].get_xlim()
            ax[0].axhline(max_allowed_diff, xmin=xlim_left, xmax=xlim_right, label='Upper boundary', color='r')
            ax[0].axhline(-max_allowed_diff, xmin=xlim_left, xmax=xlim_right, label='Lower boundary', color='r')
            ax[0].set_xlabel('Timesteps')
            ax[0].set_ylabel('Time difference [ms]')
            ax[0].legend()
            ax[0].set_title('Time difference between timesteps')

            ax[1].plot(observed_frequency, color='b', label='Observed Frequency')
            ax[1].axhline(expected_freq, xmin=xlim_left, xmax=xlim_right, label='Expected frequency', color='g')
            ax[1].axhline(expected_freq + max_allowed_freq_diff, xmin=xlim_left, xmax=xlim_right, label='Upper boundary', color='r')
            ax[1].axhline(expected_freq - max_allowed_freq_diff, xmin=xlim_left, xmax=xlim_right, label='Lower boundary', color='r')
            ax[1].set_ylim(bottom=min([min(observed_frequency), expected_freq])-1,
                           top=max([max(observed_frequency), expected_freq])+1)
            ax[1].set_xlabel('Timesteps')
            ax[1].set_ylabel('Frequency [Hz]')
            ax[1].legend()
            ax[0].set_title('Frequency')
            
            plt.tight_layout()
            fig.savefig('consistent_timestamps.png')

        return invalid_timesteps

    def flat_signal(self, 
                    eeg: np.array,
                    plot: bool=False) -> list:
        '''
        Flag if channel has single unique value
        '''

        flagged_channels = self._has_n_unique_values(eeg, 1)

        self.results.update({'flat_signal': {
                                'flagged_channels': flagged_channels,
                                'flagged_values': eeg[0, flagged_channels]}
                            })

        if plot and len(flagged_channels) > 0:
                fig, ax = plt.subplots(nrows=flagged_channels.size, ncols=1)
                for i, ch in enumerate(flagged_channels):
                    ax[i].plot(eeg[:, ch])
                    ax[i].set_title('Channel #{}'.format(ch))
                fig.suptitle('Channels flagged for flat signal')

        return flagged_channels

    def abnormal_amplitude(self, 
                           eeg: np.array,
                           plot: bool=False) -> list:
        # Flags channels if amplitude in the channel
        # is significantly higher than the other channels
        # assuming a normal distribution

        lrms = np.log(np.sqrt(np.mean(eeg**2, axis=0)))
        lrms_norm = zscore(lrms)
        ampl_p = 1-norm.cdf(np.abs(lrms_norm))
        flagged_channels = np.where(ampl_p <= 0.05)[0]
        self.results.update({'abnormal_amplitude': {
                                'flagged_channels': flagged_channels,
                                'lrms_zscore': lrms_norm,
                                'p_values': ampl_p}
                            })
        if plot and len(flagged_channels) > 0:
            fig, ax = plt.subplots(nrows=len(flagged_channels), ncols=1)
            for i, ch in enumerate(flagged_channels):
                ax[i].plot(eeg[:, ch])
                ax[i].set_title('Channel #{}'.format(ch))
            fig.suptitle('Channels flagged for abnormal amplitude')

        return flagged_channels

    def detect_outliers(self, 
                        eeg: np.array,
                        plot: bool=False) -> list:
        lrms = np.log(np.sqrt(eeg**2))#+1e-10)
        nan1 = np.isnan(lrms)
        lrms_norm = zscore(lrms, axis=0)
        nan2 = np.isnan(lrms_norm)




        ampl_p = 1-norm.cdf(np.abs(lrms_norm))
        flagged_channels = np.where(ampl_p <= 0.05)[0]
        pass

    def run_all(self,
                eeg: np.array,
                timestamps: np.array,
                fs: Union[int, float],
                channel_names: list=None,
                return_all_flagged_channels: bool=False,
                plot: bool=False) -> Union[None, list]:

        self.consistent_timestamps(timestamps, float(fs), plot=plot)
        self.excessive_line_noise(eeg, float(fs), plot=plot)
        self.abnormal_amplitude(eeg, plot=plot)
        self.flat_signal(eeg, plot=plot)
        self.get_marker_channels(eeg, channel_names=channel_names, plot=plot)
        self.get_ekg_channel(eeg, channel_names=channel_names, plot=plot)
        self.get_disconnected_channels(eeg, channel_names=channel_names, plot=plot)
        self.detect_outliers() # TODO

        if self.results['consistent_timestamps']['invalid_timesteps'].size > 0:
            print('WARNING: invalid timestamps found!')
        
        if plot:
            plt.show()

        if return_all_flagged_channels:
            flagged_channels = \
                np.unique(np.concatenate(
                        [results.get('flagged_channels', np.array([])) \
                         for results in self.results.values()]
                    )).astype(int)
            return flagged_channels

if __name__=='__main__':
    import sys
    paths = [r"./../../Utility/Helpers/xdf_reader/"]
    for path in paths:
        sys.path.insert(0, path)
    from read_xdf import read_xdf
    path = r'your_path_to_xdf'
    # path = r'L:\FHML_MHeNs\sEEG\kh25\20210802\grasp.xdf'
    # path = r'L:\FHML_MHeNs\sEEG\kh14\sentences1.xdf'
    path = r'L:\FHML_MHeNs\sEEG\kh14\imagine.xdf'
    data, raw = read_xdf(path)
    
    plot = True
    qc = QualityChecker()
    # qc.consistent_timestamps(data['Micromed']['ts'], data['Micromed']['fs'], plot=plot)
    irrelevant_channels = np.concatenate([
            qc.get_marker_channels(data['Micromed']['data'], 
                                channel_names=data['Micromed']['channel_names']),
            qc.get_ekg_channel(data['Micromed']['data'], 
                            channel_names=data['Micromed']['channel_names']),
            qc.get_disconnected_channels(data['Micromed']['data'], 
                                        channel_names=data['Micromed']['channel_names'])
            ]).astype(int)
    eeg = np.delete(data['Micromed']['data'], irrelevant_channels, axis=1)
    channel_names = np.delete(data['Micromed']['channel_names'], irrelevant_channels)
    qc.detect_outliers(eeg)



    # flagged_channels = \
    #     qc.run_all(data['Micromed']['data'],
    #            data['Micromed']['ts'],
    #            data['Micromed']['fs'],
    #            channel_names=data['Micromed']['channel_names'],
    #            return_all_flagged_channels=True,
    #            plot=plot)
    # print(flagged_channels)
    
    plt.show()

    