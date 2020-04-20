import os
import re
import glob
import pickle
import itertools

import numpy as np

import mne
import tqdm

from skimage import measure
from joblib import Parallel, delayed

from crosspy.preprocessing.seeg.seeg_utils import create_reference_mask

np.random.seed(42)


def _joblib_wrapper_with_amplitude_surr(data_angle, data_amplitude, window_len, i, j, n_shuffles=100):
    n_windows = data_angle.shape[~0] // window_len
        
    phase_diff = np.exp(1j*(data_angle[:, i] - data_angle[:, j]))

    relevance = np.product(data_amplitude[(i,j),:], axis=0) * np.max(np.sign(data_amplitude[(i,j),:]), axis=0)        
    cplv_trials = np.array([np.mean(phase_diff[:, i*window_len:(i+1)*window_len]) for i in range(n_windows)])
    relevance_trials = relevance.reshape(n_windows, window_len).mean(axis=1)
    
    surr_cplv = np.zeros((n_shuffles,) + cplv_trials.shape, dtype=np.complex)
    
    for k in range(n_shuffles):
        indices = np.random.randint(0, data_angle.shape[0]-1, size=data_angle.shape[0])
        surr_diff = np.exp(1j*(data_angle[:, i] - data_angle[indices, j]))
        surr_cplv[k] = np.array([np.mean(surr_diff[:, idx*window_len:(idx+1)*window_len]) for idx in range(n_windows)])
        
    return cplv_trials, relevance_trials, surr_cplv.mean(axis=0)


def baseline_zscore(arr, baseline):
    m = arr[...,:baseline].mean(axis=-1, keepdims=True)
    s = arr[...,:baseline].std(axis=-1, keepdims=True)
    
    return (arr - m)/s


def is_monopolar(chan):
    tokens = chan.split('-')
    return len(tokens) == 1 or len(tokens[1]) == 0


def create_task_masks(x_indices, n_chans, y_indices=None):
    if y_indices is None:
        y_indices = x_indices.copy()

    x_mask = np.zeros(n_chans, dtype=bool)
    x_mask[x_indices] = True

    y_mask = np.zeros(n_chans, dtype=bool)
    y_mask[y_indices] = True
        
    task_mask = np.zeros((n_chans, n_chans), dtype=bool)
    
    for i,j in itertools.product(range(n_chans), range(n_chans)):
        if x_mask[i] == True and y_mask[j] == True:
            task_mask[i,j] = True
    
    return task_mask


def main():
    hga_frequencies = np.array(
            [
                110.        , 130.        , 150.        , 170.        ,
                190.        , 210.        , 230.        , 250.        ,
                270.        , 290.        , 310.        , 330.        ,
                350.        , 370.        , 390.        , 410.        ,
                430.        , 450.        
                ])


    clean_subjects = [
                    'data/sub-06_task-gonogo_mon-cw_epo.fif', 
                     'data/sub-03_task-gonogo_mon-cw_epo.fif',
                     'data/sub-04_task-gonogo_mon-cw_epo.fif',
                     'data/sub-09_task-gonogo_mon-cw_epo.fif',
                     'data/sub-05_task-gonogo_mon-cw_epo.fif',
                     'data/sub-10_task-gonogo_mon-cw_epo.fif',
                     'data/sub-08_task-gonogo_mon-cw_epo.fif',
                     'data/sub-01_task-gonogo_mon-cw_epo.fif',
                     'data/sub-11_task-gonogo_mon-cw_epo.fif',
                                     
    ]

    event_dict = np.load('event_dict.npy', allow_pickle=True)
    event_codes = event_dict.flatten()[0]

    cohort_cplv = list()
    cohort_relevance = list()
    cohort_ref_mask = list()
    cohort_task_indices = list()

    for fname in tqdm.tqdm(clean_subjects, desc='subjects'):
        subj_idx = re.findall(r'\d+', fname)[0]
        subj_codes = event_codes['sub-{}'.format(subj_idx)]

        go_code = [value for key,value in subj_codes.items() if '_Go' in key][0]
        nogo_code = [value for key,value in subj_codes.items() if '_NoGo' in key][0]

        data_fname = 'sub-{}_evoked_plv_go.pickle'.format(subj_idx)
        data_path = os.path.join('derivatives', 'evoked_plv', data_fname)
        
        if os.path.exists(data_path):
            print('Subject {} is processed!'.format(subj_idx))
            continue
        
        epochs = mne.read_epochs(fname, proj=False, verbose=False)
        go_mask = (epochs.events[:, 2] == go_code)
        nogo_mask = (epochs.events[:, 2] == nogo_code)
        
        if epochs._data.shape[2] < 1500:
            print('Subject {} has wrong shape!'.format(subj_idx))
            continue
        
        epochs.drop(nogo_mask)
        epochs.drop_channels([ch for ch in epochs.ch_names if is_monopolar(ch)])
        epochs.apply_baseline((None,None))
        
        ref_mask = np.triu(create_reference_mask(epochs).astype(bool), 1)
        
        n_chans = ref_mask.shape[0]
        n_freqs = len(hga_frequencies)
        
        contact_pairs = list(zip(*np.where(ref_mask)))
        cplv_freqwise = np.zeros((n_freqs, n_chans, n_chans, 130), dtype=np.complex)
        cplv_freqwise_surr = np.zeros((n_freqs, n_chans, n_chans, 130), dtype=np.complex)
        relevance_freqwise = np.zeros((n_freqs, n_chans, n_chans, 130))
        
        subj_task_indices = list()
        subj_hga_indices = list()
        subj_hga_responses = list()
        
        ersd_avg = epochs._data.mean(axis=0)
        ersd_corrected = baseline_zscore(ersd_avg, 500)
        
        raw_sorted = np.argsort(ersd_corrected[:, 650:950].mean(axis=1))
        abs_sorted = np.argsort(np.abs(ersd_corrected[:, 650:950]).mean(axis=1))
        
        indices_response = np.zeros(epochs._data.shape[1], dtype=int)
        indices_response[:30] = abs_sorted[:30]
        indices_response[-30:] = raw_sorted[:30][::-1]

        for freq_idx, freq in enumerate(tqdm.tqdm(hga_frequencies, leave=False, desc='Frequencies')):        
            data_preprocessed = mne.time_frequency.tfr_array_morlet(epochs._data, 1000, [freq], 7.5, n_jobs=32, verbose=False).squeeze()
            data_preprocessed = data_preprocessed[..., 100:-101] #cutoff part of the data to remove filtering artefacts
            
            data_envelope = np.abs(data_preprocessed)
            amp_profile = data_envelope.mean(axis=0)
            amp_profile_zs = baseline_zscore(amp_profile, 400)
            amp_response = amp_profile_zs[:, 500:800].mean(axis=1)
            
            hga_indices = np.argsort(amp_response)
            
            data_angle = np.angle(data_preprocessed) # N_trials x N_contacts x Trial_size
                    
            contact_results = Parallel(n_jobs=32)(delayed(_joblib_wrapper_with_amplitude_surr)(data_angle, amp_profile_zs,
                                                                                        10, *ch_pair, n_shuffles=20) 
                                                for ch_pair in tqdm.tqdm(contact_pairs, leave=False, desc='Edges'))
            
            for pair, (vec_cplv, vec_rel, vec_surr) in zip(contact_pairs, contact_results):
                cplv_freqwise[freq_idx, pair[0], pair[1]] = vec_cplv
                relevance_freqwise[freq_idx, pair[0], pair[1]] = vec_rel
                cplv_freqwise_surr[freq_idx, pair[0], pair[1]] = vec_surr
                        
            subj_task_indices.append(indices_response)
            subj_hga_indices.append(hga_indices)
            subj_hga_responses.append(amp_response)
        
        subj_task_indices = np.array(subj_task_indices)
        subj_hga_indices = np.array(subj_hga_indices)
        subj_hga_responses = np.array(subj_hga_responses)
        
        cohort_ref_mask.append(ref_mask)
        cohort_cplv.append(cplv_freqwise)
        cohort_relevance.append(relevance_freqwise)
        cohort_task_indices.append(subj_task_indices)
        
        out_data = {'ref_mask': ref_mask, 'cplv': cplv_freqwise, 'surr_cplv': cplv_freqwise_surr,
                    'ersd_raw': ersd_corrected, 'hga_response': subj_hga_responses,
                    'relevance': relevance_freqwise, 'task_indices_sorted': subj_task_indices, 'hga_indices': subj_hga_indices}
        pickle.dump(out_data, open(data_path, 'wb'))


if __name__ == '__main__':
    main()