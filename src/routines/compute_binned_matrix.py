import os
import glob
import re
import itertools
import pickle

from collections import defaultdict

import pickle

import numpy as np
import pandas as pd
import cupy as cp

import mne

from matplotlib import pyplot as plt

from joblib import Parallel, delayed

import tqdm

from crosspy.preprocessing.seeg.seeg_utils import create_reference_mask
from crosspy.preprocessing.signal import preprocess_data_morlet
from crosspy.core.python_utils import zip_constant_argument
from crosspy.preprocessing.seeg.support import clean_montage, drop_monopolar_channels
from crosspy.core.phaselags.support import is_different_ref_pair

eps = 1e-10
def create_surrogate(data, module=np):
    res = module.zeros_like(data)

    for i in range(data.shape[0]):
        res[i] = module.roll(data[i], np.random.randint(data.shape[1]))

    return res


def cplv_pairwise(data: np.array, return_surr=False, module=np) -> tuple:
    """
        computes cPLV for given data.
    :param data:  numpy array of size [N, S] where N is number of channels, S is number of samples
    :param temp_mask: binary mask indicating which samples should be excluded from analysis
    :param return_surr: return surrogate cPLV or not
    :return: numpy array or tuple of two numpy arrays indicating cPLV and surrogate cPLV
    """
    n_channels, n_samples = data.shape

    data_normalized = data / module.abs(data)
    values = module.inner(data_normalized, module.conj(data_normalized)) / n_samples

    if return_surr:
        data_shift_normalized = create_surrogate(data_normalized)
        values_surr = module.inner(data_shift_normalized, module.conj(data_shift_normalized)) / n_samples

        return values, values_surr
    else:
        return values


def compute_phase_diff(x_analog, y_analog, module=np):    
    x_normed = x_analog/module.abs(x_analog)
    y_normed = y_analog/module.abs(y_analog)
    
    return x_normed * module.conj(y_normed)


def rotate_signal(signal, module=np):
    shift = np.random.randint(signal.shape[0])
    return module.roll(signal, shift)


def normalize_signal(sig, module=np):
    sig_abs = module.abs(sig)
    return sig_abs / module.mean(sig_abs)


def quantile_bins_numpy(sig, quantiles):
    ch_x_quantiles = np.quantile(sig, quantiles)
    return np.digitize(sig, ch_x_quantiles) - 1


_digitize_kernel = cp.core.ElementwiseKernel(
    'S x, raw T bins, int32 n_bins',
    'raw U y',
    '''
    if (x < bins[0] or bins[n_bins - 1] < x) {
        return;
    }
    int high = n_bins - 1;
    int low = 0;
    while (high - low > 1) {
        int mid = (high + low) / 2;
        if (bins[mid] <= x) {
            low = mid;
        } else {
            high = mid;
        }
    }
    y[i] = low;
    ''')

def digitize_cupy(sig_cp, bins):
    sig_lbl = cp.zeros(sig_cp.shape[0], dtype=int)
    _digitize_kernel(sig_cp, bins, bins.size, sig_lbl)
    
    return sig_lbl

def percentile_bins_gpu(sig, quantiles):
    sig_cp = cp.array(sig)
    sig_quantiles = cp.percentile(sig_cp, cp.array(quantiles))
    
    return digitize_cupy(sig_cp, sig_quantiles)


def percentile_bins(sig, bins, module):
    if module is np:
        return np.percentile(sig, bins)
    else:
        return percentile_bins_gpu(sig, bins)


def digitize_wrapper(sig, bins, module=np):
    if module is np:
        return np.digitize(sig, bins) - 1
    else:
        bins_cp = cp.array(bins)
        return digitize_cupy(sig, bins_cp)


def take_rows_cols(m, rows, cols, module=np):
    rows_arr = module.array(rows)[:, None]
    return m[rows_arr, cols]


def _process_single_pair(pair_indexes, theta_phase, data_theta, data_ripples, n_bins=5, n_boots=25, module=np):
    ch_x, ch_y = pair_indexes
    
    phase_bins = module.linspace(-np.pi, np.pi, n_bins+1)
    amp_quantiles = [0, 25, 50, 75, 100]

    pair_matrix = module.zeros((n_bins, n_bins), dtype=module.complex)
    pair_matrix_surr = module.zeros((n_boots, n_bins, n_bins), dtype=module.complex)
    count_matrix = module.zeros((n_bins, n_bins), dtype=int)
    amp_matrix = module.zeros((n_bins, n_bins), dtype=float)
    
    amp_ch_x = normalize_signal(data_ripples[ch_x], module)
    amp_ch_y = normalize_signal(data_ripples[ch_y], module)

    bins_amp_ch_x = percentile_bins(amp_ch_x, amp_quantiles, module)
    bins_amp_ch_y = percentile_bins(amp_ch_y, amp_quantiles, module)
    
    bins_ch_x = digitize_wrapper(theta_phase[ch_x], phase_bins, module=module)
    bins_ch_y = digitize_wrapper(theta_phase[ch_y], phase_bins, module=module)

    phase_diff = compute_phase_diff(data_ripples[ch_x], data_ripples[ch_y], module=module)
    amp_idxs = ((bins_amp_ch_x == 3) & (bins_amp_ch_y == 3)) 
    for x_phase_idx, y_phase_idx in itertools.product(range(n_bins), range(n_bins)):
        phase_idxs = (bins_ch_x == x_phase_idx) & (bins_ch_y == y_phase_idx)
        
        count_matrix[x_phase_idx, y_phase_idx] = module.sum(phase_idxs & amp_idxs)
    
    n_samples = module.min(count_matrix)
    for x_phase_idx, y_phase_idx in itertools.product(range(n_bins), range(n_bins)):
        phase_idxs = (bins_ch_x == x_phase_idx) & (bins_ch_y == y_phase_idx)
        signal_bin_idx = module.where(phase_idxs & amp_idxs)[0][:n_samples]
                
        bin_cplv = module.mean(phase_diff[signal_bin_idx])   
        pair_matrix[x_phase_idx, y_phase_idx] = bin_cplv

        # bin_amp = module.mean(module.abs(take_rows_cols(data_ripples, (ch_x, ch_y), signal_bin_idx)))
        # amp_matrix[x_phase_idx, y_phase_idx] = bin_amp

    boot_idx = 0
    while boot_idx < n_boots:
        success = True
        bins_ch_x_surr = rotate_signal(bins_ch_x, module=module)
        bins_ch_y_surr = rotate_signal(bins_ch_y, module=module)

        for x_phase_idx, y_phase_idx in itertools.product(range(n_bins), range(n_bins)):
            phase_idxs_surr = (bins_ch_x_surr == x_phase_idx) & (bins_ch_y_surr == y_phase_idx)
            signal_bin_idx_surr = module.where(phase_idxs_surr & amp_idxs)[0][:n_samples]

            if len(signal_bin_idx_surr) < n_samples:
                success = False
                break

            bin_cplv_surr = module.mean(phase_diff[signal_bin_idx_surr])  
            pair_matrix_surr[boot_idx, x_phase_idx, y_phase_idx] = bin_cplv_surr
        
        boot_idx += success

    if module is cp:
        pair_matrix = cp.asarray(pair_matrix)
        pair_matrix_surr = cp.asarray(pair_matrix_surr)
        amp_matrix = cp.asarray(amp_matrix)
        count_matrix = cp.asarray(count_matrix)

    return pair_matrix, pair_matrix_surr, amp_matrix, count_matrix


def collapse_bootstrapped_surrogate(bootstrapped_surr, plv_extract_func=np.abs):
    surr_abs = plv_extract_func(bootstrapped_surr)
    surr_abs_flatten = surr_abs.reshape(surr_abs.shape[:2] + (-1,))
    surr_binned_avg = surr_abs_flatten.mean(axis=2).reshape(surr_abs.shape[:2] + (1,))
    surr_diff_abs = np.abs(surr_abs_flatten - surr_binned_avg)
    return surr_diff_abs.mean(axis=(2,1))


def collapse_binned(pairwise_binned_cplv, plv_extract_func=np.abs):
    plv_abs = plv_extract_func(pairwise_binned_cplv)
    pairwise_cplv_abs = plv_abs.reshape((len(pairwise_binned_cplv), -1))
    return np.abs(pairwise_cplv_abs - pairwise_cplv_abs.mean(axis=1).reshape((-1,1))).mean(axis=1)


def take_2d(m, indices):
    idx, idy = zip(*indices)
    return m[list(idx), list(idy)]


def make_bipolar(data_fname, montage_filename):
    raw = mne.io.read_raw_edf(data_fname, preload=False, verbose=False)
    mne.rename_channels(raw.info, lambda name: re.sub(r'(POL|SEEG)\s+', '', name).strip())

    channel_types = dict()

    for ch in raw.ch_names:
        result = re.match(r'^[A-Z][\']?\d+', ch)
        if result:
            channel_types[ch] = 'seeg'

    raw.set_channel_types(channel_types)

    montage = pd.read_csv(montage_filename, delimiter=',', names=['name', 'anode', 'cathode'])
    montage.drop_duplicates(subset='name', inplace=True)

    anode,cathode = clean_montage(raw.ch_names, montage.anode.tolist(), montage.cathode.tolist())

    raw.load_data()

    bipo = mne.set_bipolar_reference(raw, list(anode), list(cathode), copy=True, verbose=False)
    bipo = drop_monopolar_channels(bipo)
    bipo.drop_channels(bipo.info['bads'])

    picks_seeg = mne.pick_types(bipo.info, meg=False, seeg=True)

    non_seeg_chans = [ch_name for ch_idx, ch_name in enumerate(bipo.ch_names) if not(ch_idx in picks_seeg)]
    bipo.drop_channels(non_seeg_chans)

    bipo.notch_filter(np.arange(50, bipo.info['sfreq']//2, 50))

    return bipo

def main():
    cp.cuda.Device(0).use()

    for subj_num in [7,8,9,10,11]:
        base_name = 'sub-{:02d}'.format(subj_num)
        subj_path = 'data/gonogo_data/{}/ieeg'.format(base_name)
        montage_fname = os.path.join(subj_path, '{}_refTables_CW.csv'.format(base_name))
        data_fname = os.path.join(subj_path, 'sub-{}_task-gonogo_run-01_ieeg.edf'.format(subj_num))

        bipo = make_bipolar(data_fname, montage_fname)

        ref_mask = create_reference_mask(bipo).astype(int)
        ref_mask = cp.array(ref_mask)

        all_uidx = list(zip(*np.triu_indices(bipo._data.shape[0], 1)))
        all_uidx = [(i,j) for (i,j) in all_uidx if ref_mask[i,j] == 1]

        r_frequencies = np.arange(105, 455, 20)
        l_frequencies = np.arange(1,16)

        for rf in tqdm.tqdm(r_frequencies):
            data_path = os.path.join('tmp', '{}_ripples_f{}'.format(base_name, np.round(rf,2)))

            if not(os.path.exists(data_path + '.npy')):
                data_ripples = preprocess_data_morlet(bipo, rf, int(bipo.info['sfreq']), omega=7.5).astype(np.complex64)
                np.save(data_path, data_ripples)

        original_values = np.zeros((len(l_frequencies), len(r_frequencies), 100))
        surr_values = np.zeros((len(l_frequencies), len(r_frequencies), 100))

        uidx_for_frequencies = np.ndarray(shape=((len(l_frequencies), len(r_frequencies))), dtype=object)
        binned_cplv_for_frequencies = np.zeros((len(l_frequencies), len(r_frequencies), 100, 5,5), dtype=np.complex)
        binned_surrogate_for_frequencies = np.zeros((len(l_frequencies), len(r_frequencies), 100, 25, 5,5), dtype=np.complex)

        for l_freq_idx, lf in enumerate(tqdm.tqdm(l_frequencies)):
            data_theta = preprocess_data_morlet(bipo, frequency=lf, decimate_sampling=(bipo.info['sfreq']), omega=7.5).astype(np.complex64)
            data_theta = cp.asarray(data_theta)
                
            theta_cplv, theta_cplv_surr = cplv_pairwise(data_theta, return_surr=True, module=cp)

            avg_theta_surr = cp.abs(take_2d(theta_cplv_surr,  all_uidx)).mean()
            theta_significant_mask = (cp.abs(theta_cplv) > avg_theta_surr*2.42).astype(int)
            
            theta_phase = cp.angle(data_theta).astype(cp.float16)
            
            for h_freq_idx, rf in enumerate(tqdm.tqdm(r_frequencies, leave=False)):
                data_path = os.path.join('tmp', '{}_ripples_f{}.npy'.format(base_name, np.round(rf,2)))
                data_ripples =  np.load(data_path)

                data_ripples = cp.array(data_ripples)

                ripples_cplv, ripples_cplv_surr = cplv_pairwise(data_ripples, return_surr=True, module=cp)

                avg_ripples_surr = cp.abs(take_2d(ripples_cplv_surr, all_uidx)).mean()
                ripples_significant_mask = (cp.abs(ripples_cplv) > avg_ripples_surr*2.42).astype(int)

                uidx = [(i,j) for (i,j) in all_uidx if ripples_significant_mask[i,j] == 1 and theta_significant_mask[i,j] == 1]
                uidx_to_analyze = sorted(uidx, key=lambda p: np.abs(np.imag(ripples_cplv[p])), reverse=True)[:100]

                if len(uidx_to_analyze) < 100:
                    uidx = list(zip(*np.triu_indices(bipo._data.shape[0], 1)))
                    uidx = [(i,j) for (i,j) in uidx if ref_mask[i,j] == 1]

                    uidx_to_analyze = sorted(uidx, key=lambda p: np.abs(np.imag(ripples_cplv[p])), reverse=True)[:100]

                pairwise_binned_cplv, pairwise_binned_cplv_surr, _, _ = \
                            zip(*[_process_single_pair(*args) for args in
                                            tqdm.tqdm(
                                                zip_constant_argument(uidx_to_analyze, 
                                                                    theta_phase,
                                                                    data_theta, data_ripples,
                                                                    5, 25, cp), 
                                                total=len(uidx_to_analyze), leave=False)])


                original_values[l_freq_idx, h_freq_idx] = collapse_binned(pairwise_binned_cplv)
                surr_values[l_freq_idx, h_freq_idx] = collapse_bootstrapped_surrogate(pairwise_binned_cplv_surr)

                uidx_for_frequencies[l_freq_idx, h_freq_idx] = uidx_to_analyze
                binned_cplv_for_frequencies[l_freq_idx, h_freq_idx] = np.array(pairwise_binned_cplv)
                binned_surrogate_for_frequencies[l_freq_idx, h_freq_idx] = np.array(pairwise_binned_cplv_surr)

        orig_surr_diff = np.mean(original_values, axis=2) - np.mean(surr_values, axis=2)

        res_dict = {'low_frequencies': l_frequencies, 'high_frequencies': r_frequencies, 'original': original_values, 'surrogate': surr_values, 'diff': orig_surr_diff,
                    'binned_cplv': binned_cplv_for_frequencies, 'uidxes': uidx_for_frequencies, 'binned_surrogate': binned_surrogate_for_frequencies}
        res_path = os.path.join('derivatives', 'gonogo_binned_{}_results.bin'.format(base_name))
        pickle.dump(res_dict, open(res_path, 'wb'))

        fig, ax = plt.subplots(figsize=(10,7))

        im_handle = ax.imshow(orig_surr_diff, cmap='bwr', origin='lower', vmin=-1*np.max(orig_surr_diff))

        ax.set_xticks(np.arange(len(r_frequencies))[::5])
        ax.set_xticklabels(r_frequencies[::5])
        ax.set_yticks(np.arange(len(l_frequencies))[::4])
        ax.set_yticklabels(l_frequencies[::4])

        ax.set_xlabel('High frequency', fontsize=18)
        ax.set_ylabel('Low frequency', fontsize=18)
        ax.tick_params(labelsize=18)
        ax.set_title('gonogo, Averaged PLV for original data - surrogate, {}'.format(base_name), fontsize=20)
        fig.colorbar(im_handle, ax=ax)

        figname = 'gonogo_binned_{}.png'.format(base_name)
        fig_path = os.path.join('images', figname)
        fig.savefig(fig_path)

if __name__ == '__main__':
    main()