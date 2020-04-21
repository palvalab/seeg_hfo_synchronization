import re
import itertools 

import mne

from crosspy.preprocessing.seeg.support import clean_montage, drop_monopolar_channels

import cupy as cp
import numpy as np
import pandas as pd


def get_bipolar_contacts(channel_names):
    anode = list()
    cathode = list()
    dropList = list()
    for name in channel_names:
        if re.match(r'^[A-Z][\']?[0-9]+$',name):
            splitLabel = re.split(r'([0-9]+)',name)
            if splitLabel:
                cathodeName = splitLabel[0]+str(int(splitLabel[1])+1)
                if cathodeName in channel_names:
                    anode.append(name)
                    cathode.append(cathodeName) 
        else:
            dropList.append(name)
    return (anode, cathode)


def make_bipolar(data_fname, montage_filename, lowpass_frequency, pure_bipolar=False):
    raw = mne.io.read_raw_edf(data_fname, preload=False, verbose=False)
    mne.rename_channels(raw.info, lambda name: re.sub(r'(POL|SEEG)\s+', '', name).strip())

    channel_types = dict()

    for ch in raw.ch_names:
        result = re.match(r'^[A-Z][\']?\d+', ch)
        if result:
            channel_types[ch] = 'seeg'

    raw.set_channel_types(channel_types)

    if pure_bipolar:
        print('Print bipolar!')
        anode, cathode = get_bipolar_contacts(raw.ch_names)
    else:
        montage = pd.read_csv(montage_filename, delimiter='\t')
        montage.drop_duplicates(subset='name', inplace=True)

        anode,cathode = clean_montage(raw.ch_names, montage.anode.tolist(), montage.cathode.tolist())

    raw.load_data()

    bipo = mne.set_bipolar_reference(raw, list(anode), list(cathode), copy=True, verbose=False)
    bipo = drop_monopolar_channels(bipo)
    bipo.drop_channels(bipo.info['bads'])

    picks_seeg = mne.pick_types(bipo.info, meg=False, seeg=True)

    non_seeg_chans = [ch_name for ch_idx, ch_name in enumerate(bipo.ch_names) if not(ch_idx in picks_seeg) or len(ch_name.split('-')) == 1]
    bipo.drop_channels(non_seeg_chans)

    bipo.notch_filter(np.arange(50, bipo.info['sfreq']//2, 50), n_jobs=32)
    bipo.filter(None, lowpass_frequency, verbose=False, n_jobs=32)

    return bipo


def get_ez_samples_mask(windows_data, data):
    mask = np.full(data.shape[1], fill_value=True)
    
    for start, end in windows_data[['Start', 'End']].values:
        mask[start:end] = False
    
    return mask


def is_monopolar(chan):
    tokens = chan.split('-')
    return len(tokens) == 1 or len(tokens[1]) == 0


def is_bipolar(x):
    tokens = x.split('-')
    
    return len(tokens) == 2 and len(tokens[1]) > 0


def baseline_zscore(arr, baseline):
    m = arr[...,:baseline].mean(axis=-1, keepdims=True)
    s = arr[...,:baseline].std(axis=-1, keepdims=True)
    
    return (arr - m)/s


def compute_psd_scaled(bipo):
    psds, freqs = mne.time_frequency.psd_welch(bipo, n_fft=2048)
    psds *= 1e3 * 1e3

    np.log10(np.maximum(psds, np.finfo(float).tiny), out=psds)
    psds *= 10
    
    return freqs, psds


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

_digitize_kernel = cp.core.ElementwiseKernel(
    'S x, raw T bins, int32 n_bins',
    'raw U y',
    '''
    int low = 0;
    if (x < bins[0]) {
        low = 0;
    } else if (bins[n_bins - 1] < x) {
        low = n_bins;
    } else {
        int high = n_bins - 1;

        while (high - low > 1) {
            int mid = (high + low) / 2;
            if (bins[mid] <= x) {
                low = mid;
            } else {
                high = mid;
            }
        }
        low += 1;
    }
    y[i] = low;
    ''')


def digitize_cupy(x, bins, out=None):
    if out is None:
        out = cp.zeros_like(x, dtype=cp.uint8)

    _digitize_kernel(x, bins, bins.size, out)
    
    return out