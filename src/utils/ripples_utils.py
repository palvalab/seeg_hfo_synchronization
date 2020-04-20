import re

import mne

from crosspy.preprocessing.seeg.support import clean_montage, drop_monopolar_channels

import numpy as np
import pandas as pd

def make_bipolar(data_fname, montage_filename, lowpass_frequency=None, notch=True):
    raw = mne.io.read_raw_edf(data_fname, preload=False, verbose=False)
    mne.rename_channels(raw.info, lambda name: re.sub(r'(POL|SEEG)\s+', '', name).strip())

    channel_types = dict()

    for ch in raw.ch_names:
        result = re.match(r'^[A-Z][\']?\d+', ch)
        if result:
            channel_types[ch] = 'seeg'

    raw.set_channel_types(channel_types)

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
    
    if notch:
        bipo.notch_filter(np.arange(50, bipo.info['sfreq']//2, 50), n_jobs=32)
    
    if not(lowpass_frequency is None):
        bipo.filter(None, lowpass_frequency, verbose=False, n_jobs=32)

    return bipo


def get_ez_samples_mask(windows_data, data):
    mask = np.full(data.shape[1], fill_value=True)
    
    for start, end in windows_data[['Start', 'End']].values:
        mask[start:end] = False
    
    return mask