import mne
import re
import os

import numpy as np
import pandas as pd

from crosspy.core.methods import cplv, cplv_pairwise
from crosspy.preprocessing.seeg.support import clean_montage, drop_monopolar_channels
from crosspy.preprocessing.seeg.seeg_utils import create_reference_mask, get_electrode_distance
from crosspy.preprocessing.signal import preprocess_data_morlet

from bids import BIDSLayout

from joblib import Parallel, delayed

import pickle

import tqdm
import glob

import cupy as cp
import matplotlib.pyplot as plt

import scipy as sp
import scipy.signal

from nptdms import TdmsFile

def create_surrogate(data):
    res = np.zeros_like(data)
    for i in range(data.shape[0]):
        idx = np.random.randint(data.shape[1])
        res[i] = np.roll(data[i], idx)
    
    return res


def make_bipolar(data_fname, montage_filename):
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

    bipo.notch_filter(np.arange(50, bipo.info['sfreq']//2, 50))

    return bipo


def get_ez_mask(data_ref, ez_set):
    ez_mask = [int(ch_name.split('-')[0] in ez_set) for ch_name in data_ref.ch_names]
    return np.array(ez_mask)


def fast_hilbert(x):
    return sp.signal.hilbert(x, sp.fftpack.next_fast_len(len(x)))[:len(x)]


def main():
    high_freqs = np.logspace(2.35, 9, base=2, num=20, endpoint=False)
    low_freqs = np.logspace(1, 6.75, base=2, num=15, endpoint=False)

    omega = 5

    root_path = os.path.join('../seeg_phases/data', 'SEEG_redux_BIDS')
    layout = BIDSLayout(root_path)

    for subject in layout.get(target='subject', extension='edf'): 
        res_fname = os.path.join('derivatives', 'pac_sub_{}.pickle'.format(subject.entities['subject']))

        if os.path.exists(res_fname):
            print('Subject {} is processed!'.format(subject.entities['subject']))
            continue

        print('Working with subject {}'.format(subject.entities['subject']))

        montage_filename = os.path.join(subject.dirname,  'sub-{}_montage.tcsv'.format(subject.entities['subject']))
        data_filename = subject.path

        if not(os.path.exists(montage_filename) and os.path.exists(data_filename)):
            print('Cannot find data for subject {}'.format(subject.entities['subject']))
            continue
        
        bipo = make_bipolar(data_filename, montage_filename)
        ref_mask = create_reference_mask(bipo).astype(bool)

        n_chans = len(bipo.ch_names)

        phase_amplitude_correlation = np.full(shape=(len(high_freqs), len(low_freqs), n_chans, n_chans), fill_value=np.nan)    
        phase_amplitude_surrogates = np.full(shape=(len(high_freqs), len(low_freqs), n_chans, n_chans), fill_value=np.nan)    

        for low_f in tqdm.tqdm(low_freqs, desc='Preprocessing...', leave=False):
            low_fname = 'temp/sub_{}_freq_{:.2f}.npy'.format(subject.entities['subject'], low_f)
            if os.path.exists(low_fname):
                continue
                
            data_low_f = preprocess_data_morlet(bipo, low_f, bipo.info['sfreq'], omega=omega)
            data_low_f /= np.abs(data_low_f)
            data_low_f = np.conjugate(data_low_f)

            np.save(low_fname, data_low_f)
            
        sr = bipo.info['sfreq']
        bar = tqdm.tqdm(total=225, leave=False)
        for high_idx, high_f in enumerate(high_freqs):
            
            data_high_freq = preprocess_data_morlet(bipo, high_f, sr, omega=omega)
            high_amp = np.abs(data_high_freq)

            # high_amp_complex =  mne.time_frequency.tfr_array_morlet(high_amp[np.newaxis, ...], sr, [low_f],
            #                             omega, verbose=False, n_jobs=32).squeeze()
            # high_amp_complex = np.apply_along_axis(fast_hilbert, -1, high_amp)

            # fast_norm = high_amp_complex / np.abs(high_amp_complex)
            # fast_surr = create_surrogate(fast_norm)

            for low_idx, low_f in enumerate(low_freqs):
                if low_f > high_f:
                    break
                
                high_amp_complex =  mne.time_frequency.tfr_array_morlet(high_amp[np.newaxis, ...], sr, [low_f],
                                                        omega, verbose=False, n_jobs=32).squeeze()

                fast_norm = high_amp_complex / np.abs(high_amp_complex)
                fast_surr = create_surrogate(fast_norm)
                
                low_fname = 'temp/sub_{}_freq_{:.2f}.npy'.format(subject.entities['subject'], low_f)

                slow_conj = np.load(low_fname)
                
                corr = np.inner(fast_norm, slow_conj) / slow_conj.shape[1]
                corr_surr = np.inner(fast_surr, slow_conj) / slow_conj.shape[1]
                
                phase_amplitude_correlation[high_idx, low_idx] = np.abs(corr)*ref_mask.astype(int)
                phase_amplitude_surrogates[high_idx, low_idx] = np.abs(corr_surr)*ref_mask.astype(int)
                
                bar.update(1)

        bar.close()

        res = {'phase_amplitude_correlation': phase_amplitude_correlation, 'surrogate': phase_amplitude_surrogates, 'high_frequencies': high_freqs, 'low_frequencies': low_freqs, 'ref_mask': ref_mask, 'ch_names': bipo.ch_names}
        pickle.dump(res, open(res_fname, 'wb'))

        for low_f in tqdm.tqdm(low_freqs, desc='Preprocessing...', leave=False):
            low_fname = 'temp/sub_{}_freq_{:.2f}.npy'.format(subject.entities['subject'], low_f)
            if os.path.exists(low_fname):
                os.remove(low_fname)


if __name__ == '__main__':
    main()


