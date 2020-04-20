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

import cusignal

import scipy as sp
import scipy.signal

import json

import argparse

def create_surrogate(data):
    res = cp.zeros_like(data)
    for i in range(data.shape[0]):
        idx = np.random.randint(data.shape[1]-1)
        res[i] = cp.roll(data[i], idx)
    
    return res

def create_surrogate_inplace(data):
    for i in range(data.shape[0]):
        idx = np.random.randint(data.shape[1]-1)
        data[i] = cp.roll(data[i], idx)


def make_bipolar(data_fname, montage_filename, lowpass_frequency):
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

    bipo.notch_filter(np.arange(50, bipo.info['sfreq']//2, 50), n_jobs=32)
    bipo.filter(None, lowpass_frequency, verbose=False, n_jobs=32)

    return bipo


def get_ez_mask(data_ref, ez_set):
    ez_mask = [int(ch_name.split('-')[0] in ez_set) for ch_name in data_ref.ch_names]
    return np.array(ez_mask)


def filter_morlet_gpu(data, sr, omega, morlet_frequency):
    n_chans, n_ts = data.shape

    data_gpu = cp.asarray(data)
    win = cp.array(mne.time_frequency.morlet(sr, [morlet_frequency], omega)[0])
    
    data_preprocessed = cp.zeros_like(data_gpu, dtype=cp.complex64)
    for i in range(n_chans):
        data_preprocessed[i] = cusignal.fftconvolve(data_gpu[i], win, 'same')
    
    return data_preprocessed


def get_ez_samples_mask(windows_data, data):
    mask = np.full(data.shape[1], fill_value=True)
    
    for start, end in windows_data[['Start', 'End']].values:
        mask[start:end] = False
    
    return mask


def main(args):   
    analysis_params = json.load(open(args.config_file))

    high_freqs = np.logspace(2.35, 9, base=2, num=20, endpoint=False)
    low_freqs = np.logspace(1, 6.75, base=2, num=15, endpoint=False)

    epleptic_windows = pd.read_csv(analysis_params['epleptic_windows_file'])
    epleptic_windows[['Start', 'End']] = (epleptic_windows[['Start', 'End']] * 1000).astype(int)
    epleptic_windows = epleptic_windows.groupby('subject_number')

    root_path = os.path.join('../seeg_phases/data', 'SEEG_redux_BIDS')
    layout = BIDSLayout(root_path)

    for subject in layout.get(target='subject', extension='edf'): 
        subject_code = int(subject.entities['subject'])
        res_fname = os.path.join('derivatives', 'pac_no_ez', 'pac_sub_{}.pickle'.format(subject.entities['subject']))

        if os.path.exists(res_fname):
            print('Subject {} is processed!'.format(subject.entities['subject']))
            continue

        print('Working with subject {}'.format(subject.entities['subject']))

        montage_filename = os.path.join(subject.dirname,  'sub-{}_montage.tcsv'.format(subject.entities['subject']))
        data_filename = subject.path

        if not(os.path.exists(montage_filename) and os.path.exists(data_filename)):
            print('Cannot find data for subject {}'.format(subject.entities['subject']))
            continue
        
        bipo = make_bipolar(data_filename, montage_filename, analysis_params['lowpass_filter'])
        ref_mask = create_reference_mask(bipo)
        ref_mask_gpu = cp.array(ref_mask).astype(int)

        if subject_code in epleptic_windows.groups:
            subject_ez_windows = epleptic_windows.get_group(subject_code)
            subject_ez_samples_mask = get_ez_samples_mask(subject_ez_windows, bipo._data)
        else:
            subject_ez_samples_mask = np.ones(bipo._data.shape[1], dtype=bool)

        n_chans = len(bipo.ch_names)

        phase_amplitude_correlation = np.full(shape=(len(high_freqs), len(low_freqs), n_chans, n_chans), fill_value=np.nan)    
        phase_amplitude_surrogates = np.full(shape=(len(high_freqs), len(low_freqs), n_chans, n_chans), fill_value=np.nan) 

        print('Subject {} has {:.2f} a total of {} of ez samples!'.format(subject.entities['subject'], 1 - subject_ez_samples_mask.mean(), subject_ez_samples_mask.shape[0] - subject_ez_samples_mask.sum()))

        data_gpu = cp.array(bipo._data[:, subject_ez_samples_mask])   

        print('Subject {} has {} data shape'.format(subject.entities['subject'], data_gpu.shape))

        for low_f in tqdm.tqdm(low_freqs, desc='Preprocessing...', leave=False):
            low_fname = 'temp/sub_{}_freq_{:.2f}.npy'.format(subject.entities['subject'], low_f)
            if os.path.exists(low_fname):
                continue
                
            data_low_f = filter_morlet_gpu(data_gpu, bipo.info['sfreq'], analysis_params['omega'], low_f)
            data_low_f /= cp.abs(data_low_f)
            data_low_f = cp.conj(data_low_f)

            cp.save(low_fname, data_low_f)
        
        data_low_f = None
    
        sr = bipo.info['sfreq']
        bar = tqdm.tqdm(total=225, leave=False)
        for high_idx, high_f in enumerate(high_freqs):
            
            # data_high_freq = filter_morlet_gpu(data_gpu, sr, analysis_params['omega'], high_f)
            # high_amp = cp.abs(data_high_freq)
            high_amp = cp.abs(filter_morlet_gpu(data_gpu, sr, analysis_params['omega'], high_f))

            for low_idx, low_f in enumerate(low_freqs):
                if low_f > high_f:
                    break

                low_fname = 'temp/sub_{}_freq_{:.2f}.npy'.format(subject.entities['subject'], low_f)
                slow_conj = cp.load(low_fname)
                
                high_amp_complex = filter_morlet_gpu(high_amp, sr, analysis_params['omega'], low_f)
                high_amp_complex /= cp.abs(high_amp_complex)

                corr = cp.inner(high_amp_complex, slow_conj) / slow_conj.shape[1]

                create_surrogate_inplace(high_amp_complex)
                              
                corr_surr = cp.inner(high_amp_complex, slow_conj) / slow_conj.shape[1]
                
                phase_amplitude_correlation[high_idx, low_idx] = cp.asnumpy(cp.abs(corr)*ref_mask_gpu)
                phase_amplitude_surrogates[high_idx, low_idx] = cp.asnumpy(cp.abs(corr_surr)*ref_mask_gpu)
                                
                bar.update(1)

        bar.close()

        res = {'phase_amplitude_correlation': phase_amplitude_correlation, 'surrogate': phase_amplitude_surrogates, 'high_frequencies': high_freqs, 'low_frequencies': low_freqs, 'ref_mask': ref_mask, 'ch_names': bipo.ch_names}
        pickle.dump(res, open(res_fname, 'wb'))

        for low_f in tqdm.tqdm(low_freqs, desc='Preprocessing...', leave=False):
            low_fname = 'temp/sub_{}_freq_{:.2f}.npy'.format(subject.entities['subject'], low_f)
            if os.path.exists(low_fname):
                os.remove(low_fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute PLV spectrum for a BIDS dataset')
    parser.add_argument('--config_file', required=True, help='Name of the configuration file')

    args = parser.parse_args()

    main(args)


