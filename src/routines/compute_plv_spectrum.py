"""
@author: Vladislav Myrov
"""

import os

import mne

import re
import json
import pickle
import glob

import itertools
import argparse
import tqdm

import numpy as np
import pandas as pd
import cupy as cp

import cusignal

from bids import BIDSLayout

from crosspy.core.methods import cplv, cplv_pairwise
from crosspy.preprocessing.seeg.support import clean_montage, drop_monopolar_channels
from crosspy.preprocessing.seeg.seeg_utils import create_reference_mask, get_electrode_distance
from crosspy.preprocessing.signal import preprocess_data_morlet


np.random.seed(42)
cp.random.seed(42)


def create_bipolar_ref_mask(ch_names):
    n_channels = len(ch_names)
    
    mask = np.ones((n_channels, n_channels), dtype=int)

    for i,j in itertools.product(range(n_channels), range(n_channels)):
        ch1, ch2 = ch_names[i], ch_names[j]

        ch1 = ch1.split('-')
        ch2 = ch2.split('-')

        if len(set(ch1) & set(ch2)) > 0 or i == j :
            mask[i,j] = 0
    
    return mask


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


def get_frequencies():
    return np.array(
        [2.        ,   2.26576777,   2.5668518 ,   2.90794503,
            3.29436407,   3.73213197,   4.22807216,   4.78991482,
            5.42641731,   6.14750073,   6.96440451,   7.88986164,
            8.9382971 ,  10.12605275,  11.47164198,  12.99603834,
            14.72300241,  16.67945217,  18.89588258,  21.40684088,
            24.25146506,  27.47409397,  31.12495832,  35.26096371,
            39.94657756,  45.254834  ,  51.26847217,  58.08122594,
            65.79928491,  74.54294953,  84.44850629,  95.67035191,
            110.        , 130.        , 150.        , 170.        ,
            190.        , 210.        , 230.        , 250.        ,
            270.        , 290.        , 310.        , 330.        ,
            350.        , 370.        , 390.        , 410.        ,
            430.        , 450.        ])

def is_processed(res_path):
    return os.path.exists(res_path)


def routine_gpu(data, sr, omega, morlet_frequency):
    n_chans, n_ts = data.shape

    data_gpu = cp.asarray(data)
    win = cp.array(mne.time_frequency.morlet(sr, [morlet_frequency], omega)[0])
    
    data_preprocessed = cp.zeros_like(data_gpu, dtype=cp.complex64)
    surr_data = cp.zeros_like(data_preprocessed)
    for i in range(n_chans):
        data_preprocessed[i] = cusignal.fftconvolve(data_gpu[i], win, 'same')
        data_preprocessed[i] /= cp.abs(data_preprocessed[i])

        surr_data[i] = cp.roll(data_preprocessed[i], np.random.randint(n_ts-1))
        
    plv = cp.inner(data_preprocessed, cp.conj(data_preprocessed)) / n_ts
    plv_surr = cp.inner(surr_data, cp.conj(surr_data)) / n_ts
    
    return cp.asnumpy(plv), cp.asnumpy(plv_surr)


def routine_cpu(data, sr, omega, morlet_frequency):
    n_chans, n_ts = data.shape

    data_preprocessed = mne.time_frequency.tfr_array_morlet(data[np.newaxis, ...], sr, [morlet_frequency], omega, n_jobs=32).squeeze()
    data_preprocessed /= np.abs(data_preprocessed, axis=1, keepdims=True)

    surr_data = np.zeros_like(data_preprocessed)
    for i in range(n_chans):
        surr_data[i] = np.roll(data_preprocessed[i], np.random.randint(n_ts))

    plv = np.inner(data_preprocessed, np.conj(data_preprocessed)) / n_ts
    plv_surr = np.inner(surr_data, np.conj(surr_data)) / n_ts
    
    return plv, plv_surr


def compute_psd_scaled(bipo):
    psds, freqs = mne.time_frequency.psd_welch(bipo, n_fft=2048)
    psds *= 1e3 * 1e3

    np.log10(np.maximum(psds, np.finfo(float).tiny), out=psds)
    psds *= 10
    
    return freqs, psds


def get_ez_samples_mask(windows_data, data):
    mask = np.full(data.shape[1], fill_value=True)
    
    for start, end in windows_data[['Start', 'End']].values:
        mask[start:end] = False
    
    return mask


def main(args):   
    analysis_params = json.load(open(args.config_file))
    
    epleptic_windows = pd.read_csv(analysis_params['epleptic_windows_file'])
    epleptic_windows[['Start', 'End']] = (epleptic_windows[['Start', 'End']] * 1000).astype(int)
    epleptic_windows = epleptic_windows.groupby('subject_number')

    root_path = os.path.join(analysis_params['data_path'])
    layout = BIDSLayout(root_path)

    for subject in layout.get(target='subject', extension='edf'): 
        subject_code = int(subject.entities['subject'])
        result_fname = os.path.join(analysis_params['output_path'], 'sub-{}_spectrum_bipolar_ref.pickle'.format(subject.entities['subject'])) 
        if os.path.exists(result_fname):
            print('Subject {} is processed!'.format(subject.entities['subject']))
            continue

        montage_filename = os.path.join(subject.dirname,  'sub-{}_montage_bipolar.csv'.format(subject.entities['subject']))
        electrodes_filename = os.path.join(subject.dirname,  'sub-{}_electrodes.tcsv'.format(subject.entities['subject']))
        data_filename = subject.path

        if not(os.path.exists(montage_filename) and os.path.exists(electrodes_filename) and os.path.exists(data_filename)):
            print('Cannot find data for subject {}'.format(subject.entities['subject']))
            continue

        bipo = make_bipolar(data_filename, montage_filename, analysis_params['lowpass_filter'], analysis_params['pure_bipolar'])
        psds_freqs, psds = compute_psd_scaled(bipo)

        ref_mask = create_bipolar_ref_mask(bipo.ch_names).astype(int)
        electrodes_distance = get_electrode_distance(bipo.ch_names, electrodes_filename)

        if subject_code in epleptic_windows.groups:
            subject_ez_windows = epleptic_windows.get_group(subject_code)
            subject_ez_samples_mask = get_ez_samples_mask(subject_ez_windows, bipo._data)
        else:
            subject_ez_samples_mask = np.ones(bipo._data.shape[1], dtype=bool)

        n_chans = len(bipo.ch_names)

        frequencies = get_frequencies()

        cplv_spectrum = np.zeros((len(frequencies), n_chans, n_chans), dtype=np.complex)
        cplv_surrogate = np.zeros((len(frequencies), n_chans, n_chans), dtype=np.complex)

        data_gpu = cp.array(bipo._data[:, subject_ez_samples_mask])

        for freq_idx, frequency in enumerate(tqdm.tqdm(frequencies, leave=False, desc='Subject {}'.format(subject.entities['subject']))):
            freq_cplv, freq_cplv_surr = routine_gpu(data_gpu, bipo.info['sfreq'], analysis_params['omega'], frequency)

            cplv_spectrum[freq_idx] = freq_cplv*ref_mask
            cplv_surrogate[freq_idx] = freq_cplv_surr*ref_mask

        res = {'frequencies': frequencies, 'ch_names': bipo.ch_names,
                'cplv_spectrum': cplv_spectrum, 'surrogate_spectrum': cplv_surrogate, 
                'reference_mask': ref_mask, 'electrodes_distance': electrodes_distance, 
                'analysis_parameters': analysis_params,
                'psd_frequencies': psds_freqs, 'psds': psds}
    
        pickle.dump(res, open(result_fname, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute PLV spectrum for a BIDS dataset')
    parser.add_argument('--config_file', required=True, help='Name of the configuration file')
    parser.add_argument('--cuda', default=0, type=int, help='Name of the configuration file')

    args = parser.parse_args()

    main(args)
