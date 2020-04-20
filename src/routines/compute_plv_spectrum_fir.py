"""
@author: Vladislav Myrov
"""

import os

import mne

import re
import json
import pickle
import glob

import argparse
import tqdm

import numpy as np
import pandas as pd
import cupy as cp

from bids import BIDSLayout

from crosspy.core.methods import cplv, cplv_pairwise
from crosspy.preprocessing.seeg.support import clean_montage, drop_monopolar_channels
from crosspy.preprocessing.seeg.seeg_utils import create_reference_mask, get_electrode_distance
from crosspy.preprocessing.signal import preprocess_data_morlet

from ..utils.ripples_utils import get_ez_samples_mask, make_bipolar, compute_psd_scaled

np.random.seed(42)
cp.random.seed(42)


def get_frequencies():
    return np.arange(25,500,50)

def is_processed(res_path):
    return os.path.exists(res_path)


def routine_cpu(bipo, sr, frequency, ez_mask):
    data = bipo.copy()

    hp_cut = frequency - 10
    lp_cut = frequency + 10 

    data.filter(hp_cut, lp_cut, l_trans_bandwidth=15, h_trans_bandwidth=15, n_jobs = 32, filter_length='1s')
    data.apply_hilbert(n_jobs=32) 

    data_preprocessed = data._data[:, ez_mask]

    n_chans, n_ts = data_preprocessed.shape
    
    return cplv_pairwise(data_preprocessed, return_surr=True)


def main(args):   
    analysis_params = json.load(open(args.config_file))
    
    epleptic_windows = pd.read_csv(analysis_params['epleptic_windows_file'])
    epleptic_windows[['Start', 'End']] = (epleptic_windows[['Start', 'End']] * 1000).astype(int)
    epleptic_windows = epleptic_windows.groupby('subject_number')

    root_path = os.path.join(analysis_params['data_path'])
    layout = BIDSLayout(root_path)

    for subject in layout.get(target='subject', extension='edf'): 
        subject_code = int(subject.entities['subject'])
        result_fname = os.path.join(analysis_params['output_path'], 'sub-{}_spectrum.pickle'.format(subject.entities['subject'])) 

        montage_filename = os.path.join(subject.dirname,  'sub-{}_montage.tcsv'.format(subject.entities['subject']))
        electrodes_filename = os.path.join(subject.dirname,  'sub-{}_electrodes.tcsv'.format(subject.entities['subject']))
        data_filename = subject.path

        if not(os.path.exists(montage_filename) and os.path.exists(electrodes_filename) and os.path.exists(data_filename)):
            print('Cannot find data for subject {}'.format(subject.entities['subject']))
            continue

        bipo = make_bipolar(data_filename, montage_filename, analysis_params['lowpass_filter'], analysis_params['pure_bipolar'])

        ref_mask = create_reference_mask(bipo).astype(int)
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

        for freq_idx, frequency in enumerate(tqdm.tqdm(frequencies, leave=False, desc='Subject {}'.format(subject.entities['subject']))):
            freq_cplv, freq_cplv_surr = routine_cpu(bipo, bipo.info['sfreq'], frequency, subject_ez_samples_mask)

            cplv_spectrum[freq_idx] = freq_cplv*ref_mask
            cplv_surrogate[freq_idx] = freq_cplv_surr*ref_mask

        res = {'frequencies': frequencies, 
                'cplv_spectrum': cplv_spectrum, 'surrogate_spectrum': cplv_surrogate, 
                'reference_mask': ref_mask, 'electrodes_distance': electrodes_distance, 
                'analysis_parameters': analysis_params}
        
        pickle.dump(res, open(result_fname, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute PLV spectrum for a BIDS dataset')
    parser.add_argument('--config_file', required=True, help='Name of the configuration file')
    parser.add_argument('--cuda', default=0, type=int, help='Name of the configuration file')

    args = parser.parse_args()

    main(args)
