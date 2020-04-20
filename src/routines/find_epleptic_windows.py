
import argparse
import glob
import itertools
import json
import os
import pickle
import re
import argparse

import mne

import cupy as cp
import numpy as np
import pandas as pd

import cusignal

from bids import BIDSLayout

import tqdm

from crosspy.preprocessing.seeg.seeg_utils import create_reference_mask, get_electrode_distance
from crosspy.preprocessing.signal import preprocess_data_morlet
from crosspy.core.phase import find_event_indexes

from ..utils.ripples_utils import make_bipolar

def get_orig_frequencies():
    return np.load('gabriele/frequencies.npy')

def is_spiky(arr):
    indices = find_event_indexes(arr, True)
    if len(indices) == 0:
        return False
    
    return any([len(idx) > 3 for idx in indices])

def main(args):
    config = json.load(open(args, 'r'))

    std_ratio = config.std_ratio
    window_size = config.window_size
    affected_frequencies_ratio = config.affected_frequencies_ratio
    affected_channels_ratio = config.affected_channels_ratio

    orig_freq = get_orig_frequencies()

    root_path = config.data_root
    layout = BIDSLayout(root_path)

    for subject in layout.get(target='subject', extension='edf'): 
        subject_code = int(subject.entities['subject'])
        res_fname = os.path.join('derivatives/epleptic_mask', 'sub-{}_epleptic_samples.pickle'.format(subject.entities['subject']))

        if os.path.exists(res_fname):
            print('Subject {} is processed!'.format(subject_code))
            continue

        montage_filename = os.path.join(subject.dirname,  'sub-{}_montage.tcsv'.format(subject.entities['subject']))
        electrodes_filename = os.path.join(subject.dirname,  'sub-{}_electrodes.tcsv'.format(subject.entities['subject']))
        data_filename = subject.path

        if not(os.path.exists(montage_filename) and os.path.exists(electrodes_filename) and os.path.exists(data_filename)):
            print('Cannot find data for subject {}'.format(subject.entities['subject']))
            continue

        bipo = make_bipolar(data_filename, montage_filename, 440)

        suspicious_frequencywise = np.zeros((len(orig_freq), *bipo._data.shape), dtype=bool)

        for freq_idx, freq in enumerate(tqdm.tqdm(orig_freq)):
            data_preprocessed = preprocess_data_morlet(bipo, freq, 1000, omega=7.5, n_jobs=32)
            data_envelope = np.abs(data_preprocessed)
            
            threshold = data_envelope.mean(axis=1, keepdims=True) + std_ratio*np.std(data_envelope, axis=1, keepdims=True)
            
            suspicious_frequencywise[freq_idx] = data_envelope >= threshold
            
            for i in np.arange(0, data_envelope.shape[1], window_size):
                for ch_idx in range(data_envelope.shape[0]):
                    suspicious_frequencywise[freq_idx, ch_idx, i:i+window_size] = is_spiky(suspicious_frequencywise[freq_idx, ch_idx, i:i+window_size])

        
        frequency_mean = suspicious_frequencywise.mean(axis=0)
        channel_mean = (frequency_mean >= affected_frequencies_ratio).mean(axis=0)
        ez_mask = channel_mean >= affected_channels_ratio

        res_data = {'bad_samples': suspicious_frequencywise, 'mask': ez_mask}
        pickle.dump(res_data, open(res_fname, 'wb'), protocol=4)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute PAC spectrum for a BIDS dataset')
    parser.add_argument('--config_file', required=True, help='Name of the configuration file')

    args = parser.parse_args()

    main(args)