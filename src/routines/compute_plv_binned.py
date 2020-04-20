"""
@author: Vladislav Myrov
"""

import argparse
import glob
import itertools
import json
import os
import pickle
import re

import mne

import cupy as cp
import numpy as np
import pandas as pd

import cusignal

from bids import BIDSLayout

import tqdm

from crosspy.preprocessing.seeg.seeg_utils import create_reference_mask, get_electrode_distance
from crosspy.preprocessing.signal import preprocess_data_morlet

from ripples_utils import make_bipolar, get_ez_samples_mask

from crosspy.core.phase import find_event_indexes

np.random.seed(42)
cp.random.seed(42)


def get_frequencies():
    return np.array(
        [
            110.        , 130.        , 150.        , 170.        ,
            190.        , 210.        , 230.        , 250.        ,
            270.        , 290.        , 310.        , 330.        ,
            350.        , 370.        , 390.        , 410.        ,
            430.        , 450.        
            ])


# median is really bad to compute on GPU but its better than copy data from GPU to CPU and back
def cupy_median(x, axis=-1):
    xp = cp.get_array_module(x)
    n = x.shape[axis]
    s = xp.sort(x, axis)
    m_odd = xp.take(s, n // 2, axis)
    
    if n % 2 == 1:
        return m_odd
    else: 
        m_even = xp.take(s, n // 2 - 1, axis)
        return (m_odd + m_even) / 2


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


class PLVProcessor(object):
    def __init__(self, frequencies: np.ndarray, number_of_bins: int, sfreq: int, omega: float, show_progress: bool=True):
        # analysis parameters
        self.frequencies = frequencies

        self.n_bins = number_of_bins
        self.sfreq = sfreq
        self.omega = omega
        
        #output result
        self.cplv_spectrum = None
        self.number_of_samples = None

        #verbose parameters
        self.show_progress = show_progress

    def get_results(self):
        if self.cplv_spectrum is None:
            raise RuntimeError('You need to compute results first!')
            
        return self.cplv_spectrum, self.number_of_samples

   
    def process_data(self, data: np.ndarray, reference_mask: np.ndarray):
        self.data = cp.array(data)
        self.reference_mask = reference_mask

        n_chans, n_ts = self.data.shape

        # allocate data on CPU for the main result
        self.cplv_spectrum = np.zeros((len(self.frequencies), self.n_bins, self.n_bins, n_chans, n_chans), dtype=np.complex)
        self.number_of_samples = np.zeros((len(self.frequencies), self.n_bins, self.n_bins, n_chans, n_chans), dtype=int)

        # allocate data on GPU as temporary buffers to avoid extra data allocation
        # also cache conjuguate
        self.data_preprocessed = cp.zeros_like(self.data, dtype=cp.complex64)
        self.data_amplitude_labels = cp.zeros_like(self.data, dtype=cp.uint8)
        self.data_thresholded = cp.zeros_like(self.data, dtype=cp.bool)

        # buffers used to store temporary data to avoid extra data allocation & improve performance and memory stability
        self.frequency_plv = cp.zeros((self.n_bins, self.n_bins, n_chans, n_chans), dtype=cp.complex64)
        self.frequency_samples = cp.zeros((self.n_bins, self.n_bins, n_chans, n_chans), dtype=cp.int32)
        self.mask_buffer = cp.zeros((2*self.n_bins + 1, n_ts), dtype=bool)

        for freq_idx, frequency in enumerate(tqdm.tqdm(self.frequencies, disable=not(self.show_progress))):
            self.cplv_spectrum[freq_idx], self.number_of_samples[freq_idx] = self._process_single_frequency(frequency)

    def _filter_data(self, frequency: float):
        n_chans = self.data.shape[0]
        amplitude_percentiles = cp.linspace(0, 100, self.n_bins + 1)

        win = cp.array(mne.time_frequency.morlet(self.sfreq, [frequency], self.omega)[0])

        data_envelope = cp.zeros_like(self.data)
        for i in range(n_chans):
            self.data_preprocessed[i] = cusignal.fftconvolve(self.data[i], win, 'same')
            data_envelope[i] = cp.abs(self.data_preprocessed[i])

            # normalize analog signal amplitude to make possible to compute PLV through inner product with conjugate
            self.data_preprocessed[i] /= data_envelope[i]
            # normalize signal envelope to make it comparable between different contacts 
            data_envelope[i] /= cupy_median(data_envelope[i])
            self.data_thresholded[i] = data_envelope[i] <= (cupy_median(data_envelope[i])*2)
            # self.data_thresholded[i] = True
        
        # for i in range(n_chans):
        #     amplitude_bins = cp.percentile(data_envelope[i][self.data_thresholded[i]], amplitude_percentiles)
        #     digitize_cupy(data_envelope[i], amplitude_bins, out=self.data_amplitude_labels[i])

        amplitude_bins = cp.percentile(data_envelope[self.data_thresholded], amplitude_percentiles)
        digitize_cupy(data_envelope, amplitude_bins, out=self.data_amplitude_labels)

        self.data_amplitude_labels -= 1

        # deleting envelope to save some space
        # I am jogging here with conjugate and envelope memory because we dont need conjugate during preprocessing
        # and dont need envelope after preprocessing
        # del data_envelope
        # data_envelope = None
        self.data_envelope = data_envelope

        self.data_conj = cp.zeros_like(self.data_preprocessed)
        cp.conj(self.data_preprocessed, out = self.data_conj)

    
    def _process_single_frequency(self, frequency: float):
        n_chans = self.data.shape[0]

        self._filter_data(frequency)

        for x in range(n_chans):
            for y in range(x+1, n_chans):
                if self.reference_mask[x, y] == 0:
                    continue

                self._process_pair(x, y)

        # delete conjugate to free some space
        # otherwise it may crash due to OOM because we need memory for signal envelope during preprocessing
        del self.data_conj
        self.data_conj = None

        return cp.asnumpy(self.frequency_plv), cp.asnumpy(self.frequency_samples)
    

    def _process_pair(self, x: int, y: int):
        n_bins = self.n_bins
        
        # compute mask of amplitude quantile for each sample in the data
        for i in range(n_bins):           
            self.mask_buffer[i] = (self.data_amplitude_labels[x] == i) & (self.data_thresholded[x])
            self.mask_buffer[i + n_bins] = (self.data_amplitude_labels[y] == i) & (self.data_thresholded[y])

        # inner product of those masks is the same as compute sum of logical for each pair but without cycles => faster
        # however I dont like bool -> int type casting
        self.frequency_samples[:, :, x, y] = cp.inner(self.mask_buffer[:n_bins].astype(int), self.mask_buffer[n_bins:~0].astype(int))
        self.frequency_samples[:, :, y, x] = self.frequency_samples[:,:, x, y]

        min_count = int(self.frequency_samples[:,:, x, y].min())

        data_x = self.data_preprocessed[x]
        data_y = self.data_conj[y]

        for i, j in itertools.product(range(n_bins), range(n_bins)):
            cp.logical_and(self.mask_buffer[i], self.mask_buffer[j + n_bins], out=self.mask_buffer[~0])

            # select data according to amplitude mask of both channels and truncate it to avoid PLV bias
            vals_x = data_x[self.mask_buffer[~0]][:min_count]
            vals_y = data_y[self.mask_buffer[~0]][:min_count]

            # just in case there are some label combinations without any samples; unluckly though
            # if vals_x.shape[0] == 0 or vals_y.shape[0] == 0:
            #     continue
            
            self.frequency_plv[i, j, x, y] = self.frequency_plv[i, j, y, x] = cp.inner(vals_x, vals_y) / min_count


def main(args):   
    analysis_params = json.load(open(args.config))

    cp.cuda.Device(args.cuda).use()
    
    frequencies = get_frequencies()

    root_path = os.path.join(analysis_params['data_path'])
    layout = BIDSLayout(root_path)

    for subject in layout.get(target='subject', extension='edf'): 
        subject_code = int(subject.entities['subject'])

        if subject_code % 2 == args.cuda:
            print('Subject {} is for another GPU!'.format(subject.entities['subject']))
            continue

        result_fname = os.path.join(analysis_params['output_path'], 'sub-{}_spectrum_binned_3.pickle'.format(subject.entities['subject'])) 
        if os.path.exists(result_fname):
            print('Subject {} is processed!'.format(subject.entities['subject']))
            continue

        montage_filename = os.path.join(subject.dirname,  'sub-{}_montage.tcsv'.format(subject.entities['subject']))
        electrodes_filename = os.path.join(subject.dirname,  'sub-{}_electrodes.tcsv'.format(subject.entities['subject']))
        data_filename = subject.path
        bad_filename = os.path.join('derivatives/epleptic_mask', 'sub-{}_epleptic_samples.pickle'.format(subject.entities['subject']))

        if not(os.path.exists(montage_filename) and os.path.exists(electrodes_filename) and os.path.exists(data_filename)):
            print('Cannot find data for subject {}'.format(subject.entities['subject']))
            continue

        bipo = make_bipolar(data_filename, montage_filename, analysis_params['lowpass_filter'])

        ref_mask = create_reference_mask(bipo).astype(int)
        electrodes_distance = get_electrode_distance(bipo.ch_names, electrodes_filename)

        bad_data = pickle.load(open(bad_filename, 'rb'))
        bad_samples = bad_data['mask']
        bad_samples[:args.bad_samples] = True

        good_samples = np.logical_not(bad_samples)

        print('Good samples: {:.2f}', good_samples.mean(), bipo._data.shape)

        processor = PLVProcessor(frequencies, analysis_params['number_of_bins'], int(bipo.info['sfreq']), analysis_params['omega'])
        processor.process_data(bipo._data[:, good_samples], ref_mask)

        res = {'frequencies': frequencies, 
                'cplv_spectrum': processor.cplv_spectrum, 'number_of_samples': processor.number_of_samples,
                'reference_mask': ref_mask, 'electrodes_distance': electrodes_distance, 
                'analysis_parameters': analysis_params}
        pickle.dump(res, open(result_fname, 'wb'))

        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute PLV spectrum for a BIDS dataset')
    parser.add_argument('--config', default='binned_spectrum_config.json', required=False, help='Name of the configuration file')
    parser.add_argument('--cuda', default=0, type=int, help='Index of CUDA devices to be used')

    args = parser.parse_args()

    main(args)
