'''
Saves a metadata catalog csv of filename, start date, end date, shortened filename and file
lenght to the same directory level as the EDF directory provided.
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dill as pickle

import subprocess

import mne

import re

from datetime import datetime, timedelta

#Define common labels for channel type
ieeg_chan = ['OFC', 'SGC', 'R A', 'L A', 'R H', 'L H', 'VC']
dc_chan   = ['DC']
ekg_chan  = ['EKG']
emg_chan  = ['EMG']

#Gets list of all EDF files in patient's main folder 
def get_edf_list(edf_dir):
    edf_list = sorted(filter(lambda x: True if 'edf' in x else False, os.listdir(edf_dir)))
    
    return edf_list

#Gets metadeta and timeseries of EDF file within extracted list
def edf_reader(edf_dir, edf_fn):

    raw = mne.io.read_raw_edf(os.path.join(edf_dir, edf_fn))
        
    edf_len = timedelta(seconds=len(raw)/raw.info['sfreq']) # seconds
    edf_start = raw.info['meas_date']
    edf_end = edf_start + edf_len
    ch_names_clean = [ch.split('-')[0].split('POL ')[1] for ch in raw.ch_names]
    
    def find_indices(lst, condition):
        return [i for i, elem in enumerate(lst) if condition(elem)]
    
    def find_chantype(chan_list, index_list):
        for i in range(len(chan_list)):
            index_list = index_list + find_indices(ch_names_clean, lambda e: True if chan_list[i] in e else False)
        return index_list

    ieeg_idx = find_chantype(ieeg_chan, [])
    dc_idx = find_chantype(dc_chan, [])
    ekg_idx = find_chantype(ekg_chan, [])
    emg_idx = find_chantype(emg_chan, [])
    not_scalp_idx = ieeg_idx + dc_idx + ekg_idx + emg_idx
    scalp_idx = list(set(list(range(len(ch_names_clean)))) - set(not_scalp_idx))
    
    chantype = ch_names_clean

    for i in range(len(ieeg_idx)):
        chantype[ieeg_idx[i]] = 'intracranial EEG'
    for i in range(len(dc_idx)):
        chantype[dc_idx[i]] = 'TTL'
    for i in range(len(ekg_idx)):
        chantype[ekg_idx[i]] = 'EKG'
    for i in range(len(emg_idx)):
        chantype[emg_idx[i]] = 'EMG'
    for i in range(len(scalp_idx)):
        chantype[scalp_idx[i]] = 'scalp EEG'

    att_dic = {
        'edf_fn': edf_fn,
        'edf_path': os.path.join(edf_dir, edf_fn),
        'edf_start': edf_start.replace(tzinfo=None), 
        'edf_end': edf_end.replace(tzinfo=None),
        'edf_timezone': 'US/Pacific',
        'edf_duration': edf_len,
        'edf_nsample': len(raw),
        'edf_sfreq': raw.info['sfreq'],
        'edf_lowpass': raw.info['lowpass'],
        'edf_highpass': raw.info['highpass'],
        'edf_nchan': raw.info['nchan'],
        'edf_channame': raw.info['ch_names'],
        'edf_chantype': chantype,
        'edf_axis': list(['chan','sample']),
    }
    
    data_dic, time = raw[:,:]
        
    return att_dic, data_dic