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

def get_edf_list(edf_dir):
    edf_list = sorted(filter(lambda x: True if 'edf' in x else False, os.listdir(edf_dir)))
    
    return edf_list

def edf_reader(edf_dir, edf_fn):

    raw = mne.io.read_raw_edf(os.path.join(edf_dir, edf_fn))
        
    edf_len = timedelta(seconds=len(raw)/raw.info['sfreq']) # seconds
    edf_start = raw.info['meas_date']
    edf_end = edf_start + edf_len

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
        'edf_axis': list('chan','sample'),
    }
    
    data_dic, time = raw[:,:]
        
    return att_dic, data_dic