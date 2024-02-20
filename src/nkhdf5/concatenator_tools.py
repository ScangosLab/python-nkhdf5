"""concatenator_tools.py

Define custom functions used in hdf5concat.py

"""

# Standard Libraries #
import pathlib
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
import time
import h5py
import scipy.io
import os
import ast

# Inputs #

patient_id   = "PR03"
stage1_path  = "/data_store0/presidio/nihon_kohden/"
convert_edf_path = "nkhdf5/edf_to_hdf5/"

# Custom Functions #

##Find subset of files (EDF or HDF5) needed to create biomarker recordings
def FilesForBiomarker(FinalDuration, FileFormat, SurveyTimes, DataFrame):
    FilesForBiomarkerList = []
    td = timedelta(minutes=FinalDuration)
    StartRec = SurveyTimes - td
    times1 = DataFrame.loc[:,['edf_name', 'h5_name', 'edf_start']]
    times2 = DataFrame.loc[:,['edf_name', 'h5_name', 'edf_end']].rename(columns={'edf_end':'edf_start'})
    TargetTimes = pd.concat([times1, times2], ignore_index=True)
    TargetTimes = TargetTimes.sort_values(by='edf_start').reset_index(drop=True)
    for i in range(len(SurveyTimes)):
        mask = (TargetTimes['edf_start'] >= StartRec[i]) & (TargetTimes['edf_start'] <= SurveyTimes[i])
        if FileFormat == 'EDF':
            FilesForBiomarkerList.append(sorted(list(TargetTimes['edf_name'].loc[mask].unique())))
        if FileFormat == 'HDF5':
            FilesForBiomarkerList.append(sorted(list(TargetTimes['h5_name'].loc[mask].unique())))
    return FilesForBiomarkerList

##Converts list of timestamps to list of datetime objects
def timestamps_to_datetime(timestamps_list):
    abs_timestamps = []
    for i in range(len(timestamps_list)):
        abs_timestamps.append(datetime.datetime.fromtimestamp(timestamps_list[i]/1e9))
    return abs_timestamps

##Converts list of objects to strings and then to datetime
def str_to_datetime(list_of_values):
    to_datetime = [datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S') for x in list_of_values]
    return to_datetime

#def str_to_datetime(object_value):
#    to_datetime = datetime.datetime.strptime(str(object_value), '%Y-%m-%d %H:%M:%S')
#    return to_datetime

##Concatenates timeseries given a list of h5 files associated to biomarker survey
##Also gets metadata shared by h5 files 
def concat_timeseries(h5_in_bm):
    data_array = []
    time_array = []
    for i in range(len(h5_in_bm)):
        file_path = pathlib.Path(stage1_path,patient_id,convert_edf_path,h5_in_bm[i])
        file_obj = h5py.File(file_path, 'r')
        data_array = data_array + list(np.array(file_obj['intracranialEEG']))
        time_array = time_array + list(np.array(file_obj['intracranialEEG_time_axis']))
    
    channellabel_axis = np.array(file_obj['intracranialEEG_channellabel_axis'])
    channelcoord_axis = np.array(file_obj['intracranialEEG_channelcoord_axis'])
    channel_count = file_obj['intracranialEEG'].attrs['channel_count']
    filter_highpass = file_obj['intracranialEEG'].attrs['filter_highpass']
    filter_lowpass = file_obj['intracranialEEG'].attrs['filter_lowpass']
    sample_rate = file_obj['intracranialEEG_time_axis'].attrs['sample_rate']
    time_zone = file_obj['intracranialEEG_time_axis'].attrs['time_zone']

    timeseries_dict = {
        'data_array' : data_array,
        'time_array' : time_array,
        'channellabel_axis' : channellabel_axis,
        'channelcoord_axis' : channelcoord_axis,
        'channel_count' : channel_count,
        'filter_highpass' : filter_highpass,
        'filter_lowpass' : filter_lowpass,
        'sample_rate' : sample_rate,
        'time_zone' : time_zone
    }

    return timeseries_dict

"""

End of code

"""
