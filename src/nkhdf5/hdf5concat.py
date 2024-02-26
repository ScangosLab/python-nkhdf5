"""hdf5concat.py

Creates new 6-min duration h5 file preceding each biomarker survey, accounts for duplicated timestamps

"""

# Package Header #
#from .header import *

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

# Third-Party Packages #
from nkhdf5 import hdf5nk

# Local Packages #
HDF5NK = hdf5nk.HDF5NK_0_1_0 
from concatenator_tools import FilesForBiomarker, timestamps_to_datetime, str_to_datetime, concat_timeseries

# Main #
if __name__ == "__main__":
    # Input Parameters  
    patient_id       = "PR03"
    stage1_path = "/data_store0/presidio/nihon_kohden"
    # Input/output directories
    inpath = pathlib.Path(stage1_path, patient_id, "nkhdf5/edf_to_hdf5")
    outpath = pathlib.Path(stage1_path, patient_id, "nkhdf5/biomarker")
    ## custom csv file indicating subset of HDF5 files that will be concatenated for each biomarker survey
    EDF_CATALOG = pd.read_csv(f"{stage1_path}/{patient_id}/{patient_id}_edf_catalog.csv")
    BIOMARKER_CATALOG = pd.read_csv(f"{stage1_path}/{patient_id}/clinical_scores/BiomarkerSurveys.csv")
    BiomarkerSurveyTimes = pd.to_datetime(BIOMARKER_CATALOG['SurveyStart'])

    FilesToConcat = FilesForBiomarker(10, 'HDF5', BiomarkerSurveyTimes, EDF_CATALOG)

    ## select time window to substract from start of biomarker survey, indirectly setting duration of biomarker recording
    ## if longer/shorter recordings are needed, adjust timedelta accordingly 
    td = timedelta(minutes=6) 
    bm_end_time = BiomarkerSurveyTimes
    bm_start_time = bm_end_time - td

    bm_end_datetime = str_to_datetime(bm_end_time)
    bm_start_datetime = str_to_datetime(bm_start_time)

    ## Extract data from h5 files associated to BM and concatenate timeseries
    ##Loop through each biomarker period    
    for i in range(len(FilesToConcat)):
        concat_data = concat_timeseries(inpath, FilesToConcat[i])
    ##Get concatenated timestamps as datetime objects
        timestamps_as_datetime = timestamps_to_datetime(concat_data['time_array'])
    
        start_rec = bm_start_datetime[i]
        end_rec = bm_end_datetime[i]
        new_time_array = []
        new_data_array = []
        for j in range(len(timestamps_as_datetime)):
            if start_rec<=timestamps_as_datetime[j]<=end_rec:
                new_time_array.append(concat_data['time_array'][j])
                new_data_array.append(concat_data['data_array'][j])
                
        new_time_array = np.array(new_time_array)
        new_data_array = np.array(new_data_array)

        bm_num = str(i + 1).zfill(4) 
        file_name = f"sub-{patient_id}_task-biomarker_{bm_num}_ieeg.h5"
        out_path  = pathlib.Path(outpath, file_name)

        ## Start of the actual code ##
        print("creating: ", file_name)
        print("")
        # Create the file #
        f_obj = HDF5NK(file=out_path, mode="a", create=True, construct=True)
        f_obj.attributes["subject_id"] = patient_id
        f_obj.attributes["start"] = int(time.mktime(bm_start_datetime[i].timetuple())*1e9)
        f_obj.attributes["end"] = int(time.mktime(bm_end_datetime[i].timetuple())*1e9)
    
        file_data_ieeg = f_obj["data_ieeg"]
        file_data_ieeg.append(new_data_array, component_kwargs={"timeseries": {"data": new_time_array}})
        file_data_ieeg.axes[1]["channellabel_axis"].append(concat_data["channellabel_axis"])
        file_data_ieeg.axes[1]["channelcoord_axis"].append(concat_data["channelcoord_axis"])

        file_data_ieeg.attributes["filter_lowpass"]  = concat_data["filter_lowpass"]
        file_data_ieeg.attributes["filter_highpass"] = concat_data["filter_highpass"]
        file_data_ieeg.attributes["channel_count"]   = concat_data["channel_count"]
        file_data_ieeg.axes[0]["time_axis"].attributes["sample_rate"] = concat_data["sample_rate"] 
        file_data_ieeg.axes[0]["time_axis"].attributes["time_zone"] = concat_data["time_zone"] 

        print("File after appending:")
        print("")
        print("ieeg data size: ", f_obj["data_ieeg"].shape)
        print("ieeg time axis size: ", f_obj["data_ieeg"].axes[0]["time_axis"].shape)
    #print("ieeg channel labels axis size: ", f_obj["data_ieeg"].axes[1]["channellabel_axis"].shape)
    #print("ieeg channel coordinates axis size: ", f_obj["data_ieeg"].axes[1]["channelcoord_axis"].shape)
    #print("f_obj['data_ieeg'].axes[1]['channellabel_axis']: ", f_obj["data_ieeg"].axes[1]["channellabel_axis"][...])
        print("")

    # After closing check if the file exists #
        print(f"File Exists: {out_path.is_file()}")
        print(f"File is Openable: {HDF5NK.is_openable(out_path)}")
        print("")

        f_obj.close()


"""End of code

"""
