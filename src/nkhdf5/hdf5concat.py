"""hdf5concat_test.py

Creates new 6-min duration h5 file per single biomarker survey

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
from concatenator_tools import timestamps_to_datetime, str_to_datetime, concat_timeseries

# Main #
if __name__ == "__main__":
    ## Input Parameters 
    patient_id       = "PR06"
    stage1_path      = "/data_store0/presidio/nihon_kohden/"
    convert_edf_path = "nkhdf5/edf_to_hdf5/"
    bm_catalog = pd.read_csv(f"/data_store0/presidio/nihon_kohden/{patient_id}/{patient_id}_edf_biomarker_catalog.csv")
    rel_h5_files = bm_catalog['rel_h5_10min'].apply(ast.literal_eval)

    td = timedelta(minutes=6) #select time window to substract from start of biomarker survey
    bm_end_time = pd.to_datetime(bm_catalog['SurveyStart'])
    bm_start_time = bm_end_time - td

    bm_end_datetime = str_to_datetime(bm_end_time)
    bm_start_datetime = str_to_datetime(bm_start_time)

    ##Choose example to process
    #example_idx = 0
    #h5_files_bm = rel_h5_files[example_idx]

    ## Extract data from h5 files associated to BM and concatenate timeseries
    ##Loop through each biomarker period    
    for i in range(len(rel_h5_files)):
        concat_data = concat_timeseries(rel_h5_files[i])
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
        out_path  = pathlib.Path(f"/data_store0/presidio/nihon_kohden/{patient_id}/nkhdf5/biomarker", file_name)

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
