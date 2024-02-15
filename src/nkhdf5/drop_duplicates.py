"""drop_duplicates.py

Fix biomarker recordings that have duplicated timestamps with no neural data

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

# Main #
if __name__ == "__main__":
    ## Input Parameters 
    patient_id       = "PR04"
    PREPROC_PATH     = f"/data_store0/presidio/nihon_kohden/{patient_id}/nkhdf5/biomarker/"

    #Get list of files specifying folder path and file format as string
    def GetFilePaths(FileDirectory, FileFormat):
        FileNames = sorted(filter(lambda x: True if FileFormat in x else False, os.listdir(FileDirectory)))
        FilePaths = []
        for i in range(len(FileNames)):
            FilePaths.append(FileDirectory+FileNames[i])
        return FilePaths

    file_paths = GetFilePaths(PREPROC_PATH, "ieeg.h5")

    #FOR PR06 ONLY: remove paths to recordings excluded after JF's annotations
    #recorded during stim:
    #file_paths.remove(f"{PREPROC_PATH}sub-PR06_task-biomarker_0025_ieeg.h5")
    #no neural data:
    #file_paths.remove(f"{PREPROC_PATH}sub-PR06_task-biomarker_0034_ieeg.h5")
    #recorded during stim:
    #file_paths.remove(f"{PREPROC_PATH}sub-PR06_task-biomarker_0035_ieeg.h5")

    ##Loop through each biomarker recording:    
    for i in range(len(file_paths)):
        FileObj = h5py.File(file_paths[i], "r")

        #Extract general attributes
        FileStart = FileObj.attrs['start']
        FileEnd = FileObj.attrs['end']

        #Extract attributes associated to time_axis
        FileSampleRate = FileObj['intracranialEEG_time_axis'].attrs['sample_rate']
        FileTimeZone = FileObj['intracranialEEG_time_axis'].attrs['time_zone']

        #Extract attributes associated to raw timeseries
        FileChannelCount = FileObj['intracranialEEG'].attrs['channel_count']
        FileHighPass = FileObj['intracranialEEG'].attrs['filter_highpass']
        FileLowPass = FileObj['intracranialEEG'].attrs['filter_lowpass']

        #Drop duplicates from time_axis (keep second group of duplicates that will have real neural data)
        CleanTimes = pd.Series(FileObj['intracranialEEG_time_axis']).drop_duplicates(keep='last')
        CleanTimesIdx = list(CleanTimes.index.values)

        RawData = pd.DataFrame(np.array(FileObj['intracranialEEG']))
        CleanData = RawData.iloc[CleanTimesIdx, :]
        FinalData = CleanData.to_numpy(dtype="float32")
        FinalTimes = CleanTimes.to_numpy(dtype="uint64")
        FinalChannelLabels = np.array(FileObj['intracranialEEG_channellabel_axis'])
        FinalChannelCoord  = np.array(FileObj['intracranialEEG_channelcoord_axis'])
 
        file_name = file_paths[i].split("/")[-1].replace("ieeg.h5", "clean") + "_ieeg.h5"
        out_path  = pathlib.Path(f"/data_store0/presidio/nihon_kohden/{patient_id}/nkhdf5/biomarker_clean", file_name)

        ## Start of the actual code ##
        print("creating: ", file_name)
        print("")
        # Create the file #
        f_obj = HDF5NK(file=out_path, mode="a", create=True, construct=True)
        f_obj.attributes["subject_id"] = patient_id
        f_obj.attributes["start"] = int(FileStart)
        f_obj.attributes["end"] = int(FileEnd)
    
        file_data_ieeg = f_obj["data_ieeg"]
        file_data_ieeg.append(FinalData, component_kwargs={"timeseries": {"data": FinalTimes}})
        file_data_ieeg.axes[1]["channellabel_axis"].append(FinalChannelLabels)
        file_data_ieeg.axes[1]["channelcoord_axis"].append(FinalChannelCoord)

        file_data_ieeg.attributes["filter_lowpass"]  = FileLowPass
        file_data_ieeg.attributes["filter_highpass"] = FileHighPass
        file_data_ieeg.attributes["channel_count"]   = FileChannelCount
        file_data_ieeg.axes[0]["time_axis"].attributes["sample_rate"] = FileSampleRate 
        file_data_ieeg.axes[0]["time_axis"].attributes["time_zone"] = FileTimeZone 

        print("File after appending:")
        print("")
        print("ieeg data size: ", f_obj["data_ieeg"].shape)
        print("ieeg time axis size: ", f_obj["data_ieeg"].axes[0]["time_axis"].shape)
        print("")

    # After closing check if the file exists #
        print(f"File Exists: {out_path.is_file()}")
        print(f"File is Openable: {HDF5NK.is_openable(out_path)}")
        print("")

        f_obj.close()


"""End of code

"""
