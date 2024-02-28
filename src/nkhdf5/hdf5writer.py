"""hdf5writer.py

By providing a list of file paths, converts an EDF file to HDF5 file

"""

# Standard Libraries #
import pathlib
import numpy as np
import pandas as pd
import datetime
import time
import h5py
import scipy.io
import ast

# Third-Party Packages #
from nkhdf5 import hdf5nk
HDF5NK = hdf5nk.HDF5NK_0_1_0

# Local Packages #
from edfreader import get_edf_list, edf_reader
from concatenator_tools import FilesForBiomarker

# Main #
if __name__ == "__main__":
    ## Input Parameters 
    patient_id   = "PR06"
    stage1_path  = "/data_store0/presidio/nihon_kohden"
    edf_path      = pathlib.Path(stage1_path,patient_id,patient_id)
    #imaging_path = f"/data_store2/imaging/subjects/{patient_id}/elecs/PR03_elecs_all.mat" #PR03 only
    #imaging_path = f"/data_store2/imaging/subjects/{patient_id}/elecs/stereo_elecs_all.mat" #PR04 and PR05
    imaging_path = f"/data_store2/imaging/subjects/{patient_id}/elecs/elecs_all.mat" #PR01 and PR06 
    outpath = pathlib.Path(stage1_path, patient_id, "nkhdf5/edf_to_hdf5")

    EDF_CATALOG = pd.read_csv(f"{stage1_path}/{patient_id}/{patient_id}_edf_catalog.csv")
    BiomarkerSurveys = pd.read_csv(f"{stage1_path}/{patient_id}/clinical_scores/BiomarkerSurveys.csv")
    BiomarkerSurveyTimes = pd.to_datetime(BiomarkerSurveys['SurveyStart'])

    ## Extract list of all edfs
    edf_all = get_edf_list(edf_path)
    ## Extract list of edf associated to biomarker surveys
    edf_for_bm = FilesForBiomarker(10, 'EDF', BiomarkerSurveyTimes, EDF_CATALOG)
    
    ## Start of actual code, loop edf files
    for i in range(len(edf_for_bm)):
        edf_contents = edf_reader(edf_path, edf_for_bm[i])
        date_string  = edf_contents["edf_start"].strftime("%Y%m%d")
        time_string  = edf_contents["edf_start"].strftime("%H%M")
        start_rec    = time.mktime(edf_contents["edf_start"].timetuple())*1e9 #unix epoch time
        file_name    = f"sub-{patient_id}_ses-stage1_task-continuous_acq-{date_string}_run-{time_string}_ieeg.h5"
        out_path     = pathlib.Path(outpath, file_name)

        ### Extract raw data by channel type
        ieeg_array = np.array([k for k,v in zip(edf_contents['edf_data'], edf_contents['edf_chantype']) if v == 'intracranial EEG']).T
        scalpeeg_array = np.array([k for k,v in zip(edf_contents['edf_data'], edf_contents['edf_chantype']) if v == 'scalp EEG']).T
        ekg_array = np.array([k for k,v in zip(edf_contents['edf_data'], edf_contents['edf_chantype']) if v == 'EKG']).T
        ttl_array = np.array([k for k,v in zip(edf_contents['edf_data'], edf_contents['edf_chantype']) if v == 'TTL']).T

        ### Extract raw time and convert to absolute timestamps (nanoseconds) 
        time_array = np.array((edf_contents["edf_time_axis"]*1e9).astype(int))

        def get_abs_timestamps(nanostamps_array):
            abs_timestamps = []
            for i in range(len(nanostamps_array)):
                abs_timestamps.append(start_rec+nanostamps_array[i])
            return abs_timestamps

        new_time_array = np.array(get_abs_timestamps(time_array))

        ### Extract channel labels by channel type
        chanlabs_ieeg_array = np.array([k for k,v in zip(edf_contents['edf_channellabel_axis'], edf_contents['edf_chantype']) if v == 'intracranial EEG'], dtype = h5py.special_dtype(vlen=str))
        chanlabs_scalpeeg_array = np.array([k for k,v in zip(edf_contents['edf_channellabel_axis'], edf_contents['edf_chantype']) if v == 'scalp EEG'], dtype = h5py.special_dtype(vlen=str))
        chanlabs_ekg_array = np.array([k for k,v in zip(edf_contents['edf_channellabel_axis'], edf_contents['edf_chantype']) if v == 'EKG'], dtype = h5py.special_dtype(vlen=str))
        chanlabs_ttl_array = np.array([k for k,v in zip(edf_contents['edf_channellabel_axis'], edf_contents['edf_chantype']) if v == 'TTL'], dtype = h5py.special_dtype(vlen=str))

        ### Extract electrodes coordinates (only for depth electrodes, data_ieeg)
        elecs_mat_file = scipy.io.loadmat(imaging_path)
        elecs_coor = elecs_mat_file['elecmatrix']

        ### Create the file 
        f_obj = HDF5NK(file=out_path, mode="a", create=True, construct=True)
        f_obj.attributes["subject_id"] = patient_id
        f_obj.attributes["start"] = int(time.mktime(edf_contents["edf_start"].timetuple())*1e9)
        f_obj.attributes["end"] = int(time.mktime(edf_contents["edf_end"].timetuple())*1e9)
    
        file_data_ieeg = f_obj["data_ieeg"]
        file_data_ieeg.append(ieeg_array, component_kwargs={"timeseries": {"data": new_time_array}})
        file_data_ieeg.axes[1]["channellabel_axis"].append(chanlabs_ieeg_array)
        file_data_ieeg.axes[1]["channelcoord_axis"].append(elecs_coor)

        file_data_ieeg.attributes["filter_lowpass"]  = edf_contents["edf_lowpass"]
        file_data_ieeg.attributes["filter_highpass"] = edf_contents["edf_highpass"]
        file_data_ieeg.attributes["channel_count"]   = ieeg_array.shape[1]
        file_data_ieeg.axes[0]['time_axis'].attrs['sample_rate'] = edf_contents["edf_sfreq"]
        file_data_ieeg.axes[0]['time_axis'].attrs['time_zone'] = edf_contents["edf_timezone"]

        file_data_scalpeeg = f_obj["data_scalpeeg"]
        file_data_scalpeeg.append(scalpeeg_array, component_kwargs={"timeseries": {"data": new_time_array}})
        file_data_scalpeeg.axes[1]["channellabel_axis"].append(chanlabs_scalpeeg_array)

        file_data_scalpeeg.attributes["filter_lowpass"]  = edf_contents["edf_lowpass"]
        file_data_scalpeeg.attributes["filter_highpass"] = edf_contents["edf_highpass"]
        file_data_scalpeeg.attributes["channel_count"]   = scalpeeg_array.shape[1]
        file_data_scalpeeg.axes[0]['time_axis'].attrs['sample_rate'] = edf_contents["edf_sfreq"]
        file_data_scalpeeg.axes[0]['time_axis'].attrs['time_zone'] = edf_contents["edf_timezone"]

        file_data_ekg = f_obj["data_ekg"]
        file_data_ekg.append(ekg_array, component_kwargs={"timeseries": {"data": new_time_array}})
        file_data_ekg.axes[1]["channellabel_axis"].append(chanlabs_ekg_array)

        file_data_ekg.attributes["filter_lowpass"]  = edf_contents["edf_lowpass"]
        file_data_ekg.attributes["filter_highpass"] = edf_contents["edf_highpass"]
        file_data_ekg.attributes["channel_count"]   = ekg_array.shape[1]
        file_data_ekg.axes[0]['time_axis'].attrs['sample_rate'] = edf_contents["edf_sfreq"]
        file_data_ekg.axes[0]['time_axis'].attrs['time_zone'] = edf_contents["edf_timezone"]

        file_data_ttl = f_obj["data_ttl"]
        file_data_ttl.append(ttl_array, component_kwargs={"timeseries": {"data": new_time_array}})
        file_data_ttl.axes[1]["channellabel_axis"].append(chanlabs_ttl_array)

        file_data_ttl.attributes["filter_lowpass"]  = edf_contents["edf_lowpass"]
        file_data_ttl.attributes["filter_highpass"] = edf_contents["edf_highpass"]
        file_data_ttl.attributes["channel_count"]   = ttl_array.shape[1]
        file_data_ttl.axes[0]['time_axis'].attrs['sample_rate'] = edf_contents["edf_sfreq"]
        file_data_ttl.axes[0]['time_axis'].attrs['time_zone'] = edf_contents["edf_timezone"]

        print(f"{edf_for_bm[i]} saved as: ", file_name)
        print("")
        
        f_obj.close()

        print("Converting next file...")
        print("")
        #After closing check if the file exists #
        #print(f"File Exists: {out_path.is_file()}")
        #print(f"File is Openable: {HDF5NK.is_openable(out_path)}")
        #print("")


"""End of code

"""
