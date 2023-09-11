"""hdf5writer.py
# !! Contents within this block are managed by 'conda init' !!

"""

# Package Header #
#from .header import *

# Standard Libraries #
import pathlib
import numpy as np
import datetime
import time
import h5py
import scipy.io

# Third-Party Packages #
from nkhdf5 import hdf5nk

# Local Packages #
HDF5NK = hdf5nk.HDF5NK_0_1_0 
from edfreader import get_edf_list, edf_reader

# Main #
if __name__ == "__main__":
    ## Input Parameters 
    patient_id   = "PR05"
    stage1_path  = "/data_store0/presidio/nihon_kohden/"
    imaging_path = f"/data_store2/imaging/subjects/{patient_id}/elecs/stereo_elecs_all.mat"
    edf_path      = pathlib.Path(stage1_path,patient_id,patient_id)

    ## Extract data from example EDF 
    edf_file_list = get_edf_list(edf_path)
    example_edf   = edf_file_list[11]
    print("Converting to hdf5:")
    print(example_edf)
    print("")

    edf_contents = edf_reader(edf_path,example_edf)
    print("")

    ## Output Parameters
    date_string = edf_contents["edf_start"].strftime("%Y%m%d")
    time_string = edf_contents["edf_start"].strftime("%H%M")
    start_rec    = time.mktime(edf_contents["edf_start"].timetuple())*1e9 #unix epoch time
    file_name = f"examples/sub-{patient_id}_ses-stage1_task-continuous_acq-{date_string}_run-{time_string}_ieeg.h5"
    out_path     = pathlib.Path("/userdata/dastudillo/Repos/python-nkhdf5/", file_name)

    ## Start of the actual code ##

    ### Extract raw data by channel type 
    ieeg_array = np.array([k for k,v in zip(edf_contents['edf_data'], edf_contents['edf_chantype']) if v == 'intracranial EEG']).T
    scalpeeg_array = np.array([k for k,v in zip(edf_contents['edf_data'], edf_contents['edf_chantype']) if v == 'scalp EEG']).T
    ekg_array = np.array([k for k,v in zip(edf_contents['edf_data'], edf_contents['edf_chantype']) if v == 'EKG']).T
    ttl_array = np.array([k for k,v in zip(edf_contents['edf_data'], edf_contents['edf_chantype']) if v == 'TTL']).T

    ### Extract time and convert to nanoseconds 
    time_array = np.array((edf_contents["edf_time_axis"]*1e9).astype(int))

    def get_abs_timestamps(nanostamps_array):
        abs_timestamps = []
        for i in range(len(nanostamps_array)):
            abs_timestamps.append(start_rec+nanostamps_array[i])
        return abs_timestamps

    new_time_array = np.array(get_abs_timestamps(time_array))

    ### Extract channel labels by channel type #
    chanlabs_ieeg_array = np.array([k for k,v in zip(edf_contents['edf_channellabel_axis'], edf_contents['edf_chantype']) if v == 'intracranial EEG'], dtype = h5py.special_dtype(vlen=str))

    chanlabs_scalpeeg_array = np.array([k for k,v in zip(edf_contents['edf_channellabel_axis'], edf_contents['edf_chantype']) if v == 'scalp EEG'], dtype = h5py.special_dtype(vlen=str))

    chanlabs_ekg_array = np.array([k for k,v in zip(edf_contents['edf_channellabel_axis'], edf_contents['edf_chantype']) if v == 'EKG'], dtype = h5py.special_dtype(vlen=str))

    chanlabs_ttl_array = np.array([k for k,v in zip(edf_contents['edf_channellabel_axis'], edf_contents['edf_chantype']) if v == 'TTL'], dtype = h5py.special_dtype(vlen=str))

    ### Extract electrodes coordinates (only for depth electrodes, data_ieeg)
    elecs_mat_file = scipy.io.loadmat(imaging_path)
    elecs_coor = elecs_mat_file['elecmatrix']

    # Create the file #
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

    print("File after appending:")
    print("")
    print("ieeg data size: ", f_obj["data_ieeg"].shape)
    print("ieeg time axis size: ", f_obj["data_ieeg"].axes[0]["time_axis"].shape)
    print("ieeg channel labels axis size: ", f_obj["data_ieeg"].axes[1]["channellabel_axis"].shape)
    print("ieeg channel coordinates axis size: ", f_obj["data_ieeg"].axes[1]["channelcoord_axis"].shape)
    #print("f_obj['data_ieeg'].axes[1]['channellabel_axis']: ", f_obj["data_ieeg"].axes[1]["channellabel_axis"][...])
    print("")
    print("scalp eeg data size: ", f_obj["data_scalpeeg"].shape)
    print("scalp eeg time axis size: ", f_obj["data_scalpeeg"].axes[0]["time_axis"].shape)
    print("scalp eeg channel labels axis size: ", f_obj["data_scalpeeg"].axes[1]["channellabel_axis"].shape)
    print("")
    print("ekg data size: ", f_obj["data_ekg"].shape)
    print("ekg time axis size: ", f_obj["data_ekg"].axes[0]["time_axis"].shape)
    print("ekg channel labels axis size: ", f_obj["data_ekg"].axes[1]["channellabel_axis"].shape)
    print("")
    print("ttl data size: ", f_obj["data_ttl"].shape)
    print("ttl time axis size: ", f_obj["data_ttl"].axes[0]["time_axis"].shape)
    print("ttl channel labels axis size: ", f_obj["data_ttl"].axes[1]["channellabel_axis"].shape)
    print("")
    print("New file saved as: ", file_name.split('/')[1])
    print("")

    # After closing check if the file exists #
    print(f"File Exists: {out_path.is_file()}")
    print(f"File is Openable: {HDF5NK.is_openable(out_path)}")
    print("")


"""End of code

"""
