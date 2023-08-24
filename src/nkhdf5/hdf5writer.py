"""hdf5writer.py

"""

# Package Header #
#from .header import *

# Standard Libraries #
import pathlib
import numpy as np

# Third-Party Packages #
from nkhdf5 import hdf5nk

# Local Packages #
HDF5NK = hdf5nk.HDF5NK_0_1_0 
from edfreader import get_edf_list, edf_reader

# Main #
if __name__ == "__main__":
    # Parameters #
    patient_id   = "PR05"
    stage1_path  = "/data_store0/presidio/nihon_kohden/"
    edf_path      = pathlib.Path(stage1_path,patient_id,patient_id)
    file_name    = "examples/nkhdf5_example1.h5"
    out_path     = pathlib.Path("/userdata/dastudillo/Repos/python-nkhdf5/",file_name)  # The file path as a pathlib Path

    # Start of the actual code #

    # Extract data from EDF files #
    edf_file_list = get_edf_list(edf_path)
    example_edf   = edf_file_list[9]
    print("Converting to hdf5:")
    print(example_edf)

    edf_contents = edf_reader(edf_path,example_edf)
   # print("Contents info:")
   # print(edf_contents)

    # Counter function #
    def occurance(dict_lab, elem_name):
        dict_elems = {item: dict_lab.count(item) for item in dict_lab}
        num_elems  = dict_elems.get(elem_name)
        return num_elems

    # Create the file #
    with HDF5NK(file=out_path, mode="a", create=True, construct=True) as file:
        file.attributes['subject_id'] = patient_id
        file.attributes['start'] = edf_contents[0]["edf_start"]
        file.attributes['end'] = edf_contents[0]["edf_end"]

        # Add data for iEEG channels 
        file["data_ieeg"].attributes["filter_lowpass"]  = edf_contents[0]["edf_lowpass"]
        file["data_ieeg"].attributes["filter_highpass"] = edf_contents[0]["edf_highpass"]
        file["data_ieeg"].attributes["channel_count"]   = occurance(edf_contents[0]["edf_chantype"],"intracranial EEG")
        file["data_ieeg"].axes[0]['time_axis'].attrs['sample_rate'] = edf_contents[0]["edf_sfreq"]
        file["data_ieeg"].axes[0]['time_axis'].attrs['time_zone'] = edf_contents[0]["edf_timezone"]

        # Add data for scalp EEG channels
        file["data_scalpeeg"].attributes["filter_lowpass"]  = edf_contents[0]["edf_lowpass"]
        file["data_scalpeeg"].attributes["filter_highpass"] = edf_contents[0]["edf_highpass"]
        file["data_scalpeeg"].attributes["channel_count"]   = occurance(edf_contents[0]["edf_chantype"],"scalp EEG")
        file["data_scalpeeg"].axes[0]['time_axis'].attrs['sample_rate'] = edf_contents[0]["edf_sfreq"]
        file["data_scalpeeg"].axes[0]['time_axis'].attrs['time_zone'] = edf_contents[0]["edf_timezone"]

        # Add data for EKG channels
        file["data_ekg"].attributes["filter_lowpass"]  = edf_contents[0]["edf_lowpass"]
        file["data_ekg"].attributes["filter_highpass"] = edf_contents[0]["edf_highpass"]
        file["data_ekg"].attributes["channel_count"]   = occurance(edf_contents[0]["edf_chantype"],"EKG")
        file["data_ekg"].axes[0]['time_axis'].attrs['sample_rate'] = edf_contents[0]["edf_sfreq"]
        file["data_ekg"].axes[0]['time_axis'].attrs['time_zone'] = edf_contents[0]["edf_timezone"]

         # Add data for DC channels
        file["data_ttl"].attributes["filter_lowpass"]  = edf_contents[0]["edf_lowpass"]
        file["data_ttl"].attributes["filter_highpass"] = edf_contents[0]["edf_highpass"]
        file["data_ttl"].attributes["channel_count"]   = occurance(edf_contents[0]["edf_chantype"],"TTL")
        file["data_ttl"].axes[0]['time_axis'].attrs['sample_rate'] = edf_contents[0]["edf_sfreq"]
        file["data_ttl"].axes[0]['time_axis'].attrs['time_zone'] = edf_contents[0]["edf_timezone"]


"""End of code

"""


