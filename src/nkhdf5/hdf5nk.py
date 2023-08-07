"""hdf5xltek.py
A HDF5 file which contains data for XLTEK EEG data.
"""
# Package Header #
from .header import *


# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
import pathlib
from typing import Any
from typing import Union
from typing import Mapping

import h5py
from baseobjects.functions import singlekwargdispatch

# Third-Party Packages #
from classversioning import TriNumberVersion
from classversioning import Version
from classversioning import VersionType
from hdf5objects.dataset import ElectricalSeriesMap
from hdf5objects.fileobjects import HDF5EEG
from hdf5objects.fileobjects import HDF5EEGMap
from hdf5objects.hdf5bases import HDF5Dataset
from hdf5objects.hdf5bases import HDF5File
from hdf5objects.hdf5bases import HDF5Map


# Local Packages #


# Definitions #
# Classes #
class NKElectricalSeriesMap(ElectricalSeriesMap):
    """A base outline which defines an Electrical Series for Nihon Kohden system."""

    # TODO: Create ChannelAxisMap, similar to TimeAxisMap, but holds all relevant
    # channel information (names, anatomical locations, coordinates, etc).
    default_attribute_names: Mapping[str, str] = (ElectricalSeriesMap.default_attribute_names | {
        "filter_lowpass": "filter_lowpass",
        "filter_highpass": "filter_highpass",
        "channel_type": "channel_type",
        "channel_name": "channel_name",
        "channel_count": "channel_count"})

    default_attributes: Mapping[str, Any] = (ElectricalSeriesMap.default_attributes | {
        "filter_lowpass": 0,
        "filter_highpass": 0,
        "channel_type": h5py.Empty("str"),
        "channel_name": h5py.Empty("str"),
        "channel_count": 0})


class HDF5NKMap(HDF5EEGMap):
    """A map for HDF5NK files."""

    # TODO: Do we need to specify names for the default attributes? What about the Mapping[dtype, dtype]?
    # Followup with Anthony.
    default_attributes = HDF5EEGMap.default_attributes | {
            "age": "",
            "sex": "U",
            "species": "Homo sapiens",
            "start": 0,
            "end": 0}

    default_map_names = {
            "data_ieeg": "intracranialEEG", 
            "data_scalpeeg": "scalpEEG",
            "data_ekg": "EKG",
            "data_ttl": "DCChannel"
            }
    default_maps = {
            "data_ieeg": NKElectricalSeriesMap(
                attributes={"units": "microvolts"},
                object_kwargs={"shape": (0, 0), "maxshape": (None, None)},
                ),
            "data_scalpeeg": NKElectricalSeriesMap(
                attributes={"units": "microvolts"},
                object_kwargs={"shape": (0, 0), "maxshape": (None, None)},
                ),
            "data_ekg": NKElectricalSeriesMap(
                attributes={"units": "microvolts"},
                object_kwargs={"shape": (0, 0), "maxshape": (None, None)},
                ),
            "data_ttl": NKElectricalSeriesMap(
                attributes={"units": "microvolts"},
                object_kwargs={"shape": (0, 0), "maxshape": (None, None)},
                ),
            }


class HDF5NK(HDF5EEG):
    """A HDF5 file which contains data for Nihon Kohden EEG data.

    Class Attributes:
        _registration: Determines if this class will be included in class registry.
        _VERSION_TYPE: The type of versioning to use.
        FILE_TYPE: The file type name of this class.
        VERSION: The version of this class.
        default_map: The HDF5 map of this object.
    """

    _registration: bool = True
    _VERSION_TYPE: VersionType = VersionType(name="HDF5NK", class_=TriNumberVersion)
    VERSION: Version = TriNumberVersion(0, 0, 0)
    FILE_TYPE: str = "NK_EEG"
    default_map: HDF5Map = HDF5NKMap()

    @classmethod
    def get_version_from_file(cls, file: pathlib.Path | str | h5py.File) -> Version:
        """Return a version from a file.

        Args:
            file: The path to file to get the version from.

        Returns:
            The version from the file.
        """
        v_name = cls.default_map.attribute_names["file_version"]

        if isinstance(file, pathlib.Path):
            file = file.as_posix()

        if isinstance(file, str):
            file = h5py.File(file)

        if v_name in file.attrs:
            return TriNumberVersion(file.attrs[v_name])
        elif cls.get_version_class(TriNumberVersion(0, 1, 0)).validate_file_type(file):
            return TriNumberVersion(0, 1, 0)
