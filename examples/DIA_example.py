#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" first_example.py
A basic example, which introduces Maps and file creation/reading.
"""

# Imports #
# Standard Libraries #
import pathlib

# Third-Party Packages #
from nkhdf5 import hdf5nk

import numpy as np

HDF5NK = hdf5nk.HDF5NK_0_1_0

# Main #
if __name__ == "__main__":
    # Parameters #
    file_name = "examples/DIA_first_example_file.h5"
    out_path = pathlib.Path.cwd() / file_name  # The file path as a pathlib Path

    # Start of the actual code.

    # Map Information #
    print("Map Information:")
    print("This is the map of the file:")
    hdf5nk.HDF5NKMap.print_tree_class()
    print("")

    # Check if the File Exists
    # These should print false because the file has not been made.
    print(f"File Exists: {out_path.is_file()}")
    print(f"File is Openable: {HDF5NK.is_openable(out_path)}")
    print("")

    # Create the file #
    # The create kwarg determines if the file will be created.
    # The construct kwarg determines if the file's structure will be built, which is highly suggested for SWMR.
    with HDF5NK(file=out_path, mode="a", create=True, construct=True) as file:

        """
        Can print File attributes and dataset names as a test.
        """
        print("File was created and constructed by this point.")
    print("")

    # After Closing Check if the File Exists
    print(f"File Exists: {out_path.is_file()}")
    print(f"File is Openable: {HDF5NK.is_openable(out_path)}")
    print("")

    # Open File
    # read mode is the default mode.
    # The load kwarg determines if the whole file structure will be loaded in. This is useful if you plan on looking at
    # everything in the file, but if load is False or not set it will load parts of the structure on demand which is
    # more efficient if you are looking at specific parts and not checking others.
    with HDF5NK(file=out_path, load=True, swmr=True) as file:
        # Caching is on when in read mode.
        # In normal read mode, once data is loaded into cache it has to manually be told refresh the cache.
        # In SWMR mode the cache will clear and get the values from the file again at a predefined interval.
        # There are methods which can turn caching on and off and can set the caching interval.

        # Caching
        # There are two versions of the caching methods. There methods that operate on the specific object and ones that
        # change the caching for all objects contained within that object as well. The larger scope methods have _all_
        # in their method name.
        print(f"Caching: {file.is_cache}")
        file.disable_all_caching()  # Since _all_ is present, this will apply to all contained objects as well.
        print(f"Caching: {file.is_cache}")
        file.enable_all_caching()
        print(f"Caching: {file.is_cache}")
        file.timeless_all_caching()  # Caches will not clear on their own.
        file.clear_all_caches()  # Clear all the caches.
        file.timed_all_caching()  # Caches will clear at regular intervals.
        file.set_all_lifetimes(2.0)  # The sets lifetime of the cache before it will clear in seconds.
        print("")
