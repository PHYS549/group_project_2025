# This file contains functions for processing and visualizing Fermi location data.
# Functions include extracting RA/DEC from FITS files and creating scatter plots.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from .download_data_functions import download_data
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Function: preprocess_location_data
# Input:
# - year_start (int): Start year for data processing.
# - year_end (int): End year for data processing.
# Output:
# - pd.DataFrame: Processed location data.
def preprocess_location_data(year_start, year_end):
    # Create an empty DataFrame with specific column names
    columns = ['ID', 'RA', 'DEC']
    location_data = pd.DataFrame(columns=columns)

    for year in range(year_start, year_end):
        output_dir='location'
        # Download data
        download_data(range(year, year + 1), Daily_or_Burst='Burst', url_file_string="glg_locprob_all", output_dir=output_dir)

        # Process data
        location_data = process_fits_folder(f"./fermi_data/{output_dir}", location_data)

        # Delete all FITS files after processing
        fits_folder_path = f"./fermi_data/{output_dir}"
        for file in os.listdir(fits_folder_path):
            if file.endswith(('.fit', '.fits')):
                os.remove(os.path.join(fits_folder_path, file))

        print(f"Processed and saved data for {year}")
        print(location_data.shape)

        # Save processed data as a NumPy file
        npy_file_name = f"./fermi_data/{output_dir}/location_data_{year}.npy"
        np.save(npy_file_name, location_data.to_numpy())  # Save as .npy file

    # Save processed data as a NumPy file
    npy_file_name = f"./fermi_data/{output_dir}/location_data.npy"
    np.save(npy_file_name, location_data.to_numpy())  # Save as .npy file
    return location_data

# Function: extract_fits_data
# Input:
# - filename (str): Path to the FITS file.
# Output:
# - tuple: Extracted ID, RA, and DEC from the FITS file.
def extract_fits_data(filename):
    with fits.open(filename) as hdul:
        # Extract ID from the filename
        id = re.search(r'bn\d{9}', hdul[0].header['FILENAME'])
        id = id.group(0) if id else ""

        # Extract ra, dec
        RA = hdul[1].header['CRVAL1']
        DEC = hdul[1].header['CRVAL2']
    return id, RA, DEC

# Function: process_fits_folder
# Input:
# - fits_folder (str): Path to the folder containing FITS files.
# - df (pd.DataFrame): Existing DataFrame to append data to (optional).
# Output:
# - pd.DataFrame: DataFrame containing processed data from FITS files.
def process_fits_folder(fits_folder, df=None):
    # Get all FITS files in the folder
    files = [f for f in os.listdir(fits_folder) if f.endswith(('.fit', '.fits'))]
    
    # Define columns based on data type (location or time)
    columns = ['ID', 'RA', 'DEC']
    
    # If df is not provided, create an empty DataFrame
    if df is None:
        df = pd.DataFrame(columns=columns)
    
    # Process files concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_fits_data, os.path.join(fits_folder, f)) for f in files]
        new_data = []
        for future in as_completed(futures):
            data = future.result()
            new_data.append(data)
    
    # Append new data to the existing DataFrame
    if new_data:
        df = pd.concat([df, pd.DataFrame(new_data, columns=columns)], ignore_index=True)
    
    return df

# Entry point for the script
if __name__ == "__main__":
    preprocess_location_data(2015, 2026)

