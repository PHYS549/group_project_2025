# This file contains functions for processing Fermi TTE (Time-Tagged Event) data.
# Functions include extracting photon data and plotting count rates.

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from .download_data_functions import download_data
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import pandas as pd

# Function: preprocess_tte_data
# Input:
# - year_start (int): Start year for data processing.
# - year_end (int): End year for data processing.
# Output:
# - pd.DataFrame: Processed TTE data.
def preprocess_tte_data(year_start, year_end):
    # Download Data
    detectors = [f"n{i}" for i in range(10)] + ["na", "nb", "b0", "b1"]

    # Create an empty DataFrame with specific column names
    columns = ['ID', 'DETECTOR', 'PH_CNT']
    tte_data = pd.DataFrame(columns=columns)

    for year in range(year_start, year_end):
        for detector in detectors:
            url_file_string = f"glg_tte_{detector}"
            output_dir = "tte"
            
            # Download data
            download_data(range(year, year + 1), Daily_or_Burst='Burst', url_file_string=url_file_string, output_dir=output_dir)

            # Process data
            tte_data = process_fits_folder(f"./fermi_data/{output_dir}", tte_data)

            # Delete all FITS files after processing
            fits_folder_path = f"./fermi_data/{output_dir}"
            for file in os.listdir(fits_folder_path):
                if file.endswith(('.fit', '.fits')):
                    os.remove(os.path.join(fits_folder_path, file))

            print(f"Processed and saved data for {year}, detector {detector}")
            print(tte_data.shape)
        # Save processed data as a NumPy file
        npy_file_name = f"./fermi_data/{output_dir}/tte_data_{year}.npy"
        np.save(npy_file_name, tte_data.to_numpy())  # Save as .npy file
    
    # Pivot the DataFrame to get 'PH_CNT' values for each 'ID' and 'DETECTOR'
    tte_data_pivot = tte_data.pivot_table(index=['ID'], columns='DETECTOR', values='PH_CNT', aggfunc='first')

    # Reorder the columns to match the desired order of detectors
    tte_data_pivot = tte_data_pivot[detectors]

    # Flatten the MultiIndex columns by adding '_PH_CNT' to the detector names
    tte_data_pivot.columns = [f"{detector}_PH_CNT" for detector in tte_data_pivot.columns]

    # Reset the index to make 'ID' columns instead of index
    tte_data_pivot.reset_index(inplace=True)

    # Save processed data as a NumPy file
    npy_file_name = f"./fermi_data/{output_dir}/tte_data.npy"
    np.save(npy_file_name, tte_data_pivot.to_numpy())  # Save as .npy file
    return tte_data_pivot

# Function: extract_fits_data
# Input:
# - fits_file (str): Path to the FITS file.
# Output:
# - tuple: Extracted ID, detector name, and photon count.
def extract_fits_data(fits_file):
    
    with fits.open(fits_file) as hdul:
        # Extract ID from the filename
        id = re.search(r'bn\d{9}', hdul[0].header['FILENAME'])
        id = id.group(0) if id else ""
        
        # Extract detector name from the filename
        detector = re.search(r"glg_tte_(\w+)_bn", hdul[0].header['FILENAME']).group(1)

        ph_cnt = len(hdul['EVENTS'].data)

        return (id, detector, ph_cnt)  # ID, Detector name Photon Count

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
    columns = ['ID', 'DETECTOR', 'PH_CNT']
    
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

if __name__ == "__main__":
    preprocess_tte_data(2015, 2026)