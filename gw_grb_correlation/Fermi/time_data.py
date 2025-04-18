# This file contains functions for processing Fermi time data.
# Functions include extracting time-related data from FITS files and creating time-based plots.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.ticker import LogFormatterMathtext
import re
from .download_data_functions import download_data

# Function: preprocess_time_data
# Input:
# - year_start (int): Start year for data processing.
# - year_end (int): End year for data processing.
# Output:
# - pd.DataFrame: Processed time data.
def preprocess_time_data(year_start, year_end):
    # Create an empty DataFrame with specific column names
    columns = ['ID', 'TSTART', 'TSTOP', 'T90']
    time_data = pd.DataFrame(columns=columns)

    for year in range(year_start, year_end):
        output_dir='time'
        # Download data
        download_data(range(year, year + 1), Daily_or_Burst='Burst', url_file_string="glg_bcat_all", output_dir=output_dir)

        # Process data
        time_data = process_fits_folder(f"./fermi_data/{output_dir}", time_data)

        # Delete all FITS files after processing
        fits_folder_path = f"./fermi_data/{output_dir}"
        for file in os.listdir(fits_folder_path):
            if file.endswith(('.fit', '.fits')):
                os.remove(os.path.join(fits_folder_path, file))

        print(f"Processed and saved data for {year}")
        print(time_data.shape)

        # Save processed data as a NumPy file
        npy_file_name = f"./fermi_data/{output_dir}/time_data_{year}.npy"
        np.save(npy_file_name, time_data.to_numpy())  # Save as .npy file

    # Save processed data as a NumPy file
    npy_file_name = f"./fermi_data/{output_dir}/time_data.npy"
    np.save(npy_file_name, time_data.to_numpy())  # Save as .npy file
    return time_data

# Function: extract_fits_data
# Input:
# - fits_file (str): Path to the FITS file.
# Output:
# - tuple: Extracted ID, TSTART, TSTOP, and T90 from the FITS file.
def extract_fits_data(fits_file):
    
    with fits.open(fits_file) as hdul:
        # Extract ID from the filename
        id = re.search(r'bn\d{9}', hdul[0].header['FILENAME'])
        id = id.group(0) if id else ""

        # Extract and check TSTART, TSTOP, T90 from header
        tstart = hdul[0].header['TSTART']
        tstop = hdul[0].header['TSTOP']
        t90 = hdul[0].header['T90']

        return (id, tstart, tstop, t90)  # ID, TSTART, TSTOP, T90

# Function: process_fits_folder
# Input:
# - fits_folder (str): Path to the folder containing FITS files.
# - df (pd.DataFrame): Existing DataFrame to append data to (optional).
# Output:
# - pd.DataFrame: DataFrame containing processed data from FITS files.
def process_fits_folder(fits_folder, df=None):
    # Get all FITS files in the folder
    files = [f for f in os.listdir(fits_folder) if f.endswith(('.fit', '.fits'))]
    
    # Define columns based on data type (time)
    columns_input = ['ID', 'TSTART', 'TSTOP', 'T90']

    # Process files concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_fits_data, os.path.join(fits_folder, f)) for f in files]
        new_data = []
        for future in as_completed(futures):
            data = future.result()
            new_data.append(data)
        new_data = pd.DataFrame(new_data, columns=columns_input)
    
    # Define the start date
    start_date = pd.Timestamp('2001-01-01')
    # Convert 'times' (in seconds) to datetime by adding them to the start_date
    new_data['DATE'] = start_date + pd.to_timedelta(new_data['TSTART'], unit='s')  
    
    # Define columns based on data type (time)
    columns_output = ['ID', 'TSTART', 'TSTOP', 'T90', 'DATE']

    # If df is not provided, create an empty DataFrame
    if df is None:
        df = pd.DataFrame(columns=columns_output)
    
    # Append new data to the existing DataFrame
    if len(new_data)!=0:
        df = pd.concat([df, pd.DataFrame(new_data, columns=columns_output)], ignore_index=True)
    
    return df

if __name__ == "__main__":
    preprocess_time_data(2025, 2026)
