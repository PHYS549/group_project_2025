import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.ticker import LogFormatterMathtext
import re
from .download_data_functions import download_data

def preprocess_time_data(year_start, year_end):
    """
    Process and save Fermi time data across multiple years.

    Args:
        year_start (int): Start year for data processing.
        year_end (int): End year for data processing.

    Returns:
        pd.DataFrame: Processed time data.
    """
    columns = ['ID', 'TSTART', 'TSTOP', 'T90']
    time_data = pd.DataFrame(columns=columns)

    for year in range(year_start, year_end):
        output_dir = 'time'
        download_data(range(year, year + 1), Daily_or_Burst='Burst', url_file_string="glg_bcat_all", output_dir=output_dir)
        time_data = process_fits_folder(f"./fermi_data/{output_dir}", time_data)

        fits_folder_path = f"./fermi_data/{output_dir}"
        for file in os.listdir(fits_folder_path):
            if file.endswith(('.fit', '.fits')):
                os.remove(os.path.join(fits_folder_path, file))

        print(f"Processed and saved data for {year}")
        print(time_data.shape)

        npy_file_name = f"./fermi_data/{output_dir}/time_data_{year}.npy"
        np.save(npy_file_name, time_data.to_numpy())

    npy_file_name = f"./fermi_data/{output_dir}/time_data.npy"
    np.save(npy_file_name, time_data.to_numpy())
    return time_data

def extract_fits_data(fits_file):
    """
    Extract time data fields from a FITS file.

    Args:
        fits_file (str): Path to the FITS file.

    Returns:
        tuple: Extracted ID, TSTART, TSTOP, and T90 values.
    """
    with fits.open(fits_file) as hdul:
        id = re.search(r'bn\d{9}', hdul[0].header['FILENAME'])
        id = id.group(0) if id else ""

        tstart = hdul[0].header['TSTART']
        tstop = hdul[0].header['TSTOP']
        t90 = hdul[0].header['T90']

        return (id, tstart, tstop, t90)

def process_fits_folder(fits_folder, df=None):
    """
    Process FITS files from a folder and convert to DataFrame with calculated date.

    Args:
        fits_folder (str): Path to the folder containing FITS files.
        df (pd.DataFrame): Existing DataFrame to append data to (optional).

    Returns:
        pd.DataFrame: DataFrame containing processed data from FITS files.
    """
    files = [f for f in os.listdir(fits_folder) if f.endswith(('.fit', '.fits'))]
    columns_input = ['ID', 'TSTART', 'TSTOP', 'T90']

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_fits_data, os.path.join(fits_folder, f)) for f in files]
        new_data = []
        for future in as_completed(futures):
            data = future.result()
            new_data.append(data)
        new_data = pd.DataFrame(new_data, columns=columns_input)

    start_date = pd.Timestamp('2001-01-01')
    new_data['DATE'] = start_date + pd.to_timedelta(new_data['TSTART'], unit='s')

    columns_output = ['ID', 'TSTART', 'TSTOP', 'T90', 'DATE']

    if df is None:
        df = pd.DataFrame(columns=columns_output)

    if len(new_data) != 0:
        df = pd.concat([df, pd.DataFrame(new_data, columns=columns_output)], ignore_index=True)

    return df

if __name__ == "__main__":
    preprocess_time_data(2025, 2026)
