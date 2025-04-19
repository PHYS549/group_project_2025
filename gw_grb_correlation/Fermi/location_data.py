import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from .download_data_functions import download_data
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

def preprocess_location_data(year_start, year_end):
    """
    Process and save Fermi location data across multiple years.

    Args:
        year_start (int): Start year for data processing.
        year_end (int): End year for data processing.

    Returns:
        pd.DataFrame: Processed location data.
    """
    columns = ['ID', 'RA', 'DEC']
    location_data = pd.DataFrame(columns=columns)

    for year in range(year_start, year_end):
        output_dir = 'location'
        download_data(range(year, year + 1), Daily_or_Burst='Burst', url_file_string="glg_locprob_all", output_dir=output_dir)
        location_data = process_fits_folder(f"./fermi_data/{output_dir}", location_data)

        fits_folder_path = f"./fermi_data/{output_dir}"
        for file in os.listdir(fits_folder_path):
            if file.endswith(('.fit', '.fits')):
                os.remove(os.path.join(fits_folder_path, file))

        print(f"Processed and saved data for {year}")
        print(location_data.shape)

        npy_file_name = f"./fermi_data/{output_dir}/location_data_{year}.npy"
        np.save(npy_file_name, location_data.to_numpy())

    npy_file_name = f"./fermi_data/{output_dir}/location_data.npy"
    np.save(npy_file_name, location_data.to_numpy())
    return location_data

def extract_fits_data(filename):
    """
    Extract ID, RA, and DEC from a FITS file.

    Args:
        filename (str): Path to the FITS file.

    Returns:
        tuple: Extracted ID, RA, and DEC from the FITS file.
    """
    with fits.open(filename) as hdul:
        id = re.search(r'bn\d{9}', hdul[0].header['FILENAME'])
        id = id.group(0) if id else ""

        RA = hdul[1].header['CRVAL1']
        DEC = hdul[1].header['CRVAL2']
    return id, RA, DEC

def process_fits_folder(fits_folder, df=None):
    """
    Process all FITS files in a given folder and extract location data.

    Args:
        fits_folder (str): Path to the folder containing FITS files.
        df (pd.DataFrame): Existing DataFrame to append data to (optional).

    Returns:
        pd.DataFrame: DataFrame containing processed data from FITS files.
    """
    files = [f for f in os.listdir(fits_folder) if f.endswith(('.fit', '.fits'))]
    columns = ['ID', 'RA', 'DEC']

    if df is None:
        df = pd.DataFrame(columns=columns)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_fits_data, os.path.join(fits_folder, f)) for f in files]
        new_data = []
        for future in as_completed(futures):
            data = future.result()
            new_data.append(data)

    if new_data:
        df = pd.concat([df, pd.DataFrame(new_data, columns=columns)], ignore_index=True)

    return df

if __name__ == "__main__":
    preprocess_location_data(2015, 2026)
