import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.ticker import LogFormatterMathtext
import re
from .download_data_functions import download_data

def preprocess_trigger_data(year_start, year_end):
    """
    Process and save Fermi trigger data across multiple years.

    Args:
        year_start (int): Start year for data processing.
        year_end (int): End year for data processing.

    Returns:
        pd.DataFrame: Processed trigger data.
    """
    detectors = [f"n{i}" for i in range(10)] + ["na", "nb", "b0", "b1"]
    columns = ['ID'] + [f"{detector}_TRIG" for detector in detectors]
    trigger_data = pd.DataFrame(columns=columns)

    for year in range(year_start, year_end):
        output_dir = 'trigger'
        download_data(range(year, year + 1), Daily_or_Burst='Burst', url_file_string="glg_bcat_all", output_dir=output_dir)
        trigger_data = process_fits_folder(f"./fermi_data/{output_dir}", trigger_data)

        fits_folder_path = f"./fermi_data/{output_dir}"
        for file in os.listdir(fits_folder_path):
            if file.endswith(('.fit', '.fits')):
                os.remove(os.path.join(fits_folder_path, file))

        print(f"Processed and saved data for {year}")
        print(trigger_data.shape)

        npy_file_name = f"./fermi_data/{output_dir}/trigger_data_{year}.npy"
        np.save(npy_file_name, trigger_data.to_numpy())

    npy_file_name = f"./fermi_data/{output_dir}/trigger_data.npy"
    np.save(npy_file_name, trigger_data.to_numpy())
    return trigger_data

def extract_fits_data(fits_file):
    """
    Extract trigger information for detectors from a FITS file.

    Args:
        fits_file (str): Path to the FITS file.

    Returns:
        tuple: Extracted ID and list of triggered detectors as binary indicators.
    """
    with fits.open(fits_file) as hdul:
        id = re.search(r'bn\d{9}', hdul[0].header['FILENAME'])
        id = id.group(0) if id else ""

        trigger_detectors = hdul[1].data['DETNAM']

        def extract_triggered(input_list):
            """
            Convert detector names into binary trigger status.

            Args:
                input_list (list): List of detector names.

            Returns:
                list: Binary indicators for triggered detectors.
            """
            triggered = [0] * 14
            for item in input_list:
                match = re.search(r'^(NAI|BGO)_(\d+)$', item)
                if match:
                    prefix, num = match.groups()
                    idx = int(num)
                    if prefix == 'NAI' and 0 <= idx <= 11:
                        triggered[idx] = 1
                    elif prefix == 'BGO' and 0 <= idx <= 1:
                        triggered[12 + idx] = 1
            return triggered

        result = [id] + extract_triggered(trigger_detectors)
        return tuple(result)

def process_fits_folder(fits_folder, df=None):
    """
    Process FITS files to extract trigger data and append to a DataFrame.

    Args:
        fits_folder (str): Path to the folder containing FITS files.
        df (pd.DataFrame): Existing DataFrame to append data to (optional).

    Returns:
        pd.DataFrame: DataFrame containing processed trigger data.
    """
    files = [f for f in os.listdir(fits_folder) if f.endswith(('.fit', '.fits'))]
    detectors = [f"n{i}" for i in range(10)] + ["na", "nb", "b0", "b1"]
    columns = ['ID'] + [f"{detector}_TRIG" for detector in detectors]

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
    preprocess_trigger_data(2015, 2026)
