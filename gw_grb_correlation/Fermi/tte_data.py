from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from .download_data_functions import download_data
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import pandas as pd

def preprocess_tte_data(year_start, year_end):
    """
    Process and save Fermi TTE data across multiple years and detectors.

    Args:
        year_start (int): Start year for data processing.
        year_end (int): End year for data processing.

    Returns:
        pd.DataFrame: Processed and pivoted TTE data.
    """
    detectors = [f"n{i}" for i in range(10)] + ["na", "nb", "b0", "b1"]
    columns = ['ID', 'DETECTOR', 'PH_CNT']
    tte_data = pd.DataFrame(columns=columns)

    for year in range(year_start, year_end):
        for detector in detectors:
            url_file_string = f"glg_tte_{detector}"
            output_dir = "tte"
            download_data(range(year, year + 1), Daily_or_Burst='Burst', url_file_string=url_file_string, output_dir=output_dir)
            tte_data = process_fits_folder(f"./fermi_data/{output_dir}", tte_data)

            fits_folder_path = f"./fermi_data/{output_dir}"
            for file in os.listdir(fits_folder_path):
                if file.endswith(('.fit', '.fits')):
                    os.remove(os.path.join(fits_folder_path, file))

            print(f"Processed and saved data for {year}, detector {detector}")
            print(tte_data.shape)

        npy_file_name = f"./fermi_data/{output_dir}/tte_data_{year}.npy"
        np.save(npy_file_name, tte_data.to_numpy())

    tte_data_pivot = tte_data.pivot_table(index=['ID'], columns='DETECTOR', values='PH_CNT', aggfunc='first')
    tte_data_pivot = tte_data_pivot[detectors]
    tte_data_pivot.columns = [f"{detector}_PH_CNT" for detector in tte_data_pivot.columns]
    tte_data_pivot.reset_index(inplace=True)

    npy_file_name = f"./fermi_data/{output_dir}/tte_data.npy"
    np.save(npy_file_name, tte_data_pivot.to_numpy())
    return tte_data_pivot

def extract_fits_data(fits_file):
    """
    Extract ID, detector name, and photon count from a FITS file.

    Args:
        fits_file (str): Path to the FITS file.

    Returns:
        tuple: (ID, detector name, photon count).
    """
    with fits.open(fits_file) as hdul:
        id = re.search(r'bn\d{9}', hdul[0].header['FILENAME'])
        id = id.group(0) if id else ""
        detector = re.search(r"glg_tte_(\w+)_bn", hdul[0].header['FILENAME']).group(1)
        ph_cnt = len(hdul['EVENTS'].data)
        return (id, detector, ph_cnt)

def extract_fits_excess_photon_data(bcat_data, filename, bins=256):
    """
    Calculate the excess counts in the burst time window above the baseline.

    Args:
        bcat_data (pd.DataFrame): Time data containing burst info.
        filename (str): Path to the TTE FITS file.
        bins (int): Number of time bins.

    Returns:
        tuple: (ID, detector, total excess counts).
    """
    with fits.open(filename) as hdul:
        tte_data = pd.DataFrame(columns=['TIME'])
        tte_data['TIME'] = hdul['EVENTS'].data['TIME']
        id = re.search(r'bn\d{9}', hdul[0].header['FILENAME'])
        id = id.group(0) if id else ""
        detector = re.search(r"glg_tte_(\w+)_bn", hdul[0].header['FILENAME']).group(1)

        target = bcat_data[bcat_data['ID'] == id]
        if target.empty:
            return (id, detector, np.nan)

        tstart = target['TSTART'].iloc[0]
        tstop = target['TSTOP'].iloc[0]

    time_bin = tte_data['TIME']
    bin_edges = np.linspace(time_bin.min(), time_bin.max(), bins)
    bin_size = bin_edges[1] - bin_edges[0]
    digitized = np.digitize(time_bin, bin_edges)
    time = bin_edges[1:]
    count_rate = [np.sum(digitized == i) / bin_size for i in range(1, len(bin_edges))]

    time = np.asarray(time)
    count_rate = np.asarray(count_rate)
    Baseline = np.average(count_rate)
    mask = (time >= tstart) & (time <= tstop)
    time_window = time[mask]
    counts_window = count_rate[mask]
    excess_counts = counts_window - Baseline
    total_excess_counts = np.trapz(excess_counts, time_window)

    return (id, detector, total_excess_counts)

def process_fits_folder(fits_folder, df=None):
    """
    Process TTE FITS files and append extracted data to a DataFrame.

    Args:
        fits_folder (str): Folder containing FITS files.
        df (pd.DataFrame): Existing DataFrame to append data to.

    Returns:
        pd.DataFrame: Updated DataFrame with TTE data.
    """
    files = [f for f in os.listdir(fits_folder) if f.endswith(('.fit', '.fits'))]
    columns = ['ID', 'DETECTOR', 'PH_CNT']

    if df is None:
        df = pd.DataFrame(columns=columns)

    file_path = f'./fermi_data/time/time_data.npy'
    bcat_data = pd.DataFrame(np.load(file_path, allow_pickle=True))
    bcat_data.columns = ['ID', 'TSTART', 'TSTOP', 'T90', 'DATE']

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_fits_excess_photon_data, bcat_data, os.path.join(fits_folder, f)) for f in files]
        new_data = []
        for future in as_completed(futures):
            data = future.result()
            new_data.append(data)

    if new_data:
        df = pd.concat([df, pd.DataFrame(new_data, columns=columns)], ignore_index=True)

    return df

if __name__ == "__main__":
    preprocess_tte_data(2015, 2026)
