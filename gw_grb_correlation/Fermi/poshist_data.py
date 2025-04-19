from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from .download_data_functions import download_data
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pandas as pd
import csv

def preprocess_poshist_data(year_start, year_end):
    """
    Process and save Fermi POSHIST data across multiple years.

    Args:
        year_start (int): Start year for data processing.
        year_end (int): End year for data processing.

    Returns:
        pd.DataFrame: Processed POSHIST data.
    """
    columns = ['TSTART', 'QSJ_1', 'QSJ_2', 'QSJ_3', 'QSJ_4']
    poshist_data = pd.DataFrame(columns=columns)

    for year in range(year_start, year_end):
        output_dir = 'poshist'
        download_data(range(year, year + 1), Daily_or_Burst='Daily', url_file_string="glg_poshist_all", output_dir=output_dir)

        save_data_to_csv(f"./fermi_data/{output_dir}", f"./fermi_data/{output_dir}")
        npy_path = f"./fermi_data/{output_dir}/poshist_data_{year}.npy"
        combine_csv_to_npy(csv_folder=f"./fermi_data/{output_dir}", output_path=npy_path)

        if os.path.exists(npy_path):
            loaded_array = np.load(npy_path, allow_pickle=True)[:, 0:5]
            year_df = pd.DataFrame(loaded_array, columns=columns)
            poshist_data = pd.concat([poshist_data, year_df], ignore_index=True)

        fits_folder_path = f"./fermi_data/{output_dir}"
        for file in os.listdir(fits_folder_path):
            if file.endswith('.csv') or file.endswith(('.fit', '.fits')):
                os.remove(os.path.join(fits_folder_path, file))

        print(f"Processed and saved data for {year}")
        print(poshist_data.shape)

        npy_file_name = f"./fermi_data/{output_dir}/poshist_data_{year}.npy"
        np.save(npy_file_name, poshist_data.to_numpy())

    npy_file_name = f"./fermi_data/poshist/poshist_data.npy"
    np.save(npy_file_name, poshist_data.to_numpy())
    return poshist_data

def extract_fits_data(fits_file, sample_size=1000):
    """
    Extract quaternion and timestamp data from a FITS file.

    Args:
        fits_file (str): Path to the FITS file.
        sample_size (int): Number of samples to extract.

    Returns:
        tuple: Arrays of sampled time and quaternion data.
    """
    with fits.open(fits_file) as hdul:
        data = hdul[1].data
        qs_1 = data['QSJ_1']
        qs_2 = data['QSJ_2']
        qs_3 = data['QSJ_3']
        qs_4 = data['QSJ_4']
        time = data['SCLK_UTC']

        total_len = len(time)
        if total_len == 0:
            return ([], [], [], [], [])

        indices = np.linspace(0, total_len - 1, num=min(sample_size, total_len), dtype=int)

        return (
            time[indices],
            qs_1[indices],
            qs_2[indices],
            qs_3[indices],
            qs_4[indices]
        )

def process_one_file(fits_path, output_folder):
    """
    Process a single FITS file and save extracted data as a CSV file.

    Args:
        fits_path (str): Path to the FITS file.
        output_folder (str): Folder to save processed data.

    Returns:
        None
    """
    fits_name = os.path.basename(fits_path)
    try:
        time, qs_1, qs_2, qs_3, qs_4 = extract_fits_data(fits_path)

        if len(time) == 0:
            print(f"Skipped empty file: {fits_name}")
            return

        csv_name = os.path.splitext(fits_name)[0] + '.csv'
        csv_path = os.path.join(output_folder, csv_name)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'QSJ_1', 'QSJ_2', 'QSJ_3', 'QSJ_4'])
            for t, q1, q2, q3, q4 in zip(time, qs_1, qs_2, qs_3, qs_4):
                writer.writerow([t, q1, q2, q3, q4])

    except Exception as e:
        print(f"Error processing {fits_name}: {e}")

def save_data_to_csv(fits_folder, output_folder, max_workers=8):
    """
    Save processed quaternion data from FITS files to CSV using multi-threading.

    Args:
        fits_folder (str): Folder containing FITS files.
        output_folder (str): Folder to save CSV files.
        max_workers (int): Number of threads for concurrent processing.

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)

    fits_files = [
        os.path.join(fits_folder, f)
        for f in os.listdir(fits_folder)
        if f.endswith(('.fit', '.fits'))
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_file, fp, output_folder) for fp in fits_files]
        for future in as_completed(futures):
            pass

def combine_csv_to_npy(csv_folder, output_path='combined_data.npy'):
    """
    Combine CSV files into a single NumPy file.

    Args:
        csv_folder (str): Folder containing CSV files.
        output_path (str): Path to save the combined NPY file.

    Returns:
        None
    """
    all_data = []
    csv_files = [
        f for f in os.listdir(csv_folder)
        if f.endswith('.csv')
    ]

    for csv_file in csv_files:
        csv_path = os.path.join(csv_folder, csv_file)
        try:
            df = pd.read_csv(csv_path)
            df['SourceFile'] = csv_file
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        np.save(output_path, combined_df.to_numpy())
        print(f"Combined data saved to: {output_path}")
    else:
        print("No CSV data found to combine.")

if __name__ == "__main__":
    preprocess_poshist_data(2015, 2026)
