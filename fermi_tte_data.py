from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import requests
from fermi_download_data_functions import download_data
from fermi_data_wrangling import show_data_hdu
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import pandas as pd

def extract_fits_data(fits_file):
    """
    Extracts relevant data from the FITS file, either location or time-related information.
    
    Parameters:
    fits_file (str): Path to the FITS file.
    
    Returns:
    tuple: Returns the extracted data.
    """
    with fits.open(fits_file) as hdul:
        # Extract ID from the filename
        id = re.search(r'bn\d{9}', hdul[0].header['FILENAME'])
        id = id.group(0) if id else ""
        
        # Extract detector name from the filename
        detector = re.search(r"glg_tte_(\w+)_bn", hdul[0].header['FILENAME']).group(1)

        # Extract and check TSTART, TSTOP, T90 from header
        ra = hdul[0].header['RA_OBJ']
        dec = hdul[0].header['DEC_OBJ']

        ph_cnt = len(hdul['EVENTS'].data)

        return (id, detector, ra, dec, ph_cnt)  # ID, Detector name, RA, DEC, Photon Count

# Function to process all FITS files in a folder and store the data in a DataFrame
def process_fits_folder(fits_folder, df=None):
    # Get all FITS files in the folder
    files = [f for f in os.listdir(fits_folder) if f.endswith(('.fit', '.fits'))]
    
    # Define columns based on data type (location or time)
    columns = ['ID', 'DETECTOR', 'RA', 'DEC', 'PH_CNT']
    
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

def extract_photon_data(filename):
    """Extract photon arrival times and energy channels from EVENTS HDU."""
    with fits.open(filename) as hdul:
        
        events_data = hdul['EVENTS'].data
        time = events_data['TIME']
        pha = events_data['PHA']
    return time, pha

def plot_count_rate(time, bins=256):
    """Plot the count rate over time."""
    # Create time bins
    bin_edges = np.linspace(time.min(), time.max(), bins)
    bin_size = bin_edges[1] - bin_edges[0]
    digitized = np.digitize(time, bin_edges)
    
    # Calculate count rate in each bin
    count_rate = [np.sum(digitized == i) / bin_size for i in range(1, len(bin_edges))]
    
    # Plot count rate over time
    plt.figure(figsize=(10, 5))
    plt.plot(bin_edges[1:], count_rate, color='blue', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Count Rate (counts/s)')
    plt.title('Count Rate Over Time')
    plt.show()

def test():
    filename = "./glg_tte_n2_bn250116524_v00.fit"
    time, pha = extract_photon_data(filename)
    plot_count_rate(time)

def preprocess_tte_data(year_start, year_end):
    # Download Data
    detectors = [f"n{i}" for i in range(10)] + ["na", "nb", "b0", "b1"]

    # Create an empty DataFrame with specific column names
    columns = ['ID', 'DETECTOR', 'RA', 'DEC', 'PH_CNT']
    tte_data = pd.DataFrame(columns=columns)

    for year in range(year_start, year_end):
        for detector in detectors:
            url_file_string = f"glg_tte_{detector}"
            output_dir = "ph_cnt_and_location"
            
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
    
    # Save processed data as a NumPy file
    npy_file_name = f"./fermi_data/{output_dir}/tte_data.npy"
    np.save(npy_file_name, tte_data.to_numpy())  # Save as .npy file

if __name__ == "__main__":
    preprocess_tte_data(2015, 2016)