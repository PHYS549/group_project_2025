import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from fermi_download_data_functions import download_data
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
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

def extract_fits_data(filename):
    with fits.open(filename) as hdul:
        # Extract ID from the filename
        id = re.search(r'bn\d{9}', hdul[0].header['FILENAME'])
        id = id.group(0) if id else ""

        # Extract ra, dec
        RA = hdul[1].header['CRVAL1']
        DEC = hdul[1].header['CRVAL2']
    return id, RA, DEC

# Function to process all FITS files in a folder and store the data in a DataFrame
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

def create_location_data_plots(df, output_folder):
    df.columns = ['ID', 'RA', 'DEC']
    output_dir = f"./{output_folder}/"
    os.makedirs(output_dir, exist_ok=True)
    # Plot the scatter plot for RA vs DEC (location)
    plt.scatter(df['RA'], df['DEC'], s=10, color='blue', alpha=0.5)
    plt.xlabel('RA (DEG)', fontsize=16)
    plt.ylabel('DEC (DEG)', fontsize=16)
    plt.title('GRB Events Distribution', fontsize=18)
    plt.savefig(os.path.join(output_dir, "RA_DEC_plot.png"))

# Function to plot the angular probability distribution from a FITS file
def plot_certain_event_prob_dist(fits_file):
    # Create the output folder if it doesn't exist
    output_folder = "./plots"
    os.makedirs(output_folder, exist_ok=True)

    # Load data from the FITS file
    with fits.open(fits_file) as hdul:
        header = hdul[1].header
        RA_cent = header['CRVAL1']
        DEC_cent = header['CRVAL2']
        delta_RA = header['CDELT1']
        delta_DEC = header['CDELT2']
        prob_dens_map = np.array(hdul[1].data)
        coordinates = RA_cent, DEC_cent, delta_RA, delta_DEC
    
    # Calculate the coordinate ranges
    image_size = [len(prob_dens_map), len(prob_dens_map[0])]
    RA_min = RA_cent - (image_size[1] / 2) * delta_RA
    RA_max = RA_cent + (image_size[1] / 2) * delta_RA
    DEC_min = DEC_cent - (image_size[0] / 2) * delta_DEC
    DEC_max = DEC_cent + (image_size[0] / 2) * delta_DEC

    # Plot the probability density map
    plt.imshow(prob_dens_map, cmap='viridis', interpolation='nearest',
               extent=[RA_min, RA_max, DEC_min, DEC_max], origin='lower')
    plt.xlabel('RA(deg)', fontsize=16)
    plt.ylabel('DEC(deg)', fontsize=16)
    plt.colorbar(label="Value")
    plt.title("GRB Probability Distribution", fontsize=18)
    plt.savefig(os.path.join(output_folder, "location_prob.png"))
    plt.close()

# Entry point for the script
if __name__ == "__main__":
    preprocess_location_data(2015, 2026)
    
