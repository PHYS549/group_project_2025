import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.ticker import LogFormatterMathtext
import re
from fermi_download_data_functions import download_data

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

def extract_fits_data(fits_file):
    """
    Extracts relevant data from the FITS file.
    
    Parameters:
    fits_file (str): Path to the FITS file.
    
    Returns:
    tuple: Returns the extracted data.
    """
    with fits.open(fits_file) as hdul:
        # Extract ID from the filename
        id = re.search(r'bn\d{9}', hdul[0].header['FILENAME'])
        id = id.group(0) if id else ""

        # Extract and check TSTART, TSTOP, T90 from header
        tstart = hdul[0].header['TSTART']
        tstop = hdul[0].header['TSTOP']
        t90 = hdul[0].header['T90']

        return (id, tstart, tstop, t90)  # ID, TSTART, TSTOP, T90

# Function to process all FITS files in a folder and store the data in a DataFrame
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

# Function to create plots for the time data
def create_time_data_plots(df, output_folder):
    output_dir = f"./{output_folder}/"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Convert 'DATE' column to datetime format without modifying the original DataFrame
    date_series = pd.to_datetime(df['DATE'], errors='coerce')

    # Calculate the difference in years from 2015-01-01 without adding to df
    start_date = pd.to_datetime('2015-01-01')
    years_since_2015 = (date_series - start_date).dt.total_seconds() / (60 * 60 * 24 * 365.25)

    # Drop rows with invalid 'DATE' values (from the calculated series)
    valid_dates = years_since_2015.dropna()

    # Define bins for the histogram
    years_bins = np.linspace(np.min(valid_dates), np.max(valid_dates), 200)

    # Plot the histogram for GRB events over time (in years)
    plt.figure(figsize=(10, 6))
    plt.hist(valid_dates, bins=years_bins, color='blue', alpha=0.7)
    plt.xlabel(r'Years since 2015-01-01 [yr]', fontsize=16)
    plt.ylabel('Number of Events', fontsize=16)
    plt.title('GRB Events over Time', fontsize=18)
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    plt.savefig(os.path.join(output_dir, "GRB_events_over_time_years.png"))
    plt.close()

    # Convert 'T90' values to numeric and filter valid values
    t90_values = pd.to_numeric(df['T90'], errors='coerce').dropna()
    valid_t90 = t90_values[(t90_values > 0) & (t90_values.between(1e-3, 1e3))]

    # Define custom bin edges for the T90 histogram
    t_boundary = np.log10(2)
    fine_bins = np.logspace(-3, t_boundary, 90)
    coarse_bins = np.logspace(t_boundary, 3, 150)
    bins = np.concatenate((fine_bins, coarse_bins[1:]))

    # Plot the histogram for T90 duration distribution
    plt.figure(figsize=(10, 6))
    plt.hist(valid_t90, bins=bins, color='green', alpha=0.7)
    plt.xscale('log')
    plt.xlabel(r'T90 Duration [s]', fontsize=16)
    plt.ylabel('Number of Events', fontsize=16)
    plt.title('T90 Duration Distribution with Variable Bin Size', fontsize=18)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(LogFormatterMathtext())  # Use scientific notation for x-axis
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    plt.savefig(os.path.join(output_dir, "T90_distribution.png"))
    plt.close()
    
def filtering(df, criteria):
    """
    Filter the dataframe based on the given criteria.
    
    Parameters:
    - df (pandas DataFrame): The DataFrame to filter.
    - criteria (dict): Dictionary containing column names as keys and filtering conditions as values.
    
    Returns:
    - pandas DataFrame: A new DataFrame that satisfies the filtering conditions.
    """
    
    # Loop through the criteria and apply filters
    for column, condition in criteria.items():
        # Apply the filter condition to the DataFrame
        df = df[df[column].apply(condition)]
    
    return df

def duration(df):
    return df['TSTOP'].max()-df['TSTART'].min()


if __name__ == "__main__":
    preprocess_time_data(2025, 2026)
