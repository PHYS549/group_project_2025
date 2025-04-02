import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.ticker import LogFormatterMathtext
import re
from fermi_download_data_functions import download_data

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
        

        # Extract and check TSTART, TSTOP, T90 from header
        tstart = hdul[0].header['TSTART']
        tstop = hdul[0].header['TSTOP']
        t90 = hdul[0].header['T90']

        return (id, hdul[0].header['DATE'], tstart, tstop, t90)  # ID, DATE, TSTART, TSTOP, T90

# Function to process all FITS files in a folder and store the data in a DataFrame
def process_fits_folder(fits_folder):
    # Get all FITS files in the folder
    files = [f for f in os.listdir(fits_folder) if f.endswith(('.fit', '.fits'))]
    
    # Define columns based on data type (location or time)
    columns = ['ID', 'DATE', 'TSTART', 'TSTOP', 'T90']
    df = pd.DataFrame(columns=columns)
    
    # Process files concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_fits_data, os.path.join(fits_folder, f)) for f in files]
        for future in as_completed(futures):
            data = future.result()
            # Append data to the DataFrame
            df = pd.concat([df, pd.DataFrame([data], columns=columns)], ignore_index=True)
    
    # Create the output folder if it doesn't exist
    output_folder = "./plots"
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(output_folder, f"fermi_time_data.csv"), index=False)
    return df

# Function to create plots for the location or time data
def create_data_plots(df, output_folder):
    plt.figure(figsize=(10, 6))
    
    
    # Convert 'DATE' column to datetime format
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

    # Calculate the difference in years from 2015-01-01
    start_date = pd.to_datetime('2015-01-01')
    df['Years_since_2015'] = (df['DATE'] - start_date).dt.total_seconds() / (60 * 60 * 24 * 365.25)

    # Drop rows with invalid 'DATE' values
    df = df.dropna(subset=['Years_since_2015'])

    # Define bins for the histogram
    years_bins = np.linspace(np.min(df['Years_since_2015']), np.max(df['Years_since_2015']), 200)

    # Plot the histogram for GRB events over time (in years)
    plt.figure(figsize=(10, 6))
    plt.hist(df['Years_since_2015'], bins=years_bins, color='blue', alpha=0.7)
    plt.xlabel(r'Years since 2015-01-01 [yr]', fontsize=16)
    plt.ylabel('Number of Events', fontsize=16)
    plt.title('GRB Events over Time', fontsize=18)
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    plt.savefig(os.path.join(output_folder, "GRB_events_over_time_years.png"))
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
    plt.savefig(os.path.join(output_folder, "T90_distribution.png"))
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
    # Download Data
    download_data(range(2015, 2026), Daily_or_Burst='Burst', url_file_string="glg_bcat_all", output_dir='time')

    # Process FITS files for location and time data
    time_data = process_fits_folder("./fermi_data/time")

    # Create plot of all the times
    create_data_plots(time_data, 'plots')
    
    # Define filtering criteria
    short_GRB_criteria = {
        'T90': lambda x: x < 2
    }
    short_GRB_data = filtering(time_data, short_GRB_criteria)
    # Calculate the number of events
    event_num_short_GRB = len(short_GRB_data)

    # Calculate the duration (difference between max TSTOP and min TSTART)
    duration_short_GRB = duration(short_GRB_data)

    # Calculate the average occurrence rate (events per unit of time)
    average_occurrence_rate = event_num_short_GRB / duration_short_GRB

    # Print the results
    print(f"Number of events: {event_num_short_GRB}")
    print(f"Total duration: {duration_short_GRB} seconds")
    print(f"Average occurrence rate: {average_occurrence_rate} events per second")