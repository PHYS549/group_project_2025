import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from fermi_time_correlation_analysis import create_time_data_plots
from fermi_loc_prob_display import create_location_data_plots
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_npy_to_dataframe(data_type, PRINT_HEAD = False):
    """
    Loads a .npy file into a Pandas DataFrame with allow_pickle=True and prints its info and head.
    :param data_type: Identifier for the dataset (e.g., 'time', 'tte', 'location')
    :return: Loaded DataFrame
    """
    
    file_path = f'./fermi_data/{data_type}/{data_type}_data.npy'
    df = pd.DataFrame(np.load(file_path, allow_pickle=True))
    if data_type=='time':
        df.columns = ['ID', 'DATE', 'TSTART', 'TSTOP', 'T90']
    elif data_type == 'location':
        df.columns = ['ID', 'RA', 'DEC']
    elif data_type =='tte':
        detectors = [f"n{i}" for i in range(10)] + ["na", "nb", "b0", "b1"]
        df.columns = ['ID'] + [f"{detector}_PH_CNT" for detector in detectors]
    if PRINT_HEAD:
        print(f"\nData from {file_path}:")
        print(df.info())
        print(df.head())
    return df

def merge_dataframes(time_df, tte_df, location_df, print_info = False):
    """
    Merges time, tte, and location DataFrames on a common ID using an inner join.
    :param time_df: DataFrame containing time data
    :param tte_df: DataFrame containing tte data
    :param location_df: DataFrame containing location data
    :return: Merged DataFrame
    """
    merged_df = time_df.merge(tte_df, on='ID', how='inner')
    merged_df = merged_df.merge(location_df, on='ID', how='inner')
    if print_info:
        print("\nMerged Data:")
        print(merged_df.info())
        print(merged_df.head())
    return merged_df

def preprocess_fermi_data():
    # Load and display the data
    time_data = load_npy_to_dataframe('time')
    tte_data = load_npy_to_dataframe('tte')
    location_data = load_npy_to_dataframe('location')
    create_time_data_plots(time_data, 'plots')
    create_location_data_plots(location_data, 'plots')

    print("\nMissing values in tte_data:")
    print(location_data.isnull().sum())

    # Merge the data
    merged_data = merge_dataframes(time_data, tte_data, location_data)
    # Save the merged data to a .npy file
    output_path = './fermi_data/preprocessed_fermi_data.npy'
    np.save(output_path, merged_data.to_records(index=False))  # Convert to NumPy structured array
    print(f"\nPreprocessed data saved to {output_path}")
    print(merged_data.shape)
    return merged_data

def read_csv_to_dataframe(file_path):
    """
    Reads a CSV file into a Pandas DataFrame.
    
    :param file_path: Path to the CSV file.
    :return: Pandas DataFrame containing the CSV data.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data from {file_path} loaded successfully.")
        print(df.info())  # Display DataFrame info for verification
        print(df.head())  # Show the first few rows for preview
        return df
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None
    
def remove_duplicate_times(gw_data, threshold=1):
    """
    Removes 'times' values from gw_data that are within a threshold of each other (in seconds).
    
    :param gw_data: DataFrame containing GW data with 'times' column
    :param threshold: The maximum allowed difference between two times to be considered "duplicate" (in seconds).
    :return: DataFrame with times that are separated by more than the threshold only.
    """
    # Sort by 'times' to compare successive values
    gw_data_sorted = gw_data.sort_values(by='times').reset_index(drop=True)

    # Initialize a list to hold the filtered times
    filtered_times = [gw_data_sorted.iloc[0]]  # Always keep the first entry

    # Iterate through the sorted data to find times within the threshold
    for i in range(1, len(gw_data_sorted)):
        if gw_data_sorted.loc[i, 'times'] - gw_data_sorted.loc[i - 1, 'times'] > threshold:
            filtered_times.append(gw_data_sorted.iloc[i])

    # Convert the list of filtered times back into a DataFrame
    gw_data_unique = pd.DataFrame(filtered_times)

    print(f"Number of unique times after applying threshold: {len(gw_data_unique)}")
    print(f"First few unique times:\n{gw_data_unique['times'].head()}")

    return gw_data_unique

def compare_time_within_range(fermi_data, gw_data, time_range_seconds=86400):
    """
    Compares 'TSTART' in fermi_data and 'times' in gw_data to check if they are within a specified time range.
    
    :param fermi_data: DataFrame containing Fermi data with 'TSTART' column
    :param gw_data: DataFrame containing GW data with 'times' column
    :param time_range_seconds: Time range in seconds to compare the two times (default is 86400 seconds = 1 day)
    :return: DataFrame with matched data within the specified time range
    """
    # Convert 'TSTART' and 'times' to seconds since their respective starting dates
    fermi_start_date = pd.to_datetime('2001-01-01')
    gw_start_date = pd.to_datetime('1977-01-01')
    
    # Convert 'TSTART' and 'times' to numeric and then to seconds since the respective start dates
    fermi_data['TSTART_sec'] = pd.to_numeric(fermi_data['TSTART'], errors='coerce')
    gw_data['times_sec'] = pd.to_numeric(gw_data['times'], errors='coerce')
    
    # Convert to seconds since the respective starting date
    fermi_data['TSTART_sec'] = fermi_data['TSTART_sec'] - (fermi_start_date - pd.Timestamp("1970-01-01")).total_seconds()
    gw_data['times_sec'] = gw_data['times_sec'] - (gw_start_date - pd.Timestamp("1970-01-01")).total_seconds()

    # Initialize an empty list to store the matches
    matched_data = []

    # Compare each 'TSTART_sec' in fermi_data with each 'times_sec' in gw_data
    for fermi_time in fermi_data['TSTART_sec']:
        for gw_time in gw_data['times_sec']:
            if abs(fermi_time - gw_time) <= time_range_seconds:  # Check if time difference is within the specified range
                matched_data.append({'fermi_time': fermi_time, 'gw_time': gw_time, 'time_diff': abs(fermi_time - gw_time)})

    # Create a DataFrame from the matches
    matched_df = pd.DataFrame(matched_data)
    
    if matched_df.empty:
        print(f"No matching times found within {time_range_seconds} seconds.")
    else:
        print(f"Found {len(matched_df)} matches within {time_range_seconds} seconds.")
        print(matched_df.head())  # Display the first few matched rows
    
    return matched_df

if __name__ == "__main__":
    # Get Fermi data
    fermi_data = preprocess_fermi_data()
    short_grb_data = fermi_data[fermi_data['T90'] <= 2.5].reset_index(drop=True)

    short_grb_data[short_grb_data['ID'] == 'bn170817529']

    gw_data = read_csv_to_dataframe(f"./gw_data/totalgwdata.csv")
    gw_times = remove_duplicate_times(gw_data)
    print('Now')
    print(short_grb_data.info)
    print(gw_times.info)
    
    compare_time_within_range(short_grb_data, gw_times, time_range_seconds=86400*9)
    