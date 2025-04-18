# This file contains functions for preprocessing various types of Fermi data.
# Functions include creating dataframes from .npy files and merging datasets.
# It also includes a main function to download and preprocess Fermi data.

import os
import numpy as np
import pandas as pd

from .time_data import preprocess_time_data
from .location_data import preprocess_location_data
from .tte_data import preprocess_tte_data
from .poshist_data import preprocess_poshist_data
from .trigger_data import preprocess_trigger_data
from .util import interpolate_qs_for_time

# Function: create_dataframe_and_name_column_from_data_files
# Input:
# - data_type (str): Identifier for the dataset (e.g., 'time', 'tte', 'location').
# - PRINT_HEAD (bool): Whether to print the head of the DataFrame.
# Output:
# - pd.DataFrame: Loaded DataFrame with appropriate column names.
def create_dataframe_and_name_column_from_data_files(data_type, PRINT_HEAD = False):
    
    file_path = f'./fermi_data/{data_type}/{data_type}_data.npy'
    df = pd.DataFrame(np.load(file_path, allow_pickle=True))
    if data_type=='time':
        df.columns = ['ID', 'TSTART', 'TSTOP', 'T90', 'DATE']
    elif data_type =='tte':
        detectors = [f"n{i}" for i in range(10)] + ["na", "nb", "b0", "b1"]
        df.columns = ['ID'] + [f"{detector}_PH_CNT" for detector in detectors]
    elif data_type == 'location':
        df.columns = ['ID', 'RA', 'DEC']
    elif data_type =='poshist':
        df.columns = ['TSTART', 'QSJ_1', 'QSJ_2','QSJ_3','QSJ_4']
    elif data_type =='trigger':
        detectors = [f"n{i}" for i in range(10)] + ["na", "nb", "b0", "b1"]
        df.columns = ['ID'] + [f"{detector}_TRIG" for detector in detectors]
    elif data_type =='fermi':
        detectors = [f"n{i}" for i in range(10)] + ["na", "nb", "b0", "b1"]
        df.columns = ['ID', 'TSTART', 'TSTOP', 'T90', 'DATE']  + [f"{detector}_PH_CNT" for detector in detectors] + ['RA', 'DEC'] + ['QSJ_1', 'QSJ_2','QSJ_3','QSJ_4'] + [f"{detector}_TRIG" for detector in detectors]
    
    if PRINT_HEAD:
        print(f"\nData from {file_path}:")
        print(df.info())
        print(df.head())
    return df

# Function: merge_all_datatypes_in_fermi
# Input:
# - time_df (pd.DataFrame): DataFrame containing time data.
# - tte_df (pd.DataFrame): DataFrame containing TTE data.
# - location_df (pd.DataFrame): DataFrame containing location data.
# - poshist_df (pd.DataFrame): DataFrame containing POSHIST data.
# - trigger_df (pd.DataFrame): DataFrame containing trigger data.
# - print_info (bool): Whether to print merged DataFrame info.
# Output:
# - pd.DataFrame: Merged DataFrame containing all data types.
def merge_all_datatypes_in_fermi(time_df, tte_df, location_df, poshist_df, trigger_df, print_info = False):

    merged_df = time_df.merge(tte_df, on='ID', how='inner')
    merged_df = merged_df.merge(location_df, on='ID', how='inner')
    GRB_poshist = interpolate_qs_for_time(poshist_df.astype(float), merged_df['TSTART'].astype(float))
    
    merged_df = merged_df.merge(GRB_poshist, on='TSTART', how='inner')
    merged_df = merged_df.merge(trigger_df, on='ID', how='inner')

    if print_info:
        print("\nMerged Data:")
        print(merged_df.info())
        print(merged_df.head())

    return merged_df

# Function: download_and_preprocess_fermi_data
# Input:
# - start_year (int): Start year for data processing.
# - end_year (int): End year for data processing.
# - download_or_not (bool): Whether to download data before processing.
# Output:
# - pd.DataFrame: Merged and preprocessed Fermi data.
def download_and_preprocess_fermi_data(start_year, end_year, download_or_not = True):
    
    if download_or_not:
        # Download and preprocess raw data
        time_data = preprocess_time_data(start_year, end_year)
        location_data = preprocess_location_data(start_year, end_year)
        tte_data = preprocess_tte_data(start_year, end_year)
        poshist_data = preprocess_poshist_data(start_year, end_year)
        trigger_data = preprocess_trigger_data(start_year, end_year)

    # Load and display the data
    time_data = create_dataframe_and_name_column_from_data_files('time')
    location_data = create_dataframe_and_name_column_from_data_files('location')
    tte_data = create_dataframe_and_name_column_from_data_files('tte')
    poshist_data = create_dataframe_and_name_column_from_data_files('poshist')
    trigger_data = create_dataframe_and_name_column_from_data_files('trigger')

    # Merge the data
    merged_data = merge_all_datatypes_in_fermi(time_data, tte_data, location_data, poshist_data, trigger_data)
    
    output_dir = f"./fermi_data/fermi/"
    os.makedirs(output_dir, exist_ok=True)
    # Save the merged data to a .npy file
    np.save(output_dir + "fermi_data.npy", merged_data.to_records(index=False))  # Convert to NumPy structured array
    print(f"\nPreprocessed data saved to {output_dir}")
    
    # Save the merged data to a .csv file
    merged_data.to_csv(output_dir + "fermi_data.csv", index=False)  # Save without row indices
    print(f"\nPreprocessed data saved to {output_dir}")

    return merged_data
    


if __name__ == "__main__":
    start_year = 2015
    end_year = 2026
    fermi_data = download_and_preprocess_fermi_data(start_year=start_year, end_year=end_year, download_or_not=False)