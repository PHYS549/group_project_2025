from astropy.io import fits
import sys
import io
import numpy as np
import pandas as pd

from tensorflow.linalg import norm
from tensorflow import reduce_sum

"""
Function: show_data_hdu
Input:
- fits_file (str): Path to the FITS file.
- hdu_num (int): HDU number to display.
- snapshot_filename (str): File to save the header snapshot.
Output:
- None: Prints HDU information and saves the header snapshot to a file.
"""
def show_data_hdu(fits_file, hdu_num, snapshot_filename="header_snapshot.txt"):
    with fits.open(fits_file) as hdul:
        """
        Capture HDU info output
        """
        old_stdout = sys.stdout  # Backup original stdout
        sys.stdout = io.StringIO()  # Redirect stdout to capture the output
        hdul.info()  # This will print HDU info to the redirected stdout
        hdu_info = sys.stdout.getvalue()  # Get the captured output
        sys.stdout = old_stdout  # Restore original stdout

        """
        Print HDU list information
        """
        print(hdu_info)

        """
        Determine which header to print based on hdu_num
        """
        header = hdul[hdu_num].header
        
        """
        Print header in a more structured format
        """
        print("\nHeader"+str(hdu_num)+" Information:")
        
        header_info = []
        for key, value in header.items():
            print(f"{key:20} = {value}")
            header_info.append(f"{key:20} = {value}")

        """
        Save the HDU info and header information to the snapshot file
        """
        with open(snapshot_filename, 'w') as snapshot_file:
            snapshot_file.write("HDU Information:\n")
            snapshot_file.write(hdu_info)  # Write captured HDU info
            snapshot_file.write("\nHeader"+str(hdu_num)+" Information:")
            for line in header_info:
                snapshot_file.write(line + "\n")

        print(f"\nHeader snapshot saved to {snapshot_filename}")

"""
Function: interpolate_qs_for_time
Input:
- df (pd.DataFrame): DataFrame containing time and quaternion columns.
- time_values (pd.Series): Times for which to interpolate quaternion values.
Output:
- pd.DataFrame: DataFrame with interpolated quaternion values.
"""
def interpolate_qs_for_time(df, time_values):
    
    """
    Ensure that the time column is sorted
    """
    df = df.sort_values(by='TSTART')

    """
    Interpolate the quaternion values using linear interpolation
    """
    df_interpolated = df.set_index('TSTART').interpolate(method='index', limit_direction='both')

    """
    Initialize lists to store interpolated results for each time in `time_values`
    """
    interpolated_qs = []

    for time_value in time_values:
        nearest_time_index = df_interpolated.index.searchsorted(time_value, side='left')

        """
        Handle out-of-bounds case
        """
        if nearest_time_index >= len(df_interpolated):
            qs_1 = qs_2 = qs_3 = qs_4 = np.nan
        else:
            row = df_interpolated.iloc[nearest_time_index]
            qs_1 = row.get('QSJ_1', np.nan)
            qs_2 = row.get('QSJ_2', np.nan)
            qs_3 = row.get('QSJ_3', np.nan)
            qs_4 = row.get('QSJ_4', np.nan)

        interpolated_qs.append([time_value, qs_1, qs_2, qs_3, qs_4])

    interpolated_df = pd.DataFrame(interpolated_qs, columns=['TSTART', 'QSJ_1', 'QSJ_2', 'QSJ_3', 'QSJ_4'])
    return interpolated_df

"""
Function: filtering
Input:
- df (pd.DataFrame): DataFrame to filter.
- criteria (dict): Dictionary of filtering conditions.
Output:
- pd.DataFrame: Filtered DataFrame.
"""
def filtering(df, criteria):
        
    """
    Loop through the criteria and apply filters
    """
    for column, condition in criteria.items():
        """
        Apply the filter condition to the DataFrame
        """
        df = df[df[column].apply(condition)]
    
    return df

"""
Function: duration
Input:
- df (pd.DataFrame): DataFrame containing time data.
Output:
- float: Duration calculated as the difference between max TSTOP and min TSTART.
"""
def duration(df):
    return df['TSTOP'].max()-df['TSTART'].min()

"""
Reads the GW data from a CSV file into a Pandas DataFrame.

:param file_path: Path to the CSV file.
:return: Pandas DataFrame containing the CSV data.
"""
def read_GW_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Data from {file_path} loaded successfully.")
        print(df.info())  # Display DataFrame info for verification
        print(df.head())  # Show the first few rows for preview
        return df
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None
    
"""
Removes 'times' values from gw_data that are within a threshold of each other (in seconds).

:param gw_data: DataFrame containing GW data with 'times' column
:param threshold: The maximum allowed difference between two times to be considered "duplicate" (in seconds).
:return: DataFrame with times that are separated by more than the threshold only.
"""
def remove_duplicate_times_in_gw_data(gw_data, threshold=1.0):
    """
    Define the start date
    """
    start_date = pd.Timestamp('1980-01-06')

    """
    Convert 'times' (in seconds) to datetime by adding them to the start_date
    """
    gw_data['date'] = start_date + pd.to_timedelta(gw_data['times'], unit='s')

    """
    Sort by 'times' to compare successive values
    """
    gw_data_sorted = gw_data.sort_values(by='times').reset_index(drop=True)

    """
    Initialize a list to hold the filtered times
    """
    filtered_times = [gw_data_sorted.iloc[0]]  # Always keep the first entry

    """
    Iterate through the sorted data to find times within the threshold
    """
    for i in range(1, len(gw_data_sorted)):
        if gw_data_sorted.loc[i, 'times'] - gw_data_sorted.loc[i - 1, 'times'] > threshold:
            filtered_times.append(gw_data_sorted.iloc[i])

    """
    Convert the list of filtered times back into a DataFrame
    """
    gw_data_unique = pd.DataFrame(filtered_times)


    print(f"Number of unique times after applying threshold: {len(gw_data_unique)}")
    print(f"First few unique times:\n{gw_data_unique[['times', 'date']].head(15)}")

    return gw_data_unique

"""
Compares 'TSTART' in fermi_data and 'times' in gw_data to check if they are within a specified time range.

:param fermi_data: DataFrame containing Fermi data with 'TSTART' column
:param gw_data: DataFrame containing GW data with 'times' column
:param time_range_seconds: Time range in seconds to compare the two times (default is 86400 seconds = 1 day)
:return: DataFrame with matched data within the specified time range
"""
def compare_time_within_range(fermi_data, gw_data, time_range_seconds=86400):
    """
    Convert 'TSTART' and 'times' to seconds since their respective starting dates
    """
    fermi_start_date = pd.to_datetime('2001-01-01')
    gw_start_date = pd.to_datetime('1980-01-06')
    
    """
    Convert 'TSTART' and 'times' to numeric and then to seconds since the respective start dates
    """
    fermi_data['ALIGNED_SEC'] = pd.to_numeric(fermi_data['TSTART'], errors='coerce')
    gw_data['ALIGNED_SEC'] = pd.to_numeric(gw_data['times'], errors='coerce')
    
    """
    Convert to seconds since the respective starting date
    """
    fermi_data['ALIGNED_SEC'] = fermi_data['ALIGNED_SEC'] + (fermi_start_date - pd.Timestamp("1980-01-01")).total_seconds()
    gw_data['ALIGNED_SEC'] = gw_data['ALIGNED_SEC'] + (gw_start_date - pd.Timestamp("1980-01-01")).total_seconds()

    """
    Initialize an empty list to store the matches
    """
    matched_data = []

    """
    Compare each 'ALIGNED_SEC' in fermi_data with each 'ALIGNED_SEC' in gw_data
    """
    for _, fermi_row in fermi_data.iterrows():
        for _, gw_row in gw_data.iterrows():
            if abs(fermi_row['ALIGNED_SEC'] - gw_row['ALIGNED_SEC']) <= time_range_seconds:  # Check if time difference is within the specified range
                """
                Create a dictionary to store the matched data, including all fermi_data columns
                """
                match = {'grb_time': fermi_row['TSTART'],
                         'grb_ra': fermi_row['RA'],
                         'grb_dec': fermi_row['DEC'], 
                         'gw_time': gw_row['times'], 
                         'time_diff': abs(fermi_row['ALIGNED_SEC'] - gw_row['ALIGNED_SEC']),
                         'GRB_ID': fermi_row['ID']}
                matched_data.append(match)

    """
    Create a DataFrame from the matches
    """
    matched_df = pd.DataFrame(matched_data)

    if len(matched_df)==0:
        print(f"No matching times found within {time_range_seconds} seconds.")
    else:
        print(f"Found {len(matched_df)} matches within {time_range_seconds} seconds.")
    
    return matched_df

def convert_cartesian_to_spherical(cartesian_coords):
    """
    Convert Cartesian coordinates to spherical coordinates (RA, DEC).
    """
    x, y, z = cartesian_coords
    r = np.sqrt(x**2 + y**2 + z**2)
    ra = np.arctan2(y, x)
    dec = np.arcsin(z / r)
    ra = np.degrees(ra) % 360  # Convert to degrees and wrap to [0, 360)
    dec = np.degrees(dec)  # Convert to degrees
    return ra, dec

def extract_tte_data(tte_file):
    
    with fits.open(tte_file) as hdul:
        df = pd.DataFrame(columns=['TIME'])
        df['TIME'] = hdul['EVENTS'].data['TIME']
        return df

def process_data(fermi_data, input_columns):
    # Extract inputs (X)
    X = fermi_data[input_columns].values.astype(np.float64)

    # Convert RA and DEC to radians and then to Cartesian unit vectors
    ra_rad = np.radians(fermi_data['RA'].values.astype(np.float64))
    dec_rad = np.radians(fermi_data['DEC'].values.astype(np.float64))
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    y = np.stack((x, y, z), axis=-1)

    # Clean and split the data
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    split_ratio = 0.8
    train_size = int(len(X) * split_ratio)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Normalize inputs
    mean_X = np.mean(X_train, axis=0)
    std_X = np.std(X_train, axis=0)
    std_X = np.where(std_X == 0, 1, std_X)

    X_train_scaled = (X_train - mean_X) / std_X
    X_test_scaled = (X_test - mean_X) / std_X
    X_scaled = (X - mean_X) / std_X

    return X_scaled, X_train_scaled, X_test_scaled, y, y_train, y_test

# Custom cosine similarity loss
def cosine_similarity_loss(y_true, y_pred):
    import tensorflow as tf
    dot_product = reduce_sum(y_true * y_pred, axis=-1)
    norm_true = norm(y_true, axis=-1)
    norm_pred = norm(y_pred, axis=-1)
    cosine_sim = dot_product / (norm_true * norm_pred)
    return -cosine_sim