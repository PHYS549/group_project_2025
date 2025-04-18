from astropy.io import fits
import sys
import io
import numpy as np
import pandas as pd

# Function: show_data_hdu
# Input:
# - fits_file (str): Path to the FITS file.
# - hdu_num (int): HDU number to display.
# - snapshot_filename (str): File to save the header snapshot.
# Output:
# - None: Prints HDU information and saves the header snapshot to a file.
def show_data_hdu(fits_file, hdu_num, snapshot_filename="header_snapshot.txt"):
    with fits.open(fits_file) as hdul:
        # Capture HDU info output
        old_stdout = sys.stdout  # Backup original stdout
        sys.stdout = io.StringIO()  # Redirect stdout to capture the output
        hdul.info()  # This will print HDU info to the redirected stdout
        hdu_info = sys.stdout.getvalue()  # Get the captured output
        sys.stdout = old_stdout  # Restore original stdout

        # Print HDU list information
        print(hdu_info)

        # Determine which header to print based on hdu_num
        header = hdul[hdu_num].header
        
        # Print header in a more structured format
        print("\nHeader"+str(hdu_num)+" Information:")
        
        header_info = []
        for key, value in header.items():
            print(f"{key:20} = {value}")
            header_info.append(f"{key:20} = {value}")

        # Save the HDU info and header information to the snapshot file
        with open(snapshot_filename, 'w') as snapshot_file:
            snapshot_file.write("HDU Information:\n")
            snapshot_file.write(hdu_info)  # Write captured HDU info
            snapshot_file.write("\nHeader"+str(hdu_num)+" Information:")
            for line in header_info:
                snapshot_file.write(line + "\n")

        print(f"\nHeader snapshot saved to {snapshot_filename}")

# Function: interpolate_qs_for_time
# Input:
# - df (pd.DataFrame): DataFrame containing time and quaternion columns.
# - time_values (pd.Series): Times for which to interpolate quaternion values.
# Output:
# - pd.DataFrame: DataFrame with interpolated quaternion values.
def interpolate_qs_for_time(df, time_values):
    
    # Ensure that the time column is sorted
    df = df.sort_values(by='TSTART')

    # Interpolate the quaternion values using linear interpolation
    df_interpolated = df.set_index('TSTART').interpolate(method='index', limit_direction='both')

    # Initialize lists to store interpolated results for each time in `time_values`
    interpolated_qs = []

    for time_value in time_values:
        nearest_time_index = df_interpolated.index.searchsorted(time_value, side='left')

        # Handle out-of-bounds case
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

# Function: filtering
# Input:
# - df (pd.DataFrame): DataFrame to filter.
# - criteria (dict): Dictionary of filtering conditions.
# Output:
# - pd.DataFrame: Filtered DataFrame.
def filtering(df, criteria):
        
    # Loop through the criteria and apply filters
    for column, condition in criteria.items():
        # Apply the filter condition to the DataFrame
        df = df[df[column].apply(condition)]
    
    return df

# Function: duration
# Input:
# - df (pd.DataFrame): DataFrame containing time data.
# Output:
# - float: Duration calculated as the difference between max TSTOP and min TSTART.
def duration(df):
    return df['TSTOP'].max()-df['TSTART'].min()