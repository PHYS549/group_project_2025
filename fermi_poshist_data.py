from astropy.coordinates import SkyCoord
import astropy.units as u
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

def preprocess_poshist_data(year_start, year_end):
    # Create an empty DataFrame with specific column names
    columns = ['poshist_time', 'QSJ_1', 'QSJ_2','QSJ_3','QSJ_4']
    poshist_data = pd.DataFrame(columns=columns)

    for year in range(year_start, year_end):
        output_dir = 'poshist'
        # Download data
        download_data(range(year, year + 1), Daily_or_Burst='Daily', url_file_string="glg_poshist_all", output_dir=output_dir)

        # Process data
        poshist_data = process_fits_folder(f"./fermi_data/{output_dir}", poshist_data)

        # Delete all FITS files after processing
        fits_folder_path = f"./fermi_data/{output_dir}"
        for file in os.listdir(fits_folder_path):
            if file.endswith(('.fit', '.fits')):
                os.remove(os.path.join(fits_folder_path, file))

        print(f"Processed and saved data for {year}")
        print(poshist_data.shape)

        # Save processed data as a NumPy file
        npy_file_name = f"./fermi_data/{output_dir}/poshist_data_{year}.npy"
        np.save(npy_file_name, poshist_data.to_numpy())  # Save as .npy file

    # Save processed data as a NumPy file
    npy_file_name = f"./fermi_data/{output_dir}/poshist_data.npy"
    np.save(npy_file_name, poshist_data.to_numpy())  # Save as .npy file
    return poshist_data

def extract_fits_data(fits_file):
    """
    Extracts relevant data from the FITS file.
    
    Parameters:
    fits_file (str): Path to the FITS file.
    
    Returns:
    tuple: Returns the extracted data (time, qs_1, qs_2, qs_3, qs_4)
    """
    with fits.open(fits_file) as hdul:
        data = hdul[1].data
        
        # Extract quaternion components (pointing vectors)
        qs_1 = data['QSJ_1']
        qs_2 = data['QSJ_2']
        qs_3 = data['QSJ_3']
        qs_4 = data['QSJ_4']
        
        # Extract the time array (e.g., SCLK_UTC or another time column)
        time = data['SCLK_UTC']  # Use appropriate time field for X-axis

        # Ensure all columns have the same length
        length = len(time)
        assert len(qs_1) == length
        assert len(qs_2) == length
        assert len(qs_3) == length
        assert len(qs_4) == length
        
        return (time, qs_1, qs_2, qs_3, qs_4)

def process_fits_folder(fits_folder, df=None):
    """
    Processes all FITS files in the given folder and appends the extracted data to a DataFrame.
    
    Parameters:
    fits_folder (str): Path to the folder containing the FITS files.
    df (pd.DataFrame, optional): Existing DataFrame to append the new data to. Defaults to None.
    
    Returns:
    pd.DataFrame: The updated DataFrame containing all processed data.
    """
    # Get all FITS files in the folder
    files = [f for f in os.listdir(fits_folder) if f.endswith(('.fit', '.fits'))]
    
    # Define columns for the DataFrame
    columns = ['poshist_time', 'QSJ_1', 'QSJ_2', 'QSJ_3', 'QSJ_4']
    
    # If df is not provided, create an empty DataFrame
    if df is None:
        df = pd.DataFrame(columns=columns)
    
    # Process files concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_fits_data, os.path.join(fits_folder, f)) for f in files]
        new_data = []
        for future in as_completed(futures):
            time, qs_1, qs_2, qs_3, qs_4 = future.result()
            # Flatten the lists and append the data to the new_data list
            for t, q1, q2, q3, q4 in zip(time, qs_1, qs_2, qs_3, qs_4):
                new_data.append([t, q1, q2, q3, q4])
    
    # Append new data to the existing DataFrame
    if new_data:
        new_df = pd.DataFrame(new_data, columns=columns)
        df = pd.concat([df, new_df], ignore_index=True)
    
    return df


def plot_rotation_to_ra_dec(filename):
    # Open the FITS file and read the data
    with fits.open(filename) as hdul:
        data = hdul[1].data
        
        # Extract quaternion components (pointing vectors)
        qs_1 = data['QSJ_1']
        qs_2 = data['QSJ_2']
        qs_3 = data['QSJ_3']
        qs_4 = data['QSJ_4']
        
        # Extract the time array (e.g., SCLK_UTC or another time column)
        time = data['SCLK_UTC']  # Use appropriate time field for X-axis
        
        # Define quaternion to direction conversion
        def quaternion_to_direction(q0, q1, q2, q3):
            # Convert quaternion to direction vector (unit vector)
            # Assuming the quaternion represents the orientation of the spacecraft
            x = 2 * (q0 * q1 + q2 * q3)
            y = 1 - 2 * (q1 * q1 + q2 * q2)
            z = 2 * (q0 * q2 - q3 * q1)
            return np.array([x, y, z])
        
        # Initialize arrays for RA and Dec
        ra_values = []
        dec_values = []
        
        # Loop through each quaternion and calculate RA and Dec
        for i in range(len(qs_1)):
            # Get direction vector from quaternion
            direction = quaternion_to_direction(qs_4[i], qs_1[i], qs_2[i], qs_3[i])
            
            # Normalize the direction vector (ensure it's a unit vector)
            direction = direction / np.linalg.norm(direction)
            
            # Convert to RA and Dec
            ra = np.arctan2(direction[1], direction[0])  # Right ascension (rad)
            dec = np.arcsin(direction[2])                # Declination (rad)
            
            # Convert from radians to degrees
            ra_deg = np.degrees(ra)
            dec_deg = np.degrees(dec)
            
            ra_values.append(ra_deg)
            dec_values.append(dec_deg)
        
        # Plot the RA and Dec over time
        plt.figure(figsize=(10, 6))
        
        # Plot RA and Dec
        plt.subplot(2, 1, 1)
        plt.plot(time, ra_values, label='RA (deg)', color='r')
        plt.xlabel('Time (s)')
        plt.ylabel('Right Ascension (deg)')
        plt.title('Spacecraft Pointing: Right Ascension')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(time, dec_values, label='Dec (deg)', color='g')
        plt.xlabel('Time (s)')
        plt.ylabel('Declination (deg)')
        plt.title('Spacecraft Pointing: Declination')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def azzen_to_cartesian(az, zen, deg=True):
    """Convert azimuth and zenith angle to Cartesian coordinates."""
    if deg:
        az = np.radians(az)
        zen = np.radians(zen)
    
    x = np.cos(zen) * np.cos(az)
    y = np.cos(zen) * np.sin(az)
    z = np.sin(zen)
    
    return np.array([x, y, z])

def spacecraft_direction_cosines(quat):
    """Calculate the direction cosine matrix from the attitude quaternions."""
    # Quaternion to Direction Cosine Matrix (DCM) conversion
    q0, q1, q2, q3 = quat
    # Rotation matrix calculation based on quaternion components
    sc_cosines = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    return sc_cosines

def spacecraft_to_radec(az, zen, quat, deg=True):
    """Convert a position in spacecraft coordinates (Az/Zen) to J2000 RA/Dec.
    
    Args:
        az (float or np.array): Spacecraft azimuth
        zen (float or np.array): Spacecraft zenith
        quat (np.array): (4, `n`) spacecraft attitude quaternion array
        deg (bool, optional): True if input/output in degrees.
    
    Returns:
        (np.array, np.array): RA and Dec of the transformed position
    """
    ndim = len(quat.shape)
    if ndim == 2:
        numquats = quat.shape[1]
    else:
        numquats = 1

    # Convert azimuth and zenith to Cartesian coordinates
    pos = azzen_to_cartesian(az, zen, deg=deg)
    ndim = len(pos.shape)
    if ndim == 2:
        numpos = pos.shape[1]
    else:
        numpos = 1

    # Spacecraft direction cosine matrix
    sc_cosines = spacecraft_direction_cosines(quat)

    # Handle different cases: one sky position over many transforms, or multiple positions with one transform
    if (numpos == 1) & (numquats > 1):
        pos = np.repeat(pos, numquats).reshape(3, -1)
        numdo = numquats
    elif (numpos > 1) & (numquats == 1):
        sc_cosines = np.repeat(sc_cosines, numpos).reshape(3, 3, -1)
        numdo = numpos
    elif numpos == numquats:
        numdo = numpos
        if numdo == 1:
            sc_cosines = sc_cosines[:, :, np.newaxis]
            pos = pos[:, np.newaxis]
    else:
        raise ValueError(
            'If the size of az/zen coordinates is > 1 AND the size of quaternions is > 1, then they must be of the same size'
        )

    # Convert numpy arrays to list of arrays for vectorized calculations
    sc_cosines_list = np.squeeze(np.split(sc_cosines, numdo, axis=2))
    pos_list = np.squeeze(np.split(pos, numdo, axis=1))
    if numdo == 1:
        sc_cosines_list = [sc_cosines_list]
        pos_list = [pos_list]

    # Convert position to J2000 frame
    cartesian_pos = np.array(list(map(np.dot, sc_cosines_list, pos_list))).T
    cartesian_pos[2, (cartesian_pos[2, np.newaxis] < -1.0).reshape(-1)] = -1.0
    cartesian_pos[2, (cartesian_pos[2, np.newaxis] > 1.0).reshape(-1)] = 1.0

    # Transform Cartesian position to RA/Dec in J2000 frame
    dec = np.arcsin(cartesian_pos[2, np.newaxis])
    ra = np.arctan2(cartesian_pos[1, np.newaxis], cartesian_pos[0, np.newaxis])
    ra[(np.abs(cartesian_pos[1, np.newaxis]) < 1e-6) & (
                np.abs(cartesian_pos[0, np.newaxis]) < 1e-6)] = 0.0
    ra[ra < 0.0] += 2.0 * np.pi

    if deg:
        ra = np.rad2deg(ra)
        dec = np.rad2deg(dec)
    
    return np.squeeze(ra), np.squeeze(dec)

import numpy as np
import pandas as pd

def interpolate_qs_for_time(df, time_values):
    """
    Interpolates the values of QSJ_1, QSJ_2, QSJ_3, QSJ_4 for each time in the `time_values` column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the time and quaternion columns.
    time_values (pd.Series): A pandas Series containing the times for which you want to interpolate the quaternion values.

    Returns:
    pd.DataFrame: DataFrame with interpolated quaternion values for each time in `time_values`.
    """
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


if __name__ == "__main__":
    preprocess_poshist_data(2015, 2026)