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
import csv

def preprocess_poshist_data(year_start, year_end):
    # Create an empty DataFrame
    columns = ['TSTART', 'QSJ_1', 'QSJ_2', 'QSJ_3', 'QSJ_4']
    poshist_data = pd.DataFrame(columns=columns)

    for year in range(year_start, year_end):
        output_dir = 'poshist'

        # Download data
        download_data(range(year, year + 1), Daily_or_Burst='Daily', url_file_string="glg_poshist_all", output_dir=output_dir)

        # Process data -> Save as CSV -> Merge to NPY
        save_data_to_csv(f"./fermi_data/{output_dir}", f"./fermi_data/{output_dir}")
        npy_path = f"./fermi_data/{output_dir}/poshist_data_{year}.npy"
        combine_csv_to_npy(csv_folder=f"./fermi_data/{output_dir}", output_path=npy_path)

        # Load data from npy and convert to DataFrame, merge
        if os.path.exists(npy_path):
            loaded_array = np.load(npy_path, allow_pickle=True)[:,0:5]
            year_df = pd.DataFrame(loaded_array, columns=columns)
            poshist_data = pd.concat([poshist_data, year_df], ignore_index=True)

        # Delete intermediate CSV files
        fits_folder_path = f"./fermi_data/{output_dir}"
        for file in os.listdir(fits_folder_path):
            if file.endswith('.csv'):
                os.remove(os.path.join(fits_folder_path, file))
            if file.endswith(('.fit', '.fits')):
                os.remove(os.path.join(fits_folder_path, file))


        print(f"Processed and saved data for {year}")
        print(poshist_data.shape)

        # Save processed data as a NumPy file
        npy_file_name = f"./fermi_data/{output_dir}/poshist_data_{year}.npy"
        np.save(npy_file_name, poshist_data.to_numpy())  # Save as .npy file

    # Save processed data as a NumPy file
    npy_file_name = f"./fermi_data/poshist/poshist_data.npy"
    np.save(npy_file_name, poshist_data.to_numpy())  # Save as .npy file
    return poshist_data

def extract_fits_data(fits_file, sample_size=1000):
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

        # Take only sample_size of data
        indices = np.linspace(0, total_len - 1, num=min(sample_size, total_len), dtype=int)

        return (
            time[indices],
            qs_1[indices],
            qs_2[indices],
            qs_3[indices],
            qs_4[indices]
        )

def process_one_file(fits_path, output_folder):
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
    os.makedirs(output_folder, exist_ok=True)

    fits_files = [
        os.path.join(fits_folder, f)
        for f in os.listdir(fits_folder)
        if f.endswith(('.fit', '.fits'))
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_file, fp, output_folder) for fp in fits_files]
        for future in as_completed(futures):
            pass  # We already log inside `process_one_file`
    
def combine_csv_to_npy(csv_folder, output_path='combined_data.npy'):
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
    q1, q2, q3, q0 = quat # On Fermi, it's x, y, z, w
    # Rotation matrix calculation based on quaternion components
    sc_cosines = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    #sc_cosines = np.identity(3)
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

def detector_orientation(df):
    def RA_DEC_all_detector_at_quat(row):
        quat = np.array([row['QSJ_1'], row['QSJ_2'], row['QSJ_3'], row['QSJ_4']])

        detectors = {
            'n0': ('NAI_00', 0, 45.89, 90.00 - 20.58),
            'n1': ('NAI_01', 1, 45.11, 90.00 - 45.31),
            'n2': ('NAI_02', 2, 58.44, 90.00 - 90.21),
            'n3': ('NAI_03', 3, 314.87, 90.00 - 45.24),
            'n4': ('NAI_04', 4, 303.15, 90.00 - 90.27),
            'n5': ('NAI_05', 5, 3.35, 90.00 - 89.79),
            'n6': ('NAI_06', 6, 224.93, 90.00 - 20.43),
            'n7': ('NAI_07', 7, 224.62, 90.00 - 46.18),
            'n8': ('NAI_08', 8, 236.61, 90.00 - 89.97),
            'n9': ('NAI_09', 9, 135.19, 90.00 - 45.55),
            'na': ('NAI_10', 10, 123.73, 90.00 - 90.42),
            'nb': ('NAI_11', 11, 183.74, 90.00 - 90.32),
            'b0': ('BGO_00', 12, 0.00, 90.00 - 90.00),
            'b1': ('BGO_01', 13, 180.00, 90.00 - 90.00),
        }

        ra_dec_dict = {}
        for key, (_, _, az, zen) in detectors.items():
            ra, dec = spacecraft_to_radec(az, zen, quat, deg=True)
            ra_dec_dict[key] = (ra, dec)
        return ra_dec_dict

    orientation = []
    for _, row in df.iterrows():
        ra_dec_dict = RA_DEC_all_detector_at_quat(row)

        for name, (ra, dec) in ra_dec_dict.items():
            orientation.append(azzen_to_cartesian(ra, dec))
    return orientation

def plot_all_detector_positions(df, output_dir="detector_plots"):
    def RA_DEC_all_detector_at_quat(row):
        quat = np.array([row['QSJ_1'], row['QSJ_2'], row['QSJ_3'], row['QSJ_4']])

        detectors = {
            'n0': ('NAI_00', 0, 45.89, 90.00 - 20.58),
            'n1': ('NAI_01', 1, 45.11, 90.00 - 45.31),
            'n2': ('NAI_02', 2, 58.44, 90.00 - 90.21),
            'n3': ('NAI_03', 3, 314.87, 90.00 - 45.24),
            'n4': ('NAI_04', 4, 303.15, 90.00 - 90.27),
            'n5': ('NAI_05', 5, 3.35, 90.00 - 89.79),
            'n6': ('NAI_06', 6, 224.93, 90.00 - 20.43),
            'n7': ('NAI_07', 7, 224.62, 90.00 - 46.18),
            'n8': ('NAI_08', 8, 236.61, 90.00 - 89.97),
            'n9': ('NAI_09', 9, 135.19, 90.00 - 45.55),
            'na': ('NAI_10', 10, 123.73, 90.00 - 90.42),
            'nb': ('NAI_11', 11, 183.74, 90.00 - 90.32),
            'b0': ('BGO_00', 12, 0.00, 90.00 - 90.00),
            'b1': ('BGO_01', 13, 180.00, 90.00 - 90.00),
        }

        ra_dec_dict = {}
        for key, (_, num, az, zen) in detectors.items():
            ra, dec = spacecraft_to_radec(az, zen, quat, deg=True)
            ra_dec_dict[key] = (ra, dec, num)
        return ra_dec_dict

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    detectors = [f"n{i}" for i in range(10)] + ["na", "nb", "b0", "b1"]
    PHCNT_col_name = [f"{detector}_PH_CNT" for detector in detectors]\
    
    for _, row in df.iterrows():
        ra_dec_dict = RA_DEC_all_detector_at_quat(row)
    
        plt.figure(figsize=(10, 8))
        for name, (ra, dec, num) in ra_dec_dict.items():
            plt.scatter(ra, dec, num)
            plt.text(ra, dec, f"{name}, ph_cnt: {row[PHCNT_col_name[num]]}", fontsize=16, ha='left', va='center')

        plt.xlabel("Right Ascension (deg)", fontsize=14)
        plt.ylabel("Declination (deg)", fontsize=14)
        plt.title("GRB " + str(row['ID']), fontsize=16)
        plt.grid(True)
        plt.tight_layout()

        filename = os.path.join(output_dir, f"GRB_{row['ID']}.png")
        plt.savefig(filename)
        plt.close()

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