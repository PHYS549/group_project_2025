# This file contains functions for visualizing Fermi data.
# Functions include creating plots for various datasets and saving them to files.

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import LogFormatterMathtext
import pandas as pd

# Function: create_time_data_plots
# Input:
# - df (pd.DataFrame): DataFrame containing time data.
# - output_folder (str): Folder to save the plots.
# Output:
# - None: Saves time-based plots to the specified folder.
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

# Function: create_location_data_plots
# Input:
# - df (pd.DataFrame): DataFrame containing location data.
# - output_folder (str): Folder to save the plots.
# Output:
# - None: Saves location-based plots to the specified folder.
def create_location_data_plots(df, output_folder):
    output_dir = f"./{output_folder}/"
    os.makedirs(output_dir, exist_ok=True)
    # Plot the scatter plot for RA vs DEC (location)
    plt.scatter(df['RA'], df['DEC'], s=10, color='blue', alpha=0.5)
    plt.xlabel('RA (DEG)', fontsize=16)
    plt.ylabel('DEC (DEG)', fontsize=16)
    plt.title('GRB Events Distribution', fontsize=18)
    plt.savefig(os.path.join(output_dir, "RA_DEC_plot.png"))
    
# Function: plot_certain_event_prob_dist
# Input:
# - fits_file (str): Path to the FITS file.
# Output:
# - None: Saves a plot of the angular probability distribution.
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

# Function: plot_count_rate
# Input:
# - df (pd.DataFrame): DataFrame containing time data.
# - bins (int): Number of bins for the histogram.
# Output:
# - None: Displays a plot of the count rate over time.
def plot_count_rate(df, bins=256):
    
    # Create time bins
    time = df['TIME']
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

# Function: azzen_to_cartesian
# Input:
# - az (float or np.array): Azimuth angle.
# - zen (float or np.array): Zenith angle.
# - deg (bool): Whether the input is in degrees.
# Output:
# - np.array: Cartesian coordinates.
def azzen_to_cartesian(az, zen, deg=True):
    
    if deg:
        az = np.radians(az)
        zen = np.radians(zen)
    
    x = np.cos(zen) * np.cos(az)
    y = np.cos(zen) * np.sin(az)
    z = np.sin(zen)
    
    return np.array([x, y, z])

# Function: spacecraft_direction_cosines
# Input:
# - quat (np.array): Quaternion array.
# Output:
# - np.array: Direction cosine matrix.
def spacecraft_direction_cosines(quat):
    
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

# Function: spacecraft_to_radec
# Input:
# - az (float or np.array): Azimuth angle.
# - zen (float or np.array): Zenith angle.
# - quat (np.array): Quaternion array.
# - deg (bool): Whether the input/output is in degrees.
# Output:
# - tuple: RA and Dec in J2000 frame.
def spacecraft_to_radec(az, zen, quat, deg=True):
    
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

# Function: detector_orientation
# Input:
# - df (pd.DataFrame): DataFrame containing quaternion and detector data.
# Output:
# - list: Orientation of detectors in Cartesian coordinates.
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

# Function: plot_all_detector_positions
# Input:
# - df (pd.DataFrame): DataFrame containing detector data.
# - output_dir (str): Directory to save the plots.
# Output:
# - None: Saves plots of detector positions.
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
    PHCNT_col_name = [f"{detector}_PH_CNT" for detector in detectors]
    
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

