from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import requests

def print_fits_info(filename):
    """Print FITS file structure and headers."""
    with fits.open(filename) as hdul:
        hdul.info()
        
        # Print primary header
        print("\n==== Primary Header ====")
        for key, value in hdul[0].header.items():
            print(f"{key:20} : {value}")
        
        # Print all HDU headers
        for i, hdu in enumerate(hdul):
            print(f"\n==== Header for HDU {i} ({hdu.name}) ====")
            for key, value in hdu.header.items():
                print(f"{key:20} : {value}")

def extract_photon_data(filename):
    """Extract photon arrival times and energy channels from EVENTS HDU."""
    with fits.open(filename) as hdul:
        events_data = hdul['EVENTS'].data
        time = events_data['TIME']
        pha = events_data['PHA']
    return time, pha

def plot_photon_counts(time):
    """Plot a histogram of photon counts over time."""
    plt.figure(figsize=(10, 5))
    plt.hist(time, bins=100, color='blue', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Photon Count')
    plt.title('Photon Count Over Time')
    plt.show()

def plot_count_rate(time, bins=256):
    """Plot the count rate over time."""
    # Create time bins
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

def print_ra_dec(filename):
    """Print RA and DEC from the primary HDU header."""
    with fits.open(filename) as hdul:
        primary_header = hdul[0].header  # Access primary header
        ra_obj = primary_header.get('RA_OBJ', 'Not found')  # Get RA_OBJ value
        dec_obj = primary_header.get('DEC_OBJ', 'Not found')  # Get DEC_OBJ value
        
    print(f"RA_OBJ: {ra_obj} degrees")
    print(f"DEC_OBJ: {dec_obj} degrees")
# Print the start and stop times

def check_for_response_matrices(filename):
    with fits.open(filename) as hdul:
        # Check the structure of the HDUs
        hdul.info()
        
        # Access the 'DETECTOR DATA' extension (HDU 1)
        detector_data_hdu = hdul['DETECTOR DATA']
        
        # Extract the RSPFILE column
        rspfile_column = detector_data_hdu.data['RSPFILE']
        
        # Print RSPFILE information
        print("RSPFILE Data:")
        for rspfile in rspfile_column:
            print(rspfile)

def download_file(url, local_filename):
    """
    Downloads a file from a given URL and saves it locally.

    Parameters:
    url (str): The URL of the file to download.
    local_filename (str): The name of the local file where the content will be saved.
    """
    try:
        # Send a GET request to download the file
        response = requests.get(url, stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            # Open a local file to write the content
            with open(local_filename, "wb") as file:
                # Write the content in chunks to the local file
                for chunk in response.iter_content(chunk_size=128):
                    file.write(chunk)
            print(f"File downloaded successfully and saved as {local_filename}")
        else:
            print(f"Failed to download the file. HTTP Status code: {response.status_code}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

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
from astropy.coordinates import SkyCoord
import astropy.units as u

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

# Example usage

# Main execution
filename = "glg_poshist_all_081012_v00.fit"
plot_rotation_to_ra_dec(filename)
#print_fits_info(filename)
"""print_ra_dec(filename)  # Display FITS file info
time, pha = extract_photon_data(filename)  # Extract photon data
plot_count_rate(time)  # Plot photon count histogram"""
#check_for_response_matrices(filename)
