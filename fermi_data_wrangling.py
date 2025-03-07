import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.ticker import LogFormatterMathtext

# Function to extract location (RA, DEC) or time-related data (DATE, T90) from a FITS file
def show_data_hdu(fits_file, data_type):
    with fits.open(fits_file) as hdul:
        hdul.info()
        header = hdul[1].header if data_type == 'location' else hdul[1].header
        print(header)


# Function to extract location (RA, DEC) or time-related data (DATE, T90) from a FITS file
def extract_fits_data(fits_file, data_type):
    with fits.open(fits_file) as hdul:
        header = hdul[1].header if data_type == 'location' else hdul[0].header
        if data_type == 'location':
            return (header['CRVAL1'], header['CRVAL2'])  # RA, DEC
        else:
            return (header['DATE'], header['T90'])  # DATE, T90

# Function to process all FITS files in a folder and store the data in a DataFrame
def process_fits_folder(fits_folder, data_type):
    # Get all FITS files in the folder
    files = [f for f in os.listdir(fits_folder) if f.endswith(('.fit', '.fits'))]
    
    # Define columns based on data type (location or time)
    columns = ['RA', 'DEC'] if data_type == 'location' else ['DATE', 'T90']
    df = pd.DataFrame(columns=columns)
    
    # Process files concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_fits_data, os.path.join(fits_folder, f), data_type) for f in files]
        for future in as_completed(futures):
            data = future.result()
            # Append data to the DataFrame
            df = pd.concat([df, pd.DataFrame([data], columns=columns)], ignore_index=True)
    
    # Create the output folder if it doesn't exist
    output_folder = "./plots"
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(output_folder, f"{data_type}_data.csv"), index=False)
    return df

# Function to create plots for the location or time data
def create_data_plots(df, output_folder, data_type):
    plt.figure(figsize=(10, 6))
    
    if data_type == 'location':
        # Plot the scatter plot for RA vs DEC (location)
        plt.scatter(df['RA'], df['DEC'], s=10, color='blue', alpha=0.5)
        plt.xlabel('RA (DEG)', fontsize=16)
        plt.ylabel('DEC (DEG)', fontsize=16)
        plt.title('GRB Events Distribution', fontsize=18)
        plt.savefig(os.path.join(output_folder, "RA_DEC_plot.png"))
    else:
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

# Function to plot the angular probability distribution from a FITS file
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
    plt.savefig(os.path.join(output_folder, "loc_prob.png"))
    plt.close()

# Main function to execute the data processing and plotting pipeline
def main():
    

    # Process FITS files for location and time data
    location_data = process_fits_folder("./fermi_data/location", 'location')
    time_data = process_fits_folder("./fermi_data/time", 'time')
    
    # Generate plots for both location and time data
    create_data_plots(location_data, "./plots", 'location')
    create_data_plots(time_data, "./plots", 'time')

    # Plot the angular probability distribution for a specific FITS file
    probability_fits_file = "./fermi_data/location/glg_locprob_all_bn250121983_v00.fit"
    plot_certain_event_prob_dist(probability_fits_file)

# Entry point for the script
if __name__ == "__main__":
    main()
    #show_data_hdu("./fermi_data/time/glg_bcat_all_bn241129064_v00.fit", 'time')
    #show_data_hdu("./fermi_data/location/glg_locprob_all_bn241129064_v00.fit", 'location')
