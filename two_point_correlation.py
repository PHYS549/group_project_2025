import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas for data handling

from astropy.io import fits
from astroML.utils.decorators import pickle_results
from astroML.datasets import fetch_sdss_specgals
from astroML.correlation import bootstrap_two_point_angular

from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to load FITS file and extract event data
def load_fits_data(fits_file):
    with fits.open(fits_file) as hdul:
        Header = hdul[1].header
        RA = Header['CRVAL1']
        DEC = Header['CRVAL2']
        print(Header)
        return RA, DEC

# Function to process each FITS file
def process_fits_file(fits_file, fits_folder):
    fits_path = os.path.join(fits_folder, fits_file)
    RA, DEC = load_fits_data(fits_path)
    return RA, DEC

def compute_results(data, Nbins=16, Nbootstraps=1000, method='landy-szalay', rseed=0):
    np.random.seed(rseed)
    bins = np.linspace(0.5, 2 ,16)

    results = [bins]
    results += bootstrap_two_point_angular(data['RA'],
                                               data['DEC'],
                                               bins=bins,
                                               method=method,
                                               Nbootstraps=Nbootstraps)

    return results

# Main function
def main(fits_folder):
    # Directory to save plots and CSV
    output_folder = "./plots"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create a DataFrame to store RA and DEC values
    columns = ['RA', 'DEC']
    df = pd.DataFrame(columns=columns)

    # Create a ThreadPoolExecutor to process the FITS files concurrently
    with ThreadPoolExecutor() as executor:
        # Get all FITS files from the folder
        fits_files = [f for f in os.listdir(fits_folder) if f.endswith(".fit") or f.endswith(".fits")]

        # Submit all tasks concurrently to process the FITS files
        futures = [executor.submit(process_fits_file, fits_file, fits_folder) for fits_file in fits_files]

        # As results become available, add RA and DEC to the DataFrame
        for future in as_completed(futures):
            RA, DEC = future.result()
            # Instead of using append, we use pd.concat
            df = pd.concat([df, pd.DataFrame({'RA': [RA], 'DEC': [DEC]})], ignore_index=True)

    # Save RA and DEC values to a CSV file
    df.to_csv(os.path.join(output_folder, "ra_dec_values.csv"), index=False)

    # Plot RA vs DEC data
    plt.figure(figsize=(10, 6))
    plt.scatter(df['RA'], df['DEC'], s=10, color='blue', alpha=0.5)
    plt.xlabel('RA (DEG)')
    plt.ylabel('DEC (DEG)')
    plt.title('GRB events distribution from 2010 to now')
    # Save the plot as PNG
    plt.savefig(os.path.join(output_folder, "RA_DEC_plot.png"))
    plt.close()  # Close the plot after saving to avoid displaying it again

    # Bootstrapping to extract two-point correlation
    (bins, corr, corr_err, bootstraps) = compute_results(df)

    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    fig = plt.figure(figsize=(5, 2.5))

    ax = fig.add_subplot(111)  # , xscale='log', yscale='log'
    ax.errorbar(bin_centers, corr, corr_err,
                fmt='.k', ecolor='gray', lw=1)
    # Save the correlation plot as PNG
    fig.savefig(os.path.join(output_folder, "correlation_plot.png"))
    plt.close()  # Close the plot after saving

# Run the main function
if __name__ == "__main__":
    fits_folder = "./fermi_data"
    main(fits_folder)
