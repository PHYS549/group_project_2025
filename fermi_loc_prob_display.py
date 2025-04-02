import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from fermi_download_data_functions import download_data


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

# Entry point for the script
if __name__ == "__main__":
    download_data(range(2025, 2026), Daily_or_Burst='Burst', url_file_string="glg_locprob_all", output_dir='loc_prob')
    plot_certain_event_prob_dist("./fermi_data/loc_prob/glg_locprob_all_bn250101968_v00.fit")
