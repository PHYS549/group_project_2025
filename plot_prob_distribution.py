import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.io import fits

# Two point correlation: https://www.astroml.org/book_figures/chapter6/fig_correlation_function.html
# Function to load FITS file and extract event data
def load_fits_data(fits_file):
    with fits.open(fits_file) as hdul:
        Header = hdul[1].header
        RA_cent = Header['CRVAL1']
        DEC_cent = Header['CRVAL2']
        delta_RA = Header['CDELT1']
        delta_DEC = Header['CDELT2']
        
        prob_dens_map = np.array(hdul[1].data)
        coordinates = RA_cent, DEC_cent, delta_RA, delta_DEC
        return prob_dens_map, coordinates

# Main function
def main():
    # Path to the FITS file
    fits_file = "./fermi_data/glg_locprob_all_bn170817529_v02.fit"

    # Directory to save plots
    output_folder = "./plots"
    
    # Create the folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load data
    prob_dens_map, coordinates = load_fits_data(fits_file)
    image_size = [len(prob_dens_map), len(prob_dens_map[0])]
    RA_cent, DEC_cent, delta_RA, delta_DEC = coordinates

    # Calculate coordinate ranges
    RA_min = RA_cent - (image_size[1] / 2) * delta_RA
    RA_max = RA_cent + (image_size[1] / 2) * delta_RA
    DEC_min = DEC_cent - (image_size[0] / 2) * delta_DEC
    DEC_max = DEC_cent + (image_size[0] / 2) * delta_DEC

    # Plot with labeled coordinates
    plt.imshow(prob_dens_map, cmap='viridis', interpolation='nearest', 
    extent=[RA_min, RA_max, DEC_min, DEC_max], origin='lower')
    plt.xlabel('RA(deg)')
    plt.ylabel('DEC(deg)')
    plt.colorbar(label="Value")
    plt.title("Gamma-ray-burst")
    plt.show()


# Run the main function
if __name__ == "__main__":
    main()
