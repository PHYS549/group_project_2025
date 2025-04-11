from astropy.io import fits
import sys
import io
from fermi_download_data_functions import download_data
import matplotlib.pyplot as plt
import numpy as np
from fermi_tte_data import plot_count_rate, extract_photon_data

# Function to extract location (RA, DEC) or time-related data (DATE, T90) from a FITS file
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


# Entry point for the script
if __name__ == "__main__":
    #download_data(range(2024, 2025), Daily_or_Burst='Burst', url_file_string="glg_locprob_all", output_dir='loc_prob')
    #show_data_hdu("./fermi_data/loc_prob/glg_locprob_all_bn240101540_v00.fit", 1, snapshot_filename="pos_data.txt")
    show_data_hdu("./glg_bcat_all_bn170817529_v01.fit", 1, snapshot_filename="catalog_data.txt")
    # Open the FITS file and extract data

# Open the file
with fits.open("glg_bcat_all_bn170817529_v01.fit") as hdul:
    data = hdul["DETECTOR DATA"].data
    row = data[2]  
    print(row)

time, pha = extract_photon_data("./glg_tte_n5_bn170817529_v00.fit")
plot_count_rate(time)
