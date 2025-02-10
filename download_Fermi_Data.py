import os
import requests

# Base URL for the Fermi Science Support Center (FSSC) image data
# For this example, we are using a sample URL where data are hosted
# You can modify this URL according to the specific Fermi data you are interested in
BASE_URL = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/bursts/2017/bn170817529/current/"

# Specify the path to the folder where you want to save the downloaded data
download_dir = "../fermi_data"

# Ensure the download directory exists
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Function to download the image or file from a given URL and save it locally
def download_image(url, filename):
    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Ensure the request was successful
        
        # Save the content to a file
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded {filename} successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename}: {e}")

# Example of downloading a gamma-ray sky image (you can modify URL for different data)
# Modify this according to the exact file path and name you want to download
# Here we assume a sample sky image URL, replace it with actual image URLs from Fermi data
data_url = BASE_URL + "glg_locprob_all_bn170817529_v02.fit"  # Replace with actual URL
data_filename = os.path.join(download_dir, "glg_locprob_all_bn170817529_v02.fit")

# Call the function to download the image
download_image(data_url, data_filename)