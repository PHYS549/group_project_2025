"""import os
import requests

# Base URL for the Fermi Science Support Center (FSSC) image data
# For this example, we are using a sample URL where data are hosted
# You can modify this URL according to the specific Fermi data you are interested in
BASE_URL = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/bursts/2017/bn170817529/current/"

# Specify the path to the folder where you want to save the downloaded data
download_dir = "./fermi_data"

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
download_image(data_url, data_filename)"""

import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Base URL for the Fermi GBM burst data
BASE_URL = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/bursts/"

# Specify the local directory to save downloaded files
download_dir = "./fermi_data"

# Ensure the download directory exists
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

def download_file(url, filename):
    """
    Download a file from the given URL and save it locally.
    Returns True if download is successful, otherwise False.
    """
    try:
        print(f"Attempting to download {filename} from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error on a bad status
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"Downloaded {filename} successfully.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename}: {e}")
        return False

def get_directories(url):
    """
    Scrape a directory listing from the given URL and return a list
    of directory names (without trailing slashes).
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        directories = []
        for link in soup.find_all('a'):
            href = link.get('href')
            # Only include directories (ending with '/') and ignore parent links
            if href and href not in ['../', '/'] and href.endswith('/'):
                directories.append(href.strip('/'))
        return directories
    except requests.exceptions.RequestException as e:
        print(f"Error fetching directories from {url}: {e}")
        return []

def process_burst(year, burst):
    """
    Process a single burst: Try to download the file using two version formats.
    """
    burst_url = f"{BASE_URL}{year}/{burst}/current/"
    versions = ['v02', 'v00']
    
    # Local path remains the base download directory; adjust as needed
    local_dir = download_dir
    downloaded = False
    for version in versions:
        file_name_only = f"glg_locprob_all_{burst}_{version}.fit"
        file_url = burst_url + file_name_only
        local_file = os.path.join(local_dir, file_name_only)
        
        # Attempt download for this version
        if download_file(file_url, local_file):
            downloaded = True
            break  # Exit loop on successful download
    
    if not downloaded:
        print(f"Failed to download glg_locprob file for burst {burst} in year {year}.")

# We'll use a ThreadPoolExecutor to process bursts concurrently
max_workers = 10  # Adjust the number of workers based on your system and network
tasks = []

# For demonstration purposes, we'll work with a single year (e.g., 2025)
# Change the range to (2008, 2026) to cover 2008 to 2025.
years = range(2025, 2026)

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    for year in years:
        year_url = f"{BASE_URL}{year}/"
        print(f"\nProcessing year: {year}")
        
        # Get list of bursts for the year
        bursts = get_directories(year_url)
        if not bursts:
            print(f"No bursts found for {year}.")
            continue
        
        # Submit a task for each burst
        for burst in bursts:
            tasks.append(executor.submit(process_burst, year, burst))
    
    # Use tqdm to display progress while waiting for tasks to complete
    for future in tqdm(as_completed(tasks), total=len(tasks), desc="Downloading bursts"):
        # Calling result() to re-raise any exceptions that occurred in worker threads
        future.result()

