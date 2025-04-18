# This file contains functions for downloading Fermi data from remote servers.
# Functions include downloading files, retrieving directories, and processing bursts or daily data.

import os
import re
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Function: download_file
# Input:
# - url (str): URL of the file to download.
# - filename (str): Local filename to save the downloaded file.
# Output:
# - int: 1 if download is successful, 0 otherwise.
def download_file(url, filename):
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return 1  # Successful download
    except requests.exceptions.RequestException:
        return 0  # Failed download

# Function: get_directories
# Input:
# - url (str): URL to retrieve directories from.
# Output:
# - list: List of directory names.
def get_directories(url):
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        return [link.get('href').strip('/') for link in soup.find_all('a') if link.get('href', '').endswith('/')]
    except requests.exceptions.RequestException:
        return []

# Function: get_available_versions
# Input:
# - data_url (str): URL to check for available versions.
# - identifier (str): Identifier for the data entry.
# - file_prefix (str): Prefix of the file to search for.
# Output:
# - list: Sorted list of available version numbers.
def get_available_versions(data_url, identifier, file_prefix):
    
    try:
        response = requests.get(data_url)
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        files = [a.text for a in soup.find_all('a')]

        version_numbers = []
        pattern = re.compile(f"{file_prefix}_{identifier}_v(\\d+).fit")
        for file in files:
            match = pattern.search(file)
            if match:
                version_numbers.append(int(match.group(1)))

        return sorted(version_numbers, reverse=True)  # Return versions from highest to lowest
    except Exception:
        return []
    
# Function: process_burst
# Input:
# - BASE_URL (str): Base URL for burst data.
# - download_dir (str): Directory to save downloaded files.
# - year (int): Year of the burst.
# - burst (str): Burst identifier.
# - file_prefix (str): Prefix of the file to download.
# - missing_bursts (list): List to store missing bursts.
# Output:
# - int: 1 if successful, 0 otherwise.
def process_burst(BASE_URL, download_dir, year, burst, file_prefix, missing_bursts):
    
    burst_url = f"{BASE_URL}{year}/{burst}/current/"
    available_versions = get_available_versions(burst_url, burst, file_prefix)

    if not available_versions:
        missing_bursts.append(burst)  # Add burst to missing list
        return 0  # No file found

    # Try to download the latest available version
    for version in available_versions:
        file_name_only = f"{file_prefix}_{burst}_v{version:02d}.fit"
        file_url = burst_url + file_name_only
        local_file = os.path.join(download_dir, file_name_only)

        if download_file(file_url, local_file):
            return 1  # Success

    return 0  # Failed to download

# Function: process_daily
# Input:
# - BASE_URL (str): Base URL for daily data.
# - download_dir (str): Directory to save downloaded files.
# - year (int): Year of the data.
# - month (int): Month of the data.
# - day (int): Day of the data.
# - file_prefix (str): Prefix of the file to download.
# Output:
# - int: 1 if successful, 0 otherwise.
def process_daily(BASE_URL, download_dir, year, month, day, file_prefix):
    
    short_year = str(year)[-2:]  # Take only the last two digits of the year
    daily_url = f"{BASE_URL}{year}/{month:02d}/{day:02d}/current/"
    identifier = f"{short_year}{month:02d}{day:02d}"
    available_versions = get_available_versions(daily_url, identifier, file_prefix)
    
    if not available_versions:
        return 0  # No file found

    for version in available_versions:
        file_name_only = f"{file_prefix}_{identifier}_v{version:02d}.fit"
        file_url = daily_url + file_name_only
        local_file = os.path.join(download_dir, file_name_only)

        if download_file(file_url, local_file):
            return 1  # Success
    
    return 0  # Failed to download

# Function: download_data
# Input:
# - year_range (range): Range of years to download data for.
# - Daily_or_Burst (str): Type of data ('Daily' or 'Burst').
# - url_file_string (str): File string to identify data.
# - output_dir (str): Directory to save downloaded data.
# Output:
# - None: Downloads data and saves missing entries to a file.
def download_data(year_range, Daily_or_Burst, url_file_string, output_dir):
    
    total_files_downloaded = 0
    tasks = []
    max_workers = 10  

    # Global list to store missing bursts
    missing_entries = []

    # Create download directory
    download_dir = f"./fermi_data/{output_dir}/"
    os.makedirs(download_dir, exist_ok=True)

    if Daily_or_Burst == 'Burst':
        BASE_URL = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/bursts/"
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for year in year_range:
                bursts = get_directories(f"{BASE_URL}{year}/")
                if not bursts:
                    continue
                
                for burst in bursts:
                    tasks.append(executor.submit(process_burst, BASE_URL, download_dir, year, burst, url_file_string, missing_entries))
            
            for future in tqdm(as_completed(tasks), total=len(tasks), desc="Downloading bursts"):
                total_files_downloaded += future.result()

    elif Daily_or_Burst == 'Daily':
        BASE_URL = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/"
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for year in year_range:
                for month in range(1, 13):
                    for day in range(1, 32):  # Handle incorrect dates later
                        tasks.append(executor.submit(process_daily, BASE_URL, download_dir, year, month, day, url_file_string))
            
            for future in tqdm(as_completed(tasks), total=len(tasks), desc="Downloading daily data"):
                total_files_downloaded += future.result()

    print(f"Total files downloaded: {total_files_downloaded}")

    # Save missing entries to a text file
    if len(missing_entries) != 0:
        with open(download_dir+"missing_entries_"+output_dir+".txt", "w") as f:
            for entry in missing_entries:
                f.write(f"{entry}\n")

        
# Run the main function
if __name__ == "__main__":
    download_data(range(2025, 2026), Daily_or_Burst='Burst', url_file_string="glg_bcat_all", output_dir='time')
    download_data(range(2025, 2026), Daily_or_Burst='Burst', url_file_string="glg_locprob_all", output_dir='loc_prob')
    download_data(range(2025, 2026), Daily_or_Burst='Daily', url_file_string="glg_poshist_all", output_dir='poshist')
