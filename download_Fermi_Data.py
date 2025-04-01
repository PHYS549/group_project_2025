import os
import re
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def download_file(url, filename):
    """Download a file from the given URL and save it locally."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return 1  # Successful download
    except requests.exceptions.RequestException:
        return 0  # Failed download

def get_directories(url):
    """Retrieve list of burst directories from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        return [link.get('href').strip('/') for link in soup.find_all('a') if link.get('href', '').endswith('/')]
    except requests.exceptions.RequestException:
        return []

def get_available_versions(burst_url, burst, file_prefix):
    """Retrieve a sorted list of available versions for a given burst."""
    try:
        response = requests.get(burst_url)
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        files = [a.text for a in soup.find_all('a')]

        version_numbers = []
        pattern = re.compile(f"{file_prefix}_{burst}_v(\\d+).fit")
        for file in files:
            match = pattern.search(file)
            if match:
                version_numbers.append(int(match.group(1)))

        return sorted(version_numbers, reverse=True)  # Return versions from highest to lowest
    except Exception:
        return []

def process_burst(BASE_URL, download_dir, year, burst, file_prefix, missing_bursts):
    """Process a single burst and download the latest available version."""
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

def process_burst_location(BASE_URL, download_dir, year, burst, missing_bursts):
    return process_burst(BASE_URL, download_dir, year, burst, "glg_locprob_all", missing_bursts)

def process_burst_time(BASE_URL, download_dir, year, burst, missing_bursts):
    return process_burst(BASE_URL, download_dir, year, burst, "glg_bcat_all", missing_bursts)

def download_data(year_range, data_type):
    """Download Fermi GBM burst data for the specified year range and data type."""
    BASE_URL = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/bursts/"
    download_dir = f"./fermi_data/{data_type}"
    os.makedirs(download_dir, exist_ok=True)

    total_files_downloaded = 0
    tasks = []
    max_workers = 10  

    # Global list to store missing bursts
    missing_bursts = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for year in year_range:
            bursts = get_directories(f"{BASE_URL}{year}/")
            if not bursts:
                continue
            
            for burst in bursts:
                if data_type == 'location':
                    tasks.append(executor.submit(process_burst_location, BASE_URL, download_dir, year, burst, missing_bursts))
                elif data_type == 'time':
                    tasks.append(executor.submit(process_burst_time, BASE_URL, download_dir, year, burst, missing_bursts))
        
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Downloading bursts"):
            total_files_downloaded += future.result()

    print(f"Total files downloaded: {total_files_downloaded}")

    # Save missing bursts to a text file
    if len(missing_bursts)!=0:
        with open("missing_bursts_"+data_type+".txt", "w") as f:
            for burst in missing_bursts:
                f.write(f"{burst}\n")

# Run the main function
if __name__ == "__main__":
    download_data(range(2015, 2026), data_type='location')
    download_data(range(2015, 2026), data_type='time')
