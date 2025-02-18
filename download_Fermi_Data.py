import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def download_data(year_range):
    # Base URL for the Fermi GBM burst data
    BASE_URL = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/bursts/"

    # Specify the local directory to save downloaded files
    download_dir = "./fermi_data"

    # Ensure the download directory exists
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Counter for successfully downloaded files
    total_files_downloaded = 0

    # We'll use a ThreadPoolExecutor to process bursts concurrently
    max_workers = 10  # Adjust the number of workers based on your system and network
    tasks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for year in year_range:
            year_url = f"{BASE_URL}{year}/"
            
            # Get list of bursts for the year
            bursts = get_directories(year_url)
            if not bursts:
                continue
            
            # Submit a task for each burst
            for burst in bursts:
                tasks.append(executor.submit(process_burst, BASE_URL, download_dir, year, burst, total_files_downloaded))
        
        # Use tqdm to display progress while waiting for tasks to complete
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Downloading bursts"):
            # Increment the counter based on the result from process_burst
            total_files_downloaded += future.result()

    # Print the total number of files successfully downloaded
    print(f"Total files downloaded: {total_files_downloaded}")

def download_file(url, filename):
    """
    Download a file from the given URL and save it locally.
    Returns 1 if download is successful, otherwise 0.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error on a bad status
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return 1  # Return 1 on successful download
    except requests.exceptions.RequestException:
        return 0  # Return 0 if download fails

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
    except requests.exceptions.RequestException:
        return []

def process_burst(BASE_URL, download_dir, year, burst, total_files_downloaded):
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
    
    if downloaded:
        return 1  # Successfully downloaded
    else:
        return 0  # Failed to download

# Run the main function
if __name__ == "__main__":
    download_data(range(2010, 2026))
