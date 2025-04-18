import pandas as pd

from gw_grb_correlation.Fermi import download_and_preprocess_fermi_data
from gw_grb_correlation.Fermi.visualization import plot_all_detector_positions, create_time_data_plots, create_location_data_plots
from gw_grb_correlation.Fermi.util import duration, filtering
"""from fermi_poshist_data import plot_all_detector_positions
from fermi_time_data import create_time_data_plots, duration, filtering
from fermi_location_data import create_location_data_plots"""

def read_GW_data(file_path):
    """
    Reads the GW data from a CSV file into a Pandas DataFrame.
    
    :param file_path: Path to the CSV file.
    :return: Pandas DataFrame containing the CSV data.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data from {file_path} loaded successfully.")
        print(df.info())  # Display DataFrame info for verification
        print(df.head())  # Show the first few rows for preview
        return df
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None
    
def remove_duplicate_times_in_gw_data(gw_data, threshold=1.0):
    """
    Removes 'times' values from gw_data that are within a threshold of each other (in seconds).
    
    :param gw_data: DataFrame containing GW data with 'times' column
    :param threshold: The maximum allowed difference between two times to be considered "duplicate" (in seconds).
    :return: DataFrame with times that are separated by more than the threshold only.
    """
    # Define the start date
    start_date = pd.Timestamp('1980-01-06')

    # Convert 'times' (in seconds) to datetime by adding them to the start_date
    gw_data['date'] = start_date + pd.to_timedelta(gw_data['times'], unit='s')

    # Sort by 'times' to compare successive values
    gw_data_sorted = gw_data.sort_values(by='times').reset_index(drop=True)

    # Initialize a list to hold the filtered times
    filtered_times = [gw_data_sorted.iloc[0]]  # Always keep the first entry

    # Iterate through the sorted data to find times within the threshold
    for i in range(1, len(gw_data_sorted)):
        if gw_data_sorted.loc[i, 'times'] - gw_data_sorted.loc[i - 1, 'times'] > threshold:
            filtered_times.append(gw_data_sorted.iloc[i])

    # Convert the list of filtered times back into a DataFrame
    gw_data_unique = pd.DataFrame(filtered_times)


    print(f"Number of unique times after applying threshold: {len(gw_data_unique)}")
    print(f"First few unique times:\n{gw_data_unique[['times', 'date']].head(15)}")

    return gw_data_unique

def compare_time_within_range(fermi_data, gw_data, time_range_seconds=86400):
    """
    Compares 'TSTART' in fermi_data and 'times' in gw_data to check if they are within a specified time range.
    
    :param fermi_data: DataFrame containing Fermi data with 'TSTART' column
    :param gw_data: DataFrame containing GW data with 'times' column
    :param time_range_seconds: Time range in seconds to compare the two times (default is 86400 seconds = 1 day)
    :return: DataFrame with matched data within the specified time range
    """
    # Convert 'TSTART' and 'times' to seconds since their respective starting dates
    fermi_start_date = pd.to_datetime('2001-01-01')
    gw_start_date = pd.to_datetime('1980-01-06')
    
    # Convert 'TSTART' and 'times' to numeric and then to seconds since the respective start dates
    fermi_data['ALIGNED_SEC'] = pd.to_numeric(fermi_data['TSTART'], errors='coerce')
    gw_data['ALIGNED_SEC'] = pd.to_numeric(gw_data['times'], errors='coerce')
    
    # Convert to seconds since the respective starting date
    fermi_data['ALIGNED_SEC'] = fermi_data['ALIGNED_SEC'] + (fermi_start_date - pd.Timestamp("1980-01-01")).total_seconds()
    gw_data['ALIGNED_SEC'] = gw_data['ALIGNED_SEC'] + (gw_start_date - pd.Timestamp("1980-01-01")).total_seconds()

    # Initialize an empty list to store the matches
    matched_data = []

    # Compare each 'ALIGNED_SEC' in fermi_data with each 'ALIGNED_SEC' in gw_data
    for _, fermi_row in fermi_data.iterrows():
        for _, gw_row in gw_data.iterrows():
            if abs(fermi_row['ALIGNED_SEC'] - gw_row['ALIGNED_SEC']) <= time_range_seconds:  # Check if time difference is within the specified range
                # Create a dictionary to store the matched data, including all fermi_data columns
                match = {'grb_time': fermi_row['TSTART'],
                         'grb_ra': fermi_row['RA'],
                         'grb_dec': fermi_row['DEC'], 
                         'gw_time': gw_row['times'], 
                         'time_diff': abs(fermi_row['ALIGNED_SEC'] - gw_row['ALIGNED_SEC']),
                         'GRB_ID': fermi_row['ID']}
                matched_data.append(match)

    # Create a DataFrame from the matches
    matched_df = pd.DataFrame(matched_data)

    if len(matched_df)==0:
        print(f"No matching times found within {time_range_seconds} seconds.")
    else:
        print(f"Found {len(matched_df)} matches within {time_range_seconds} seconds.")
    
    return matched_df

if __name__ == "__main__":

    # Data acquisition and data wrangling
    start_year = 2025
    end_year = 2026
    fermi_data = download_and_preprocess_fermi_data(start_year=start_year, end_year=end_year, download_or_not=False)

    # Explore the dataset
    create_time_data_plots(fermi_data, 'plots')
    create_location_data_plots(fermi_data, 'plots')
    plot_all_detector_positions(fermi_data.head(9))
    
    # Filter out short GRB data
    short_GRB_data = filtering(fermi_data, criteria={'T90': lambda x: x <= 2.1})

    # Calculate the duration (difference between max TSTOP and min TSTART) and count the short GRB event number
    duration_short_GRB = duration(short_GRB_data)
    event_num_short_GRB = len(short_GRB_data)

    # Calculate the average occurrence rate (events per unit of time)
    average_occurrence_rate = event_num_short_GRB / duration_short_GRB

    # Print the results
    print(f"Number of events: {event_num_short_GRB}")
    print(f"Total duration: {duration_short_GRB} seconds")
    print(f"Average occurrence rate: {average_occurrence_rate} events per second")

    print(short_GRB_data[short_GRB_data['ID'] == 'bn170817529'])

    # Load GW data
    gw_data = read_GW_data(f"./gw_data/totalgwdata.csv")
    gw_times = remove_duplicate_times_in_gw_data(gw_data)

    # Find matched GRB-GW event pairs
    match = compare_time_within_range(short_GRB_data, gw_times, time_range_seconds=86400*3)
    filtered_gw_events = gw_data[gw_data['times'].isin(match['gw_time'])]
    match.to_csv("GRB_GW_event_pairs.csv", index=False)
    filtered_gw_events.to_csv("Filtered_GW_events.csv", index=False)
    