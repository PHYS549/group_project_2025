import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.ticker import ScalarFormatter

def load_fits_data(fits_file, data_type):
    with fits.open(fits_file) as hdul:
        header = hdul[1].header if data_type == 'location' else hdul[0].header
        return (header['CRVAL1'], header['CRVAL2']) if data_type == 'location' else (header['DATE'], header['T90'])

def process_fits_files(fits_folder, data_type):
    files = [f for f in os.listdir(fits_folder) if f.endswith(('.fit', '.fits'))]
    columns = ['RA', 'DEC'] if data_type == 'location' else ['DATE', 'T90']
    df = pd.DataFrame(columns=columns)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_fits_data, os.path.join(fits_folder, f), data_type) for f in files]
        for future in as_completed(futures):
            data = future.result()
            df = pd.concat([df, pd.DataFrame([data], columns=columns)], ignore_index=True)
    
    output_folder = "./plots"
    os.makedirs(output_folder, exist_ok=True)
    df.to_csv(os.path.join(output_folder, f"{data_type}_data.csv"), index=False)
    return df

def plot_data(df, output_folder, data_type):
    plt.figure(figsize=(10, 6))
    if data_type == 'location':
        plt.scatter(df['RA'], df['DEC'], s=10, color='blue', alpha=0.5)
        plt.xlabel('RA (DEG)'); plt.ylabel('DEC (DEG)')
        plt.title('GRB Events Distribution')
        plt.savefig(os.path.join(output_folder, "RA_DEC_plot.png"))
    else:
        # Convert seconds to years since 2015-01-01
        seconds_in_year = 365.25 * 24 * 60 * 60  # 365.25 days/year
        date_years = (pd.to_datetime(df['DATE']) - pd.Timestamp("2015-01-01")).dt.total_seconds() / seconds_in_year

        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(date_years, bins=200, color='skyblue', alpha=0.7)
        plt.xlabel('Years since 2015-01-01')
        plt.ylabel('Number of Events')
        plt.title('GRB Events Over Time (in Years)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "GRB_events_over_time_years.png"))
        plt.close()


        t90_values = pd.to_numeric(df['T90'], errors='coerce').dropna()
        valid_t90 = np.log10(t90_values[(t90_values > 0) & (t90_values.between(1e-3, 1e3))])

        plt.figure(figsize=(10, 6))
        plt.hist(valid_t90, bins=200, color='green', alpha=0.7)

        # Formatting the x-axis label to display as scientific notation
        plt.xlabel(r'Log10(T90 Duration) [s]')
        plt.ylabel('Number of Events')
        plt.title('T90 Duration Distribution (Log Scale)')

        # Customize the x-axis ticks to display in scientific notation
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        # Save the plot
        plt.savefig(os.path.join(output_folder, "T90_distribution_log.png"))
        plt.close()

def main():
    output_folder = "./plots"
    loc_df = process_fits_files("./fermi_data/location", 'location')
    time_df = process_fits_files("./fermi_data/time", 'time')
    plot_data(loc_df, output_folder, 'location')
    plot_data(time_df, output_folder, 'time')

if __name__ == "__main__":
    main()
