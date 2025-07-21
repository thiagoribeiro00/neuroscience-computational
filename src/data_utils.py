import lightkurve as lk
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

def download_tess_data(target_list, label, data_dir="data"):
    """
    Downloads light curves for a list of TESS targets and saves them as CSV files.

    Args:
        target_list (list): List of target IDs (e.g., 'TIC 1234567') or names.
        label (int): 0 for non-exoplanet (false positive), 1 for exoplanet.
        data_dir (str): Directory to save the CSV files.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print(f"Downloading data for label: {label}...")
    for target in tqdm(target_list):
        try:
            # Search for the light curve
            search_result = lk.search_lightcurve(target, mission='TESS', author='SPOC')
            # Download the light curve with the best data (SPOC pipeline)
            lc_collection = search_result.download_all()
            # Stitch all observed sectors and remove NaNs
            lc = lc_collection.stitch().remove_nans()
            
            # Save in a simple format
            df = lc.to_pandas()[['time', 'flux', 'flux_err']]
            # Clean the filename
            sanitized_target = target.replace(" ", "_")
            output_path = os.path.join(data_dir, f"{sanitized_target}_label_{label}.csv")
            df.to_csv(output_path, index=False)
        except Exception as e:
            print(f"Error processing {target}: {e}")

def preprocess_light_curve(flux_data, target_length=2048):
    """
    Preprocesses a flux time series.

    1. Normalizes the flux to the range [0, 1].
    2. Inverts the signal (brightness drops become positive peaks).
    3. Resizes the series to a fixed length (padding/truncating).
    """
    # 1. Normalization (Min-Max Scaler)
    flux_min = np.min(flux_data)
    flux_max = np.max(flux_data)
    if flux_max > flux_min:
        normalized_flux = (flux_data - flux_min) / (flux_max - flux_min)
    else:
        normalized_flux = np.zeros_like(flux_data)

    # 2. Invert the signal so that transits become positive signals
    inverted_flux = 1.0 - normalized_flux

    # 3. Resize to a fixed length
    current_length = len(inverted_flux)
    if current_length > target_length:
        # Truncate the center of the series
        start = (current_length - target_length) // 2
        processed_flux = inverted_flux[start : start + target_length]
    else:
        # Add padding with zeros (no transit signal)
        padding = target_length - current_length
        pad_start = padding // 2
        pad_end = padding - pad_start
        processed_flux = np.pad(inverted_flux, (pad_start, pad_end), 'constant')

    return processed_flux.astype(np.float32)