import numpy as np
import pandas as pd
from data_utils import download_tess_data, preprocess_light_curve
from encoding import rate_encoder
from models import SNNExoplanetDetector, LSTMExoplanetDetector
import torch

def main():
    # Define parameters
    target_list = ['TIC 1234567', 'TIC 7654321']  # Example target list
    labels = [1, 0]  # Corresponding labels for the targets
    data_dir = "data"

    # Download and preprocess data
    for target, label in zip(target_list, labels):
        download_tess_data([target], label, data_dir)
        # Load the downloaded data
        flux_data = pd.read_csv(f"{data_dir}/{target.replace(' ', '_')}_label_{label}.csv")['flux'].values
        processed_flux = preprocess_light_curve(flux_data)

        # Encode the processed flux into spikes
        spike_train = rate_encoder(processed_flux)

        # Initialize models
        input_features = spike_train.shape[1]
        hidden_features = 64  # Example hidden features
        output_features = 1  # Binary classification

        snn_model = SNNExoplanetDetector(input_features, hidden_features, output_features)
        lstm_model = LSTMExoplanetDetector(input_size=input_features, hidden_size=64, num_layers=2, output_size=output_features)

        # Here you would typically train the models and evaluate their performance
        # For now, we will just print the model architectures
        print(snn_model)
        print(lstm_model)

if __name__ == "__main__":
    main()