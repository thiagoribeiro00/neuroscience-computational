import os
import pandas as pd
import numpy as np

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
FLUX_COL = "flux"
FLUX_ERR_COL = "flux_err"

def process_file(filepath):
    df = pd.read_csv(filepath)
    # Remove pontos com erro alto (ex: acima do percentil 95)
    err_threshold = df[FLUX_ERR_COL].quantile(0.95)
    df = df[df[FLUX_ERR_COL] < err_threshold]
    # Remove outliers do fluxo (ex: z-score > 3)
    z_scores = np.abs((df[FLUX_COL] - df[FLUX_COL].mean()) / df[FLUX_COL].std())
    df = df[z_scores < 3]
    # Normaliza o fluxo entre 0 e 1
    flux_min = df[FLUX_COL].min()
    flux_max = df[FLUX_COL].max()
    df[FLUX_COL] = (df[FLUX_COL] - flux_min) / (flux_max - flux_min)
    # Inverte o sinal (queda de brilho = valor positivo)
    df[FLUX_COL] = 1.0 - df[FLUX_COL]
    return df

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for fname in os.listdir(RAW_DIR):
        if fname.endswith(".csv"):
            print(f"Processando {fname}...")
            df = process_file(os.path.join(RAW_DIR, fname))
            df.to_csv(os.path.join(PROCESSED_DIR, fname), index=False)
    print("Processamento concluído.")

if __name__ == "__main__":
    main()