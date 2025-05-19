import pandas as pd

def load_metadata(metadata_path):
    """
    Carrega o arquivo train.csv com os metadados dos vídeos.
    """
    return pd.read_csv(metadata_path)

def load_landmarks(landmarks_path):
    """
    Carrega um arquivo .parquet com os landmarks extraídos.
    """
    return pd.read_parquet(landmarks_path)