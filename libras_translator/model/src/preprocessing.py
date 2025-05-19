import numpy as np
import pandas as pd

MAX_FRAMES = 30  # ajuste conforme seu modelo

def preprocess_landmarks(df):
    # Remove coluna 'frame' se existir
    if 'frame' in df.columns:
        df = df.drop(columns=['frame'])
    # Preenche NaN com zero
    df = df.fillna(0)
    # Garante que o número de frames seja MAX_FRAMES
    if len(df) < MAX_FRAMES:
        # Preenche com zeros se faltar frame
        padding = pd.DataFrame(np.zeros((MAX_FRAMES - len(df), df.shape[1])), columns=df.columns)
        df = pd.concat([df, padding], ignore_index=True)
    elif len(df) > MAX_FRAMES:
        # Trunca se tiver frames demais
        df = df.iloc[:MAX_FRAMES]
    # Retorna como numpy array
    return df.values.flatten()