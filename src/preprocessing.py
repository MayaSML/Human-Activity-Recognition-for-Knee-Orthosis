import pandas as pd
import numpy as np
from scipy import signal

HZ          = 100
SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
FALL_LABEL  = 3


def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # 1. Supprimer les lignes avec valeurs manquantes
    df = df.dropna().reset_index(drop=True)

    # 2. Écrêter les outliers sur walk/sit/run uniquement
    mask_not_fall = df["label"] != FALL_LABEL
    for col in SENSOR_COLS:
        mean = df.loc[mask_not_fall, col].mean()
        std  = df.loc[mask_not_fall, col].std()
        lo, hi = mean - 5 * std, mean + 5 * std
        df.loc[mask_not_fall, col] = df.loc[mask_not_fall, col].clip(lo, hi)

    # 3. Filtre passe-bas Butterworth 20 Hz — filtré par activité
    b, a = signal.butter(N=4, Wn=20 / (HZ / 2), btype="low")
    for label_id in df["label"].unique():
        idx = df["label"] == label_id
        for col in SENSOR_COLS:
            df.loc[idx, col] = signal.filtfilt(b, a, df.loc[idx, col].values)

    return df