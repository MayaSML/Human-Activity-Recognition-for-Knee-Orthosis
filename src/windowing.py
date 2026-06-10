import numpy as np
import pandas as pd

SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
WINDOW_SIZE = 200
OVERLAP     = 100
STEP        = WINDOW_SIZE - OVERLAP  # = 100 samples = 1 seconde


def make_windows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:

    X_list, y_list = [], []

    for label_id in sorted(df["label"].unique()):
        data  = df[df["label"] == label_id][SENSOR_COLS].values  # (N, 6)
        start = 0
        while start + WINDOW_SIZE <= len(data):
            X_list.append(data[start : start + WINDOW_SIZE])
            y_list.append(label_id)
            start += STEP

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list,  dtype=np.int64)
    return X, y