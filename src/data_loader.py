import pandas as pd
import numpy as np
from pathlib import Path

HZ        = 100
LABEL_MAP = {"walk": 0, "sit": 1, "run": 2, "fall": 3}


def merge_session(acc_path: Path, gyro_path: Path) -> pd.DataFrame:
    acc  = pd.read_csv(acc_path)
    gyro = pd.read_csv(gyro_path)

    acc  = acc.rename(columns={
        "Time (s)": "time", "X (m/s^2)": "acc_x",
        "Y (m/s^2)": "acc_y", "Z (m/s^2)": "acc_z",
    })
    gyro = gyro.rename(columns={
        "Time (s)": "time", "X (rad/s)": "gyro_x",
        "Y (rad/s)": "gyro_y", "Z (rad/s)": "gyro_z",
    })

    # Période commune aux deux capteurs
    t_start = max(acc["time"].min(),  gyro["time"].min())
    t_end   = min(acc["time"].max(),  gyro["time"].max())
    grid    = np.arange(t_start, t_end, 1.0 / HZ)

    # Interpolation de chaque signal sur la grille
    df = pd.DataFrame({"time": grid})
    for col in ["acc_x", "acc_y", "acc_z"]:
        df[col] = np.interp(grid, acc["time"].values, acc[col].values)
    for col in ["gyro_x", "gyro_y", "gyro_z"]:
        df[col] = np.interp(grid, gyro["time"].values, gyro[col].values)

    return df


def load_all(data_dir: Path) -> pd.DataFrame:
    all_dfs = []

    for name, label in LABEL_MAP.items():
        df = merge_session(
            data_dir / name / f"{name}_Accelerometer.csv",
            data_dir / name / f"{name}_Gyroscope.csv",
        )
        df["label"] = label
        print(f"  {name:<5} | {len(df):>6} samples | {len(df)/HZ:.0f}s")
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)