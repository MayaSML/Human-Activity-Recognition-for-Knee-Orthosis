import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path

from src.data_loader import merge_session
from src.preprocessing import preprocess
from src.windowing import make_windows
from src.feature_extraction import extract_features

LABEL_NAMES = ["walk", "sit", "run", "fall"]

TEST_DIR = Path("data/test")
ONNX_PATH = Path("models/har_model.onnx")


def main():

    acc_file = TEST_DIR / "test_Accelerometer.csv"
    gyro_file = TEST_DIR / "test_Gyroscope.csv"

    if not acc_file.exists():
        raise FileNotFoundError(acc_file)

    if not gyro_file.exists():
        raise FileNotFoundError(gyro_file)

    if not ONNX_PATH.exists():
        raise FileNotFoundError(ONNX_PATH)

    print("=" * 60)
    print("NOUVELLES DONNÉES PHYPHOX")
    print("=" * 60)


    df = merge_session(acc_file, gyro_file)

    df["label"] = 0

    df = preprocess(df)


    X_windows = []
    data = df[[
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z"
    ]].values

    WINDOW_SIZE = 200
    STEP = 100

    start = 0
    while start + WINDOW_SIZE <= len(data):
        X_windows.append(data[start:start + WINDOW_SIZE])
        start += STEP

    X_windows = np.array(X_windows, dtype=np.float32)

    print(f"\nFenêtres créées : {len(X_windows)}")

  
    X_features = extract_features(X_windows)

    print(f"Features : {X_features.shape}")

    session = ort.InferenceSession(str(ONNX_PATH))
    input_name = session.get_inputs()[0].name

    predictions = session.run(
        None,
        {input_name: X_features.astype(np.float32)}
    )[0]

    print("\nRésultats :")
    print("-" * 60)

    for i, pred in enumerate(predictions):
        print(
            f"Fenêtre {i+1:03d} -> "
            f"{LABEL_NAMES[int(pred)]}"
        )

    print("\nRésumé :")
    print("-" * 60)

    unique, counts = np.unique(predictions, return_counts=True)

    for label, count in zip(unique, counts):
        pct = count / len(predictions) * 100
        print(
            f"{LABEL_NAMES[int(label)]:<5} : "
            f"{count:>3} fenêtres ({pct:.1f}%)"
        )


if __name__ == "__main__":
    main()