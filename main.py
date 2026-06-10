import time
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

from src.data_loader import load_all
from src.preprocessing import preprocess
from src.windowing import make_windows
from src.feature_extraction import extract_features
from src.train import train

# ── Chemins ───────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
OUTPUT_DIR = Path("data")
MODEL_DIR  = Path("models")
ONNX_PATH  = MODEL_DIR / "har_model.onnx"

LABEL_NAMES = ["walk", "sit", "run", "fall"]


def section(titre: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {titre}")
    print(f"{'=' * 55}")


def export_onnx(model, scaler, n_features: int) -> Path:
    MODEL_DIR.mkdir(exist_ok=True)

    pipeline = Pipeline([
        ("scaler",     scaler),
        ("classifier", model),
    ])

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model   = convert_sklearn(pipeline, initial_types=initial_type, target_opset=17)

    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    size_kb = ONNX_PATH.stat().st_size / 1024
    print(f"  Fichier : {ONNX_PATH}  ({size_kb:.1f} KB)")
    return ONNX_PATH


def benchmark_latence(onnx_path: Path, n_features: int, n_runs: int = 1000) -> None:

    session    = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    dummy      = np.random.randn(1, n_features).astype(np.float32)

    for _ in range(20):
        session.run(None, {input_name: dummy})

    latences_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latences_ms.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latences_ms)
    print(f"  Inférences : {n_runs}")
    print(f"  Moyenne    : {lat.mean():.3f} ms")
    print(f"  P50        : {np.percentile(lat, 50):.3f} ms")
    print(f"  P95        : {np.percentile(lat, 95):.3f} ms")
    print(f"  P99        : {np.percentile(lat, 99):.3f} ms")
    print(f"  Seuil requis : < 1000 ms  →  ", end="")
    print(" OK" if np.percentile(lat, 99) < 1000 else "  trop lent")


def main() -> None:

    section("ÉTAPE 1 — Chargement des données")
    df = load_all(DATA_DIR)
    print(f"\n  Total : {len(df):,} samples | durée : {len(df)/100:.0f}s")

    section("ÉTAPE 2 — Preprocessing")
    print("  → Suppression NaN")
    print("  → Écrêtage outliers ±5σ (sauf fall : ses pics sont réels)")
    print("  → Filtre Butterworth passe-bas 20 Hz ordre 4")
    df = preprocess(df)
    print(f"  {len(df):,} samples après nettoyage")

    section("ÉTAPE 3 — Fenêtrage glissant")
    print("  → Taille  : 200 samples = 2s à 100 Hz")
    print("  → Overlap : 100 samples = 50%  →  step = 1s")
    print("  → Traité par activité (pas de fenêtres mixtes)")
    X_windows, y = make_windows(df)
    print(f"\n  {X_windows.shape[0]} fenêtres extraites :")
    for i, name in enumerate(LABEL_NAMES):
        print(f"    {name:<5} (label={i}) : {(y==i).sum():>4} fenêtres")

    section("ÉTAPE 4 — Extraction de features")
    print("  → 10 features × 6 canaux = 60 features temporelles/spectrales")
    print("  → 5 features inter-canaux (magnitude acc + gyro)")
    print("  → Total : 65 features par fenêtre")
    X_features = extract_features(X_windows)
    print(f"\n  Shape : {X_features.shape}  (fenêtres × features)")

    OUTPUT_DIR.mkdir(exist_ok=True)
    np.save(OUTPUT_DIR / "X_features.npy", X_features)
    np.save(OUTPUT_DIR / "y_labels.npy",   y)
    print("  Sauvegardé : data/X_features.npy + data/y_labels.npy")

    section("ÉTAPE 5 — Entraînement du modèle")
    print("  → Split stratifié 70% train / 15% val / 15% test")
    print("  → StandardScaler appris sur train uniquement (pas de data leakage)")
    print("  → Random Forest : 200 arbres, max_depth=15, class_weight=balanced\n")
    model, scaler = train(X_features, y)

    section("ÉTAPE 6 — Export ONNX")
    export_onnx(model, scaler, n_features=X_features.shape[1])

    section("ÉTAPE 7 — Benchmark latence (1000 inférences)")
    benchmark_latence(ONNX_PATH, n_features=X_features.shape[1])

    section(" PIPELINE TERMINÉ")
    print(f"  Modèle ONNX exporté → {ONNX_PATH}")
    print(f"  Pour tester sur tes données : python test_model.py")


if __name__ == "__main__":
    main()