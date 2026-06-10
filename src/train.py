import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

LABELS    = ["walk", "sit", "run", "fall"]
MODEL_DIR = Path("models")


def split_data(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Split 1 : extraire le test (15%)
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    trainval_idx, test_idx = next(s1.split(X, y))
    X_trainval, X_test = X[trainval_idx], X[test_idx]
    y_trainval, y_test = y[trainval_idx], y[test_idx]

    # Split 2 : séparer val (15%) du train (70%)
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=42)
    train_idx, val_idx = next(s2.split(X_trainval, y_trainval))
    X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

    print(f"  Train      : {len(y_train)} fenêtres")
    print(f"  Validation : {len(y_val)} fenêtres")
    print(f"  Test       : {len(y_test)} fenêtres")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train(
    X_features: np.ndarray,
    y: np.ndarray,
) -> tuple[RandomForestClassifier, StandardScaler]:

    print("  Split train / val / test...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_features, y)

    print("\n  Normalisation (StandardScaler)...")
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)  
    X_val_s   = scaler.transform(X_val)         
    X_test_s  = scaler.transform(X_test)        

    # Entraînement
    print("\n  Entraînement du Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,       
        max_depth=15,           
        min_samples_leaf=3,     
        class_weight="balanced", 
        random_state=42,        
        n_jobs=-1,              
    )
    model.fit(X_train_s, y_train)

    # Métriques sur les 3 splits
    acc_train = model.score(X_train_s, y_train)
    acc_val   = model.score(X_val_s,   y_val)
    acc_test  = model.score(X_test_s,  y_test)

    print(f"\n  Accuracy :")
    print(f"    Train      : {acc_train:.3f}")
    print(f"    Validation : {acc_val:.3f}")
    print(f"    Test       : {acc_test:.3f}")

    diff = acc_train - acc_val
    if diff > 0.1:
        print(f"\n Overfitting détecté (écart train/val = {diff:.3f})")
    else:
        print(f"\n Pas d'overfitting (écart train/val = {diff:.3f})")

    # Rapport détaillé
    y_pred = model.predict(X_test_s)
    print(f"\n  Rapport détaillé (Test) :")
    print(classification_report(y_test, y_pred, target_names=LABELS))

    print("  Matrice de confusion :")
    print("    Prédit →  walk  sit   run   fall")
    for i, row in enumerate(confusion_matrix(y_test, y_pred)):
        print(f"    Vrai {LABELS[i]:<5}: {row}")

    # Sauvegarde
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "model.pkl",  "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n  Sauvegardé dans {MODEL_DIR}/")

    return model, scaler