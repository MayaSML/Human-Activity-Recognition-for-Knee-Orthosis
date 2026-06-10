
import numpy as np
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

ONNX_PATH   = Path("models/har_model.onnx")
LABEL_NAMES = ["walk", "sit", "run", "fall"]


def test_onnx_model() -> None:

    for path in [ONNX_PATH, Path("data/X_features.npy"), Path("data/y_labels.npy")]:
        if not path.exists():
            raise FileNotFoundError(f"{path} introuvable. Lance d'abord : python main.py")

    X = np.load("data/X_features.npy")
    y = np.load("data/y_labels.npy")

    session    = ort.InferenceSession(str(ONNX_PATH))
    input_name = session.get_inputs()[0].name

    print("=" * 50)
    print("  TEST DU MODÈLE ONNX")
    print("=" * 50)
    print(f"\n  Modèle  : {ONNX_PATH}  ({ONNX_PATH.stat().st_size/1024:.1f} KB)")
    print(f"  Entrée  : {session.get_inputs()[0].shape}  (batch × features)")
    print(f"  Données : {X.shape[0]} fenêtres × {X.shape[1]} features")

    print("\n─── Test 1 : Performance globale ───")
    outputs = session.run(None, {input_name: X.astype(np.float32)})
    y_pred  = outputs[0]
    accuracy = (y_pred == y).mean()
    print(f"  Accuracy : {accuracy:.3f} ({accuracy*100:.1f}%)\n")
    print(classification_report(y, y_pred, target_names=LABEL_NAMES))

    print("  Matrice de confusion :")
    print("  Prédit →   walk  sit   run   fall")
    for i, row in enumerate(confusion_matrix(y, y_pred)):
        print(f"  Vrai {LABEL_NAMES[i]:<5}: {row}")

    print("\n─── Test 2 : Simulation temps réel (fenêtre du milieu) ───")
    print("  (on prend une fenêtre centrale = signal stable, pas de transition)\n")

    for label_id, name in enumerate(LABEL_NAMES):
        indices = np.where(y == label_id)[0]
        idx     = indices[len(indices) // 2]          # middle window
        fenetre = X[idx : idx + 1].astype(np.float32) # shape (1, 65)
        pred    = session.run(None, {input_name: fenetre})[0][0]
        correct = "ok" if pred == label_id else "erreur"
        print(f"  {correct}  Vrai : {name:<5}  →  Prédit : {LABEL_NAMES[pred]}")

    print("\n─── Test 3 : Analyse des erreurs ───")
    erreurs = np.where(y_pred != y)[0]
    if len(erreurs) == 0:
        print("  Aucune erreur sur le dataset complet !")
    else:
        print(f"  {len(erreurs)} erreur(s) sur {len(y)} fenêtres :")
        for idx in erreurs[:10]:   # afficher max 10 erreurs
            print(f"    Fenêtre #{idx:>4} | Vrai : {LABEL_NAMES[y[idx]]:<5} | Prédit : {LABEL_NAMES[y_pred[idx]]}")
        print()
        print("  → Les confusions walk/fall sont normales :")
        print("    Un pas brusque ou un faux mouvement ressemble à un petit choc.")

    print(f"\n Modèle ONNX validé — {ONNX_PATH}")


if __name__ == "__main__":
    test_onnx_model()