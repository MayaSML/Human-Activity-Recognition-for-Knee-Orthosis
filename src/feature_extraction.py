import numpy as np
from scipy import stats

HZ = 100


def _features_one_window(window: np.ndarray) -> np.ndarray:
    feats = []

    for ch in range(window.shape[1]):
        s = window[:, ch]

        feats.append(float(np.mean(s)))
        feats.append(float(np.std(s)))
        feats.append(float(np.min(s)))
        feats.append(float(np.max(s)))
        feats.append(float(np.sqrt(np.mean(s ** 2))))  
        feats.append(float(stats.skew(s)))             # asymétrie
        feats.append(float(stats.kurtosis(s)))         # pics

        # Taux de passage par zéro
        zcr = np.sum(np.diff(np.sign(s)) != 0) / len(s)
        feats.append(float(zcr))

        # Fréquence dominante - FFT
        fft_vals    = np.abs(np.fft.rfft(s))
        fft_vals[0] = 0 
        freqs       = np.fft.rfftfreq(len(s), d=1.0 / HZ)
        feats.append(float(freqs[np.argmax(fft_vals)]))

        # Énergie spectrale
        feats.append(float(np.sum(fft_vals ** 2) / len(s)))

    # Magnitude accéléromètre : sqrt(x²+y²+z²)
    # --> Indépendante de l'orientation du téléphone dans la poche
    acc_mag = np.sqrt(np.sum(window[:, :3] ** 2, axis=1))
    feats  += [float(np.mean(acc_mag)), float(np.std(acc_mag)), float(np.max(acc_mag))]

    # Magnitude gyroscope
    gyro_mag = np.sqrt(np.sum(window[:, 3:] ** 2, axis=1))
    feats   += [float(np.mean(gyro_mag)), float(np.std(gyro_mag))]

    return np.array(feats, dtype=np.float32)


def extract_features(X_windows: np.ndarray) -> np.ndarray:

    return np.vstack([_features_one_window(w) for w in X_windows])


def get_feature_names() -> list[str]:
    channels    = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    stats_names = ["mean", "std", "min", "max", "rms", "skew", "kurt",
                   "zcr", "dom_freq", "spectral_energy"]
    names  = [f"{ch}_{st}" for ch in channels for st in stats_names]
    names += ["acc_mag_mean", "acc_mag_std", "acc_mag_max",
              "gyro_mag_mean", "gyro_mag_std"]
    return names