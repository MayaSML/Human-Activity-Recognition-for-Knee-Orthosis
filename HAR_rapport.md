<small>

# REEV- HAR : Reconnaissance d'Activité pour Orthèse de Genou Motorisée
---
## 1. Contexte et objectif

Ce projet vise à classifier en temps réel **4 activités humaines** à partir de signaux inertiels (IMU) d'un smartphone, en vue du déploiement sur une orthèse de genou motorisée. Les classes cibles sont : `marche`, `assis`, `course`, `chute`.

Les contraintes système imposées sont une exécution embarquée temps réel, un modèle à empreinte réduite, et un pipeline entièrement reproductible.


## 2. Acquisition et données brutes

**Matériel :** application Phyphox — accéléromètre et gyroscope (3 axes), à 100 Hz, positionné en poche avant.

| Activité | Samples | Durée |
|----------|--------:|------:|
| Marche   | 30 927  | ~5 min |
| Chute    | 32 433  | ~5 min 24 s |
| Course   | 22 493  | ~3 min 45 s |
| Assis    | 16 363  | ~2 min 44 s |
| **Total**| **102 216** | **~17 min** |

Le déséquilibre de classes entre `assis` (16 k) et `chute` (32 k) est compensé en aval via `class_weight='balanced'`.

## 3. Prétraitement

Le pipeline de nettoyage opère les étapes suivantes sur les CSV bruts :

1. **Suppression des NaN** et détection des artefacts capteur
2. **Clipping des valeurs extrêmes** à ±5σ (hors classe `chute`, pour conserver les pics d'impact)
3. **Synchronisation accéléromètre / gyroscope** par interpolation linéaire à 100 Hz
4. **Filtrage Butterworth passe-bas** (ordre 4, coupure à 20 Hz)- les mouvements humains étant majoritairement < 10 Hz, ce filtre supprime le bruit haute fréquence tout en préservant l'information cinématique utile

Le fichier résultant `dataset.csv` contient 102 216 lignes et 7 colonnes (`acc_x`, `acc_y`, `acc_z`, `gyro_x`, `gyro_y`, `gyro_z`, `label`).

## 4. Feature Engineering

### Fenêtrage glissant

| Paramètre    | Valeur |
|--------------|-------:|
| Taille       | 200 samples (2 s) |
| Recouvrement | 50 % (step = 1 s) |
| Fenêtres totales | 1 016 |

Une fenêtre de 2 s capture un cycle locomoteur complet ; le recouvrement de 50 % augmente le corpus d'entraînement et réduit le risque de manquer des événements courts (chute ~0,5–1 s).

### Extraction de caractéristiques - 65 features par fenêtre

Chaque fenêtre brute (200 × 6 = 1 200 valeurs) est résumée en 65 caractéristiques :

**Par canal IMU (× 6) :** `mean`, `std`, `min`, `max`, `RMS`, `skewness`, `kurtosis`, `zero-crossing rate`, `fréquence dominante (FFT)`, `énergie spectrale`

**Features globales :** magnitude accéléromètre (`mean`, `std`, `max`) + magnitude gyroscope (`mean`, `std`)

Ce résumé statistique est particulièrement adapté à un Random Forest : dimensionnalité maîtrisée, invariance aux décalages temporels, interprétabilité directe.

## 5. Modèle et entraînement

**Algorithme :** Random Forest Classifier (scikit-learn)

| Hyperparamètre     | Valeur | Justification |
|--------------------|-------:|---------------|
| `n_estimators`     | 200    | Stabilise la variance sans sur-coût d'inférence |
| `max_depth`        | 15     | Limite l'overfitting sur un corpus de ~17 min |
| `min_samples_leaf` | 3      | Renforce la généralisation |
| `class_weight`     | balanced | Compense le déséquilibre inter-classes |

**Normalisation :** `StandardScaler` ajusté sur le train set uniquement, appliqué au test set.

## 6. Résultats

### Performance globale

| Ensemble   | Accuracy |
|------------|--------:|
| Train      | 0.999   |
| Validation | 0.987   |
| Test       | 0.987   |

L'écart train / validation de 1,2 point indique un léger surajustement, sans impact opérationnel significatif.

### Rapport de classification (test set)

| Classe | Precision | Recall | F1-score | Support |
|--------|----------:|-------:|---------:|--------:|
| walk   | 1.00 | 0.96 | 0.98 | 46 |
| sit    | 1.00 | 1.00 | 1.00 | 24 |
| run    | 1.00 | 1.00 | 1.00 | 34 |
| fall   | 0.96 | 1.00 | 0.98 | 49 |

**Accuracy globale : 0.987**

### Matrice de confusion (test set)

|       | walk | sit | run | fall |
|-------|-----:|----:|----:|-----:|
| walk  | 44   | 0   | 0   | 2    |
| sit   | 0    | 24  | 0   | 0    |
| run   | 0    | 0   | 34  | 0    |
| fall  | 0    | 0   | 0   | 49   |

Les 2 erreurs observées (walk → fall) correspondent à des transitions brusques lors de la mise en poche du smartphone, générant des pics d'accélération similaires à un impact de chute. Aucune confusion entre classes cinématiquement distinctes (course / assis).

### Validation sur données hors-distribution

Sur une session indépendante non utilisée à l'entraînement (233 fenêtres) : **99,1 % de prédictions correctes**, erreurs concentrées sur les transitions d'activité. La validation est à ce stade limitée à la classe `marche` - une évaluation multi-classes sur nouvelles sessions reste nécessaire.

## 7. Déploiement et performance d'inférence

Le pipeline complet (scaler + modèle) est exporté au format **ONNX** (`191 KB`), compatible avec les runtimes embarqués légers (ONNX Runtime...).

| Métrique d'inférence | Valeur |
|----------------------|-------:|
| Latence moyenne      | 0.018 ms |
| P95                  | 0.023 ms |
| P99                  | 0.036 ms |

Le système produit une prédiction toutes les **1 seconde** (fenêtrage glissant). La latence d'inférence est **55× inférieure** à la période de décision, rendant le système largement compatible avec un usage temps réel embarqué.

*Mesures réalisées sur CPU standard (benchmark local). Les performances sur cible embarquée (MCU, smartphone) peuvent varier.*

## 8. Limites et perspectives

| Limite actuelle | Impact | Piste d'amélioration |
|-----------------|--------|----------------------|
| Dataset limité (~17 min, 1 sujet) | Généralisation incertaine | Collecte multi-sujets, multi-sessions |
| Chutes simulées | Distribution non réaliste | Protocole de chute contrôlée en labo |
| Position smartphone fixe | Sensibilité aux variations d'orientation | Data augmentation + test multi-positions |
| Validation partielle (marche uniquement) | Robustesse inter-classes non confirmée | Test sur toutes les classes hors-distribution |

## 9. Conclusion

ce pipeline permet la reconnaissance de 4 activités humaines avec une accuracy de **98,7 % sur données de test** et une latence d'inférence de **0,018 ms** pour un modèle de **191 KB** au format ONNX. Le système satisfait les contraintes d'embarquabilité et de temps réel fixées.

Les priorités pour une mise en production sont : l'élargissement du corpus (volume, diversité sujets, conditions réelles) et la validation exhaustive sur toutes les classes en dehors de la distribution d'entraînement.

</small>
