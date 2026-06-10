## REEV - HAR: Reconnaissance d’Activité pour Orthèse de Genou Motorisée

### 1. Contexte

Ce projet vise à développer un système de reconnaissance d’activités humaines en temps réel pour une orthèse de genou motorisée.

**Objectif** : classifier 4 activités à partir de capteurs inertiels (IMU) de smartphone :

- 0 : Marche  
- 1 : Assis  
- 2 : Course  
- 3 : Chute  

**Contraintes :**
- exécution temps réel sur système embarqué
- modèle léger
- pipeline complet reproductible

### 2. Données

#### Acquisition
- Application : Phyphox
- Capteurs : accéléromètre et gyroscope (3 axes)
- Fréquence : 100 Hz
- Position : poche avant du pantalon

#### Volume des données

| Activité | Samples | Durée |
|----------|--------:|------:|
| Marche | 30 927 | ~5 min |
| Assis | 16 363 | ~2 min 44 |
| Course | 22 493 | ~3 min 45 |
| Chute | 32 433 | ~5 min 24 |
| Total | 102 216 | ~17 min |

#### Dataset (aprés préprocessing) 
Le dataset final est construit à partir des données IMU brutes (CSV) après nettoyage et segmentation.

- Format final : dataset.csv
- Colonnes : acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, label
- Fréquence : 100 Hz

Ce fichier représente les données synchronisées et nettoyées utilisées pour la suite du pipeline.

### 3. Pipeline de traitement

Le ***pipeline** complet est le suivant :

1. Chargement des fichiers CSV bruts
2. Nettoyage des données (NaN + bruit capteur)
3. Synchronisation accéléromètre / gyroscope
4. Fusion des signaux
5. Génération du dataset.csv
6. Fenêtrage glissant
7. Extraction des features
8. Split train / test
9. Entraînement du modèle
10. Export ONNX

### 3. Prétraitement

Étapes du pipeline :

1. Lecture des fichiers CSV
2. Nettoyage des valeurs manquantes
3. Synchronisation accéléromètre / gyroscope (interpolation 100 Hz)
4. Fusion des capteurs
5. Attribution des labels
6. Génération du dataset final

#### Filtrage

- Filtre Butterworth passe-bas (ordre 4, 20 Hz)
- Suppression du bruit haute fréquence
- Clipping des valeurs extrêmes à ±5σ (hors chute)

Les mouvements humains étant majoritairement inférieurs à 10 Hz, ce filtrage conserve l’information utile tout en réduisant le bruit capteur.
### 4.Architecture du modèle
Le modèle utilisé est un Random Forest Classifier.

### Schéma du pipeline modèle :

dataset.csv
     ↓
StandardScaler
     ↓
RandomForestClassifier (200 arbres)
     ↓
Prédiction (4 classes)
     ↓
Export ONNX (déploiement)

**Paramètres :**

- 200 arbres
- profondeur max = 15
- min_samples_leaf = 3
- class_weight = balanced

**Entrée / sortie :**

- Input : 65 features par fenêtre
- Output : 4 classes (walk, sit, run, fall)
### 4. Fenêtrage et extraction de caractéristiques

#### Fenêtrage

- Taille de fenêtre : 200 samples (2 secondes)
- Overlap : 50%
- Step : 1 seconde
- Nombre total de fenêtres : 1016

La taille de fenêtre de 2 secondes est choisie car elle permet de capturer un cycle complet des mouvements locomoteurs (marche, course) tout en limitant la latence du système pour une utilisation en temps réel. 
Un recouvrement de 50 % est utilisé afin d’augmenter le nombre d’exemples disponibles, d’assurer une continuité temporelle entre les fenêtres et de réduire le risque de manquer des événements courts comme une chute. Ce choix est conforme aux pratiques courantes en reconnaissance d’activités humaines.
#### Features extraites

Chaque fenêtre est transformée en 65 caractéristiques :
Par canal (6 canaux IMU) :
- moyenne
- écart-type
- minimum
- maximum
- RMS
- skewness
- kurtosis
- zero-crossing rate
- fréquence dominante (FFT)
- énergie spectrale

Features globales :
- magnitude accéléromètre (mean, std, max)
- magnitude gyroscope (mean, std)

Une fenêtre brute = 200 × 6 = 1200 valeurs → trop large pour Random Forest
65 features résumées = beaucoup plus compact et interprétable
Le Random Forest est très efficace sur ces features statistiques

### 5. Modèle

Modèle utilisé : Random Forest

Paramètres :
- **200 arbres**,  permet de stabiliser les prédictions sans augmenter excessivement le coût d’inférence
- **profondeur maximale : 15**,  limite la complexité des arbres afin de réduire le risque d’overfitting sur un dataset de taille réduite
- **min samples leaf : 3**, améliore la généralisation en évitant l’apprentissage de règles trop spécifiques aux données d’entraînement
- équilibrage des classes activé,**class_weight = balanced**,  permet de compenser le léger déséquilibre entre les classes et d’améliorer la robustesse globale du modèle.

### Résultats

#### Performance globale

| Ensemble | Accuracy |
|----------|----------:|
| Train | 0.999 |
| Validation | 0.987 |
| Test | 0.987 |

Le modèle ne présente pas de surapprentissage significatif (écart train/validation faible).

#### Rapport de classification (test set)

| Classe | Precision | Recall | F1-score | Support |
|--------|----------:|-------:|---------:|--------:|
| walk | 1.00 | 0.96 | 0.98 | 46 |
| sit  | 1.00 | 1.00 | 1.00 | 24 |
| run  | 1.00 | 1.00 | 1.00 | 34 |
| fall | 0.96 | 1.00 | 0.98 | 49 |

Accuracy globale : 0.99

#### Matrice de confusion (test set)

|        | walk | sit | run | fall |
|--------|-----:|----:|----:|-----:|
| walk   | 44 | 0 | 0 | 2 |
| sit    | 0 | 24 | 0 | 0 |
| run    | 0 | 0 | 34 | 0 |
| fall   | 0 | 0 | 0 | 49 |

#### Analyse des erreurs

Les erreurs observées concernent principalement des confusions entre marche et chute. Elles sont liées à des mouvements brusques ou des transitions (mise en poche / retrait du téléphone), générant des pics similaires à des impacts.


### 7. Validation sur données réelles

Sur une nouvelle session non utilisée lors de l’entraînement :

- 233 fenêtres analysées
- 99.1 % des prédictions correctes
- erreurs concentrées sur les transitions

La validation a été réalisée principalement sur des nouvelles données de marche. Pour une évaluation plus robuste, il serait nécessaire de tester le modèle sur l’ensemble des classes avec de nouvelles données, incluant différentes sessions et différentes positions du smartphone (poche avant, latérale, variations d’orientation). Cela permettrait de confirmer la généralisation du modèle dans des conditions proches d’un usage réel.


### 8. Déploiement et performance

Le pipeline complet est exporté au format ONNX, avec une taille de modèle de 191 KB, adaptée à un déploiement embarqué.

La latence a été mesurée sur 1000 inférences en environnement CPU standard :

| Métrique | Valeur |
|----------|--------:|
| Moyenne | 0.018 ms |
| P95 | 0.023 ms |
| P99 | 0.036 ms |

Le système produit une nouvelle fenêtre toutes les 1 seconde (fenêtrage glissant). Les performances d’inférence sont donc largement compatibles avec une exécution en temps réel.

Ces mesures ont été réalisées dans des conditions de test sur application mobile, et pourraient légèrement varier selon le matériel cible (smartphone ou dispositif embarqué).

### 9.Limites

- Dataset limité en taille (~17 min de données)
- Données de chute simulées
- Sensibilité aux variations de position du smartphone
- Validation encore limitée à certaines conditions

Des améliorations incluent la collecte multi-utilisateurs et des tests en conditions réelles apporteront une meilleur certitude sur le fonctionnement de modele

### 9. Conclusion

Le système développé permet la reconnaissance de quatre activités humaines à partir de signaux inertiels.

Le pipeline complet comprend :
- acquisition des données
- synchronisation des capteurs
- prétraitement et filtrage
- segmentation en fenêtres
- extraction de caractéristiques
- classification
- export et déploiement ONNX

Le modèle obtenu est léger, précis et adapté à une utilisation en temps réel sur dispositif embarqué.