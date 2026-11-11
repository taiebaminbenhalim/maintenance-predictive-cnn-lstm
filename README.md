# Maintenance prédictive de disques durs avec CNN-LSTM

Prédiction de la panne de disques durs **Seagate ST4000DM000** à partir des données SMART publiques de [Backblaze](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data), avec un modèle hybride **CNN-LSTM** (Keras / TensorFlow).

## Objectif

Classer une fenêtre de 5 jours de mesures SMART d'un disque comme :

- `0` : disque sain
- `1` : disque qui va tomber en panne dans les 2 jours suivant la fenêtre

## Données

| Jeu | Source | Période |
|---|---|---|
| Entraînement | Backblaze Hard Drive Data | T1 à T4 2017 |
| Test (téléchargé) | Backblaze Hard Drive Data | T4 2016 |

Les données ne sont pas incluses dans ce dépôt (trop volumineuses). Elles sont téléchargées automatiquement par les notebooks `01` à `03` depuis :
`https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q{1..4}_{année}.zip`

## Pipeline

1. **Téléchargement** (`01`, `02`, `03`) : récupération des archives trimestrielles et concaténation des CSV.
2. **Nettoyage** (`04`) :
   - filtrage sur le modèle `ST4000DM000`
   - suppression des colonnes vides, constantes et `normalized`
   - remplacement des valeurs manquantes par 0
   - conversion des dates, tri par disque et par date, réduction mémoire (float32 / int32)
3. **Modélisation** (`05`) :
   - sélection de tous les disques défaillants et de 50 disques sains
   - correction temporelle : réindexation journalière et `ffill` pour combler les jours manquants
   - construction des séquences : fenêtre de **5 jours**, label positif pour la fenêtre se terminant **2 jours avant** la panne ; fenêtre glissante pour les disques sains
   - split train / validation / test (64 % / 16 % / 20 %, stratifié), `StandardScaler` ajusté sur le train
   - entraînement du CNN-LSTM

## Architecture du modèle

```
Conv1D(64, kernel=2, relu)
MaxPooling1D(2)
LSTM(64)
Dropout(0.3)
Dense(32, relu)
Dense(1, sigmoid)
```

- Entrée : `(5, 25)` (5 jours × 25 variables SMART)
- Perte : `binary_crossentropy`, optimiseur Adam
- `EarlyStopping` (patience 5) et `ModelCheckpoint` sur `val_loss`, 20 époques max, batch 32

## Résultats (jeu de test)

| Métrique | Valeur |
|---|---|
| Accuracy | 98,7 % |
| Précision (classe panne) | 0,99 |
| Rappel (classe panne) | 0,67 |
| F1-score (classe panne) | 0,80 |

Matrice de confusion : `TN = 3613`, `FP = 1`, `FN = 49`, `TP = 98`.

Les classes sont très déséquilibrées (environ 4 % de séquences de panne) : l'accuracy est donc peu informative, le rappel et le F1 de la classe panne sont les métriques à suivre.

## Limites et pistes d'amélioration

- Le mélange et le split sont faits **ligne par ligne** avant de regrouper par blocs de 5 : à remplacer par un split **par séquence ou par disque** pour préserver l'ordre temporel et éviter toute fuite entre train et test.
- Le jeu de test T4 2016 est téléchargé mais pas encore utilisé pour l'évaluation finale.
- Améliorer le rappel (pondération des classes, seuil de décision, sur-échantillonnage).
- Tester d'autres tailles de fenêtre et d'autres horizons de prédiction.

## Structure du dépôt

```
.
├── notebooks/
│   ├── 01_telechargement_train_Q1-Q3_2017.ipynb
│   ├── 02_telechargement_train_Q4_2017.ipynb
│   ├── 03_telechargement_test_Q4_2016.ipynb
│   ├── 04_nettoyage_donnees.ipynb
│   └── 05_modelisation_cnn_lstm.ipynb
├── requirements.txt
└── README.md
```

## Utilisation

Les notebooks ont été exécutés sur **Google Colab** et lisent/écrivent sur Google Drive (dossier `Colab Notebooks/modele stocastique/`). Ordre d'exécution : `01` → `02` → `03` → `04` → `05`.

Pour une exécution en local :

```bash
git clone https://github.com/<ton-utilisateur>/<nom-du-depot>.git
cd <nom-du-depot>
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Il faut alors remplacer les cellules `drive.mount(...)` et les chemins `/content/drive/...` par des chemins locaux.

## Technologies

Python · Pandas · NumPy · scikit-learn · TensorFlow / Keras · Matplotlib · Seaborn

## Auteur

Tayeb
