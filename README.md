# Total Perspective Vortex — Plug your brain to the shell

## Présentation

**Total Perspective Vortex** est un projet de Brain-Computer Interface (BCI), ou interface cerveau-machine, basé sur l'électroencéphalographie (EEG) et le Machine Learning.

L'objectif est d'analyser l'activité électrique du cerveau afin de reconnaître une intention motrice : imaginer un mouvement de la main gauche ou de la main droite.

Le projet utilise principalement Python, MNE, NumPy, SciPy et Scikit-learn.

## Fonctionnement

Le traitement des signaux EEG suit plusieurs étapes :

1. **Acquisition et prétraitement** : chargement des signaux EEG avec MNE, filtrage des fréquences (8–30 Hz) et extraction des essais (*epochs*).
2. **Extraction de caractéristiques** : transformation des signaux en données exploitables par les algorithmes de classification.
3. **Réduction de dimensions** : diminution du nombre de caractéristiques pour simplifier le traitement.
4. **Classification** : apprentissage d'un modèle capable de distinguer les deux intentions motrices.
5. **Évaluation et prédiction** : validation des performances sur des données non utilisées pendant l'entraînement.

## Concepts et algorithmes

### CSP — Common Spatial Patterns

CSP est une méthode de réduction de dimensions spécialement adaptée à l'analyse EEG.

Elle cherche des combinaisons d'électrodes permettant de maximiser les différences de variance entre deux classes. Elle permet ainsi de conserver les informations spatiales les plus discriminantes pour la classification.

### Wavelets — Transformée en ondelettes

La transformée en ondelettes décompose un signal en plusieurs niveaux de fréquences et d'échelles temporelles.

Elle permet d'extraire des caractéristiques représentant différentes composantes du signal, notamment à travers les coefficients d'approximation et de détail.

### LDA — Linear Discriminant Analysis

LDA est un algorithme de classification qui cherche une séparation entre les différentes classes en exploitant leurs distributions statistiques.

Il est particulièrement adapté aux problèmes de classification avec peu de données et peut être utilisé après une réduction de dimensions par CSP.

### Régression logistique

La régression logistique est un algorithme de classification qui estime la probabilité qu'une observation appartienne à une classe.

Une fonction sigmoïde transforme le résultat du modèle en probabilité, permettant ensuite de prendre une décision de classification.

### Cross-validation

La validation croisée consiste à répéter l'entraînement et l'évaluation sur différents découpages des données afin d'obtenir une estimation plus fiable des performances du modèle.

Le projet utilise notamment une validation par groupe (*Leave-One-Group-Out*), permettant de tester le modèle sur un essai expérimental complet (*run*) qui n'a pas servi à son entraînement.

## Configuration

Le fichier `src/config.py` centralise les principaux paramètres :

- `DATASET_NAME` : choix du dataset (PhysioNet ou BCI Competition IV 2a).
- `CSP_TYPE` : choix de l'implémentation CSP.
- `FEATURE_TYPE` : choix de la méthode d'extraction des caractéristiques.
- `CLASSIFIER_TYPE` : choix du classifieur.
- `N_CSP_COMPONENTS` : nombre de composantes CSP.
- `WAVELET` et `WAVELET_LEVEL` : paramètres de la décomposition en ondelettes.
- `LOGISTIC_LEARNING_RATE` et `LOGISTIC_ITERATIONS` : paramètres de la régression logistique personnalisée.

## Fonctionnalités et bonus

- Prétraitement et filtrage des signaux EEG.
- Réduction de dimensions avec CSP.
- Extraction de caractéristiques par ondelettes.
- Classification avec des algorithmes standards et personnalisés.
- Implémentation personnalisée de la régression logistique.
- Implémentation personnalisée du calcul de covariance.
- Implémentation d'un solveur de valeurs propres.
- Évaluation multi-sujets et validation croisée.
- Prise en charge de plusieurs jeux de données.
- Simulation de lecture progressive des signaux EEG pour reproduire un flux de données.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Les jeux de données doivent être téléchargés séparément et ne sont pas inclus dans le dépôt.

## Exécution

Depuis la racine du projet :

```bash
python -m scripts.train
python -m scripts.evaluation
python -m scripts.predict
python -m scripts.realtime_simulation
```

---

*Project developed as part of the 42 curriculum.*