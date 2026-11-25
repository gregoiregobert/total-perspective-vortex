# Total Perspective Vortex
## Plug your brain to the shell

**Brain-Computer Interface using Machine Learning and EEG signals**

---

## 1. Présentation du projet

Total Perspective Vortex est un projet de Brain-Computer Interface (BCI), c'est-à-dire une interface permettant d'interpréter l'activité cérébrale à partir de signaux électroencéphalographiques (EEG).

L'objectif est de développer un modèle de Machine Learning capable de distinguer deux intentions motrices :

- Main gauche
- Main droite

Le système utilise des signaux EEG pour extraire des caractéristiques du cerveau, puis les transmet à un algorithme de classification afin de prédire l'intention correspondante.

Le projet permet de tester plusieurs méthodes d'extraction de caractéristiques et plusieurs algorithmes de classification.

### Technologies utilisées

- Python
- NumPy
- MNE
- Scikit-learn
- PyWavelets
- Matplotlib
- MOABB

---

## 2. Installation

### Prérequis

- Python 3
- Git
- Un terminal

### Créer l'environnement virtuel

Un environnement virtuel permet d'isoler les dépendances du projet des autres installations Python présentes sur la machine.

```bash
python3 -m venv .venv
```

### Activer l'environnement

Sur macOS et Linux :

```bash
source .venv/bin/activate
```

### Installer les dépendances

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Désactiver l'environnement

```bash
deactivate
```

---

## 3. Configuration du projet

Les principaux paramètres de l'expérience sont regroupés dans `src/config.py`.

### Choix du dataset

Deux datasets sont disponibles :

**PhysioNet EEG Motor Movement/Imagery**

Ce dataset contient des enregistrements EEG de plusieurs sujets, avec des mouvements réels et imaginés.

Les runs utilisés sont :

- 3, 7 et 11 : mouvements réels.
- 4, 8 et 12 : mouvements imaginés.

Le projet mélange les mouvements réels et imaginés tout en conservant deux classes : main gauche et main droite.

**BCI Competition IV - Dataset 2a**

Ce dataset contient des enregistrements EEG de neuf sujets, avec notamment des tâches d'imagination motrice.

Le projet utilise les deux classes main gauche et main droite.

Le chargement et le prétraitement sont adaptés à chaque dataset, mais les deux utilisent ensuite la même interface de traitement : les epochs, leurs labels et leurs groupes de runs.

---

## 4. Architecture générale

Le projet est organisé pour séparer le chargement des données, leur préparation, l'extraction des caractéristiques, l'entraînement et l'évaluation.

Les principaux fichiers sont :

| Fichier | Responsabilité |
|---|---|
| `src/config.py` | Configuration générale |
| `src/preprocessing.py` | Chargement et prétraitement des données |
| `src/datasets/physionet.py` | Chargement du dataset PhysioNet |
| `src/datasets/bci_competition.py` | Chargement du dataset BCI |
| `src/pipeline.py` | Construction du pipeline Machine Learning |
| `src/model.py` | Fonctions communes d'optimisation, d'entraînement et de test |
| `evaluation.py` | Évaluation sur plusieurs sujets |
| `realtime_simulation.py` | Simulation de prédictions en temps réel |

Les modules d'extraction de caractéristiques et de classification sont utilisés par le pipeline.

### Fonctionnement général

```text
             DATASET
                |
                v
        Chargement des données
                |
                v
          Prétraitement
                |
                v
       Découpage en epochs
                |
                v
      Séparation train / val / test
                |
                v
     Extraction des caractéristiques
          /           |           \
         CSP       Wavelets      Combined
          \           |           /
                |
                v
          Classification
          /             \
         LDA       Régression logistique
                |
                v
             Prédiction
                |
                v
          Évaluation finale
```

---

## 5. Prétraitement des données

Le prétraitement prépare les signaux EEG avant leur utilisation par les modèles.

### Filtrage

Les signaux sont filtrés entre 8 et 30 Hz.

Cette plage de fréquences permet notamment de conserver les bandes :

- Mu : environ 8 à 13 Hz.
- Beta : environ 13 à 30 Hz.

Ces bandes sont particulièrement intéressantes pour l'étude de l'activité cérébrale liée aux mouvements.

### Découpage en epochs

Une epoch est un segment de signal EEG correspondant à un événement précis.

Pour PhysioNet, les epochs sont extraites entre 1 et 4 secondes après l'événement.

Chaque epoch représente donc trois secondes de signal.

Avec une fréquence d'échantillonnage de 160 Hz, une epoch contient environ 481 échantillons par canal.

Les labels sont :

- `1` : main gauche.
- `2` : main droite.

Pour BCI Competition, les epochs sont préparées par MOABB avec le paradigme `LeftRightImagery`, en conservant les données de la session d'entraînement.

---

## 6. Séparation Train / Validation / Test

La séparation des données est un élément essentiel du projet.

Elle est réalisée dans :

```text
src/model.py
```

Plus précisément dans la fonction :

```python
load_subject_data()
```

Le projet sépare les données en utilisant les **runs**, et non en mélangeant aléatoirement les epochs.

Un run correspond à une session d'enregistrement contenant plusieurs essais EEG.

Cette séparation permet de tester le modèle sur un run qu'il n'a jamais utilisé pendant son apprentissage ou son optimisation.

### 6.1 PhysioNet

Avec la configuration :

```python
RUNS = (3, 4, 7, 8, 11, 12)
```

La séparation est la suivante :

| Utilisation | Runs |
|---|---|
| Données de développement | 3, 4, 7, 8, 11 |
| Test final | 12 |

Les cinq premiers runs sont utilisés pour l'entraînement et la validation croisée.

Le dernier run est conservé pour le test final.

### 6.2 BCI Competition

Le chargement MOABB fournit les données de la session d'entraînement, accompagnées d'un identifiant de run pour chaque epoch.

Les groupes sont triés, puis séparés de la même manière :

| Utilisation | Groupes |
|---|---|
| Données de développement | Les cinq premiers groupes |
| Test final | Le dernier groupe |

Les identifiants de groupes sont générés à partir des runs disponibles dans les métadonnées.

### 6.3 Validation croisée : Leave-One-Group-Out

La validation croisée est réalisée avec :

```python
LeaveOneGroupOut()
```

Cette méthode met de côté un groupe complet à chaque étape.

Un groupe correspond ici à un run entier.

Pour PhysioNet, avec cinq runs de développement, cela produit cinq évaluations :

| Fold | Entraînement | Validation |
|---|---|---|
| 1 | Runs 4, 7, 8, 11 | Run 3 |
| 2 | Runs 3, 7, 8, 11 | Run 4 |
| 3 | Runs 3, 4, 8, 11 | Run 7 |
| 4 | Runs 3, 4, 7, 11 | Run 8 |
| 5 | Runs 3, 4, 7, 8 | Run 11 |

Un **fold** est simplement une itération de la validation croisée.

Cette méthode permet de vérifier que le modèle fonctionne sur un enregistrement différent de ceux utilisés pour son entraînement.

Elle évite également de répartir aléatoirement des epochs provenant d'un même run entre l'entraînement et la validation.

### 6.4 Optimisation des paramètres

Pendant la validation croisée, plusieurs configurations du pipeline sont testées.

Par exemple :

- Nombre de composantes CSP.
- Type d'ondelette.
- Niveau de décomposition.
- Paramètres du classificateur.

Chaque configuration est évaluée sur les cinq folds.

La configuration obtenant la meilleure précision moyenne est retenue.

### 6.5 Entraînement final

Une fois les meilleurs paramètres sélectionnés :

1. Le pipeline est reconstruit avec ces paramètres.
2. Il est entraîné sur l'ensemble des cinq runs de développement.
3. Il est ensuite évalué une seule fois sur le run de test.

```text
        RUNS DE DÉVELOPPEMENT
         3    4    7    8    11
         |    |    |    |    |
         +----+----+----+----+
                  |
                  v
          VALIDATION CROISÉE
             5 FOLDS
                  |
                  v
         Choix des paramètres
                  |
                  v
        Entraînement final
         sur les 5 runs
                  |
                  v
           TEST FINAL
              Run 12
```

Le run de test n'intervient ni dans la validation croisée, ni dans le choix des paramètres.

**Attention :** l'évaluation globale sélectionne ensuite les meilleurs sujets en fonction de leur précision sur le test final. La moyenne des sujets sélectionnés est donc une mesure de sélection des meilleurs résultats, et non une estimation totalement indépendante des performances sur de nouveaux sujets.

---

## 7. Extraction des caractéristiques

Les signaux EEG bruts contiennent beaucoup de valeurs numériques.

Avant de les transmettre au classificateur, le projet extrait des caractéristiques qui résument certaines propriétés utiles des signaux.

Trois configurations sont disponibles :

- CSP
- Wavelets
- Combined

### 7.1 CSP - Common Spatial Patterns

CSP signifie Common Spatial Patterns.

C'est une méthode qui cherche des combinaisons de canaux EEG permettant de distinguer deux classes.

Par exemple, certains canaux peuvent présenter une activité différente selon que le sujet imagine un mouvement de la main gauche ou de la main droite.

CSP apprend des filtres spatiaux qui accentuent ces différences.

Le fonctionnement simplifié :

1. Calculer les caractéristiques statistiques des deux classes.
2. Chercher les combinaisons de canaux qui différencient le mieux les classes.
3. Projeter les signaux sur les composantes obtenues.
4. Calculer la variance de chaque composante.
5. Utiliser le logarithme de cette variance comme caractéristique.

Avec 64 canaux EEG et quatre composantes CSP, le modèle passe d'un signal multicanal contenant de nombreuses valeurs à quatre caractéristiques par epoch.

Cela réduit considérablement le nombre de dimensions à traiter.

Le projet propose une implémentation personnalisée et une implémentation basée sur MNE.

### 7.2 Wavelets - Transformée en ondelettes

La transformée en ondelettes permet d'étudier les variations d'un signal à différentes échelles de temps et de fréquence.

Contrairement à une simple mesure globale, elle permet de décomposer le signal en plusieurs composantes.

Le projet utilise la transformée en ondelettes discrète :

```python
pywt.wavedec(
    channel_signal,
    wavelet="db4",
    level=4
)
```

Avec une décomposition de niveau 4, on obtient cinq groupes de coefficients :

- A4 : approximation de basse fréquence.
- D4 : détails de niveau 4.
- D3 : détails de niveau 3.
- D2 : détails de niveau 2.
- D1 : détails de niveau 1.

Pour chaque groupe, le projet calcule une énergie logarithmique :

```python
np.log(np.sum(coefficients ** 2) + 1e-10)
```

Cette valeur résume l'énergie contenue dans la composante.

On obtient donc cinq caractéristiques par canal.

Exemple avec PhysioNet :

```text
64 canaux × 5 caractéristiques = 320 caractéristiques
```

Pour BCI Competition, qui utilise 22 canaux :

```text
22 canaux × 5 caractéristiques = 110 caractéristiques
```

Ces caractéristiques permettent au classificateur de rechercher des différences entre les signaux associés aux deux intentions motrices.

### 7.3 Combined

Le mode Combined combine les caractéristiques issues de CSP et des ondelettes.

L'objectif est de tirer parti de deux informations complémentaires :

- CSP : les différences spatiales entre les canaux EEG.
- Wavelets : les différences temporelles et fréquentielles du signal.

Le classificateur reçoit alors les caractéristiques combinées.

---

## 8. Classification

Après l'extraction des caractéristiques, le pipeline transmet les données à un classificateur.

Le projet propose trois possibilités.

### 8.1 LDA - Linear Discriminant Analysis

LDA signifie Linear Discriminant Analysis.

C'est un algorithme de classification qui cherche à séparer les classes à l'aide d'une frontière linéaire.

Il apprend notamment à partir des moyennes et de la dispersion des caractéristiques de chaque classe.

Dans notre cas, il cherche à distinguer les caractéristiques associées à :

- La main gauche.
- La main droite.

LDA est relativement simple et rapide, ce qui en fait un bon modèle de référence pour la classification EEG.

### 8.2 Régression logistique

La régression logistique est un algorithme qui calcule une probabilité d'appartenance à chaque classe.

Dans notre cas, elle estime la probabilité que les caractéristiques correspondent à une imagination motrice gauche ou droite.

Elle utilise une combinaison pondérée des caractéristiques :

```text
Caractéristiques EEG
        |
        v
Combinaison pondérée
        |
        v
Fonction sigmoïde
        |
        v
Probabilité
        |
        v