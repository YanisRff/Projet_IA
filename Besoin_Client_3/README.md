# Prédiction de Position de Navire

Ce projet permet de prédire la prochaine position (latitude et longitude) d'un navire à partir de ses caractéristiques actuelles et d'un modèle de machine learning préalablement entraîné.

## Fichiers

* `script_BC3_final.py` : script principal permettant de faire une prédiction de position unique.
* `script_BC3_final_bonus.py` : version avancée du script qui prédit une trajectoire future sur plusieurs étapes et génère une carte interactive.
* `model_3.pkl` : fichier pickle contenant le modèle de prédiction (prérequis pour les scripts).

## Prérequis

* Bibliothèques Python :

  * `pandas`
  * `argparse`
  * `pickle`
  * `plotly`
  * `matplotlib`

Installation des dépendances si nécessaire :

```bash
pip install pandas plotly matplotlib
```

## Utilisation

### Script simple (`script_BC3_final.py`)

Ce script prédit une unique position future du navire à partir de ses données actuelles :

```bash
python script_BC3_final.py \
  --LAT 29.72289 \
  --LON -95.23584 \
  --SOG 0.0 \
  --COG 0.0 \
  --Heading 0 \
  --VesselType 80 \
  --Length 183 \
  --Width 28.0 \
  --Draft 10.0 \
  --Cargo 83 \
  --time 183
```

### Script bonus (`script_BC3_final_bonus.py`)

Ce script prédit une trajectoire complète sur plusieurs itérations, en plus de générer une visualisation de la trajectoire sur une carte interactive (HTML).

```bash
python script_BC3_final_bonus.py \
  --LAT 28.60688 \
  --LON -94.12511 \
  --SOG 13.0 \
  --COG 245.6 \
  --Heading 245 \
  --VesselType 80 \
  --Length 243 \
  --Width 42.0 \
  --Draft 14.6 \
  --Cargo 80 \
  --time 600 \
  --steps 20000
```

### Signification des paramètres

* `--LAT` : Latitude actuelle du navire
* `--LON` : Longitude actuelle du navire
* `--SOG` : Vitesse sur le fond (Speed Over Ground)
* `--COG` : Cap sur le fond (Course Over Ground)
* `--Heading` : Cap compas (direction)
* `--VesselType` : Type de navire (regroupé en 60, 70, 80)
* `--Length` : Longueur du navire
* `--Width` : Largeur du navire
* `--Draft` : Tirant d'eau du navire
* `--Cargo` : Type de cargaison
* `--time` : Délai (en secondes) entre les prédictions
* `--steps` : Nombre d'itérations (points de trajectoire futurs à prédire) \[uniquement pour le script bonus]

## Exemple de sortie

### Pour `script_BC3_final.py`

```bash
Predicted next position:
Predicted new LAT: 29.722282600000025
Predicted new LON: -95.23577529999997
```

### Pour `script_BC3_final_bonus.py`

Un fichier `prediction_traj_bateau.html` est généré. Il contient une carte interactive affichant la dernière position connue et la trajectoire future prédite du navire.

## Notes

* Le modèle chargé via `model_3.pkl` prédit un décalage de latitude et longitude.
* Le type de navire est regroupé dans les catégories 60, 70 ou 80 selon sa valeur initiale.
* La visualisation de la trajectoire est réalisée avec Plotly.

## Auteur

*RUFFLÉ Yanis*

## Licence

Ce projet est sous licence *GNU*.

