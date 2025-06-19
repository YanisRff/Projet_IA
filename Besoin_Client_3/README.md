# Prédiction de Position de Navire

Ce projet permet de prédire la prochaine position (latitude et longitude) d'un navire à partir de ses caractéristiques actuelles et d'un modèle de machine learning préalablement entraîné.

## Fichiers

* `script_final.py` : script principal permettant de faire une prédiction de position.
* `model_3.pkl` : fichier pickle contenant le modèle de prédiction (prérequis pour le script).

## Prérequis

* Bibliothèques Python :

  * `pandas`
  * `argparse`
  * `pickle`

Installez les dépendances si nécessaire :

```bash
pip install pandas
```

## Utilisation

Lancez le script en ligne de commande avec les arguments requis :

```bash
python script_final.py \
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

### Signification des paramètres

* `--LAT` : Latitude actuelle du navire
* `--LON` : Longitude actuelle du navire
* `--SOG` : Vitesse sur le fond (Speed Over Ground)
* `--COG` : Cap sur le fond (Course Over Ground)
* `--Heading` : Cap compas (direction)
* `--VesselType` : Type de navire (normalisé par plage dans le script)
* `--Length` : Longueur du navire
* `--Width` : Largeur du navire
* `--Draft` : Tirant d'eau du navire
* `--Cargo` : Type de cargaison
* `--time` : Délai en secondes pour lequel on souhaite la prédiction de position

## Exemple de sortie

```bash
Predicted next position:
Predicted new LAT: 29.722282600000025
Predicted new LON: -95.23577529999997
```

## Notes

* Le script charge un modèle pickle (`model_3.pkl`) pour effectuer une prédiction sur les décalages de latitude et longitude.
* Le type de navire est regroupé dans les catégories 60, 70, ou 80 selon sa valeur initiale.
* La prédiction fournit directement les nouvelles coordonnées LAT et LON prédites.

## Auteur

*RUFFLÉ Yanis*

## Licence

Ce projet est sous licence *GNU*.

