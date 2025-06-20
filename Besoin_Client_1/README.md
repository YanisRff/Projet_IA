
#  Prédiction du cluster d’un navire

Ce script Python permet de prédire à quel **cluster** appartient un navire, en se basant sur ses données de navigation : latitude, longitude, vitesse, cap et direction.

##  Modèle utilisé
Le script charge un modèle de machine learning (`model_1.pkl`) et un scaler (`scale_1.pkl`) pour effectuer la prédiction.

---

##  Prérequis

- Python 3.x
- Bibliothèques :
  - `pandas`
  - `pickle`
  - `argparse`

Installe les bibliothèques nécessaires avec :
```bash
pip install pandas
```

---

##  Fichiers requis

Assurez-vous que les fichiers suivants se trouvent dans le même dossier que le script :
- `model_1.pkl` – modèle entraîné
- `scale_1.pkl` – scaler (normalisation des données)

---

##  Utilisation

Exécute le script en ligne de commande avec les arguments suivants :

```bash
python script_BC1_final.py --LAT 48.8566 --LON 2.3522 --SOG 12.5 --COG 85.0 --Heading 90.0
```

- `--LAT` : Latitude du navire
- `--LON` : Longitude du navire
- `--SOG` : Speed Over Ground (vitesse réelle)
- `--COG` : Course Over Ground (cap réel)
- `--Heading` : Direction du navire

---

##  Exemple de sortie

```
Cluster prédit : 2
```

---
