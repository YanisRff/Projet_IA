# README - script\_BC2\_final.py

## Description

Ce script Python, `script_BC2_final.py`, permet de prédire le type de navire (« VesselType ») en fonction de ses caractéristiques physiques : **Length**, **Width** et **Draft**.

Il utilise un modèle de machine learning préalablement entraîné (stocké dans `model_2.pkl`) ainsi qu'un scaler (dans `scale_2.pkl`) pour normaliser les données d'entrée.

## Prérequis

* Python 3.10 ou plus récent
* Bibliothèques Python :

  * `pandas`
  * `pickle`
  * `argparse`
  * Le modèle et scaler enregistrés : `model_2.pkl` et `scale_2.pkl`

## Utilisation

Le script se lance en ligne de commande avec les paramètres suivants :

```bash
python script_BC2_final.py --Length <valeur> --Width <valeur> --Draft <valeur>
```

### Exemple :

```bash
python script_BC2_final.py --Length 182.5 --Width 32.23 --Draft 11.6
```

### Sortie correspondante :

```text
Parametres du bateau, Length : 182.5 Width : 32.23 Draft : 11.6

Predicted VesselType: ['80']
```

## Contenu du script

1. Chargement du modèle de classification et du scaler
2. Récupération des paramètres en ligne de commande
3. Mise en forme des données dans un `DataFrame`
4. Transformation des données via le scaler
5. Prédiction du type de navire
6. Affichage des paramètres et du résultat prédit

## Auteur

Alexandre Moreau
