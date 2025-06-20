import pickle
import argparse
import pandas as pd

if __name__ == '__main__':
    # Chargement du modèle et du scaler
    with open('model_1.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scale_1.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Lecture des paramètres d'entrée
    parser = argparse.ArgumentParser(description="Prédire le cluster d’un navire")
    parser.add_argument("--LAT", type=float, required=True)
    parser.add_argument("--LON", type=float, required=True)
    parser.add_argument("--SOG", type=float, required=True)
    parser.add_argument("--COG", type=float, required=True)
    parser.add_argument("--Heading", type=float, required=True)
    args = parser.parse_args()

    # Création d’un DataFrame avec les données
    new_data = {
        'LAT': args.LAT,
        'LON': args.LON,
        'SOG': args.SOG,
        'COG': args.COG,
        'Heading': args.Heading
    }
    new_data_df = pd.DataFrame([new_data])

    # Application du scaler
    X_scaled = scaler.transform(new_data_df)

    # Prédiction
    predicted_cluster = model.predict(X_scaled)
    print(new_data_df)
    print(f"\n>>> Cluster prédit pour ce navire : {predicted_cluster[0]}")
