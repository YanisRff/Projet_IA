import pickle
import argparse
import pandas as pd

if __name__ == '__main__':
    # load
    with open('model_3.pkl', 'rb') as f:
        model = pickle.load(f)
    
    parser = argparse.ArgumentParser(description="Mon programme avec options")
    parser.add_argument("--LAT", type=float, nargs="?", help="Lattitude du navire", required=True)
    parser.add_argument("--LON", type=float, nargs="?", help="Longitude du navire", required=True)
    parser.add_argument("--SOG", type=float, nargs="?", help="Vitesse du navire", required=True)
    parser.add_argument("--COG", type=float, nargs="?", help="Cap du navire", required=True)
    parser.add_argument("--Heading", type=int, nargs="?", help="Direction du navire", required=True)
    parser.add_argument("--VesselType", type=int, nargs="?", help="Type du navire", required=True)
    parser.add_argument("--Length", type=int, nargs="?", help="Longueur du navire", required=True)
    parser.add_argument("--Width", type=float, nargs="?", help="Largeur du navire", required=True)
    parser.add_argument("--Draft", type=float, nargs="?", help="Tirant d'eau du navire", required=True)
    parser.add_argument("--Cargo", type=float, nargs="?", help="Type de cargaison du navire", required=True)
    parser.add_argument("--time", type=float, nargs="?", help="Delta temps voulu pour la prédiction", required=True)
    args = parser.parse_args()

    new_data = {}
    new_data['LAT'] = args.LAT
    new_data['LON'] = args.LON
    new_data['SOG'] = args.SOG
    new_data['COG'] = args.COG
    new_data['Heading'] = args.Heading
    new_data['VesselType'] = args.VesselType
    new_data['Length'] = args.Length
    new_data['Width'] = args.Width
    new_data['Draft'] = args.Draft
    new_data['Cargo'] = args.Cargo
    new_data['delta_t'] = args.time

    new_data_df = pd.DataFrame([new_data])
    
    new_data_df['VesselType'] = new_data_df['VesselType'].apply(lambda x: 60 if 60 <= x <= 69 else x)
    new_data_df['VesselType'] = new_data_df['VesselType'].apply(lambda x: 70 if 70 <= x <= 79 else x)
    new_data_df['VesselType'] = new_data_df['VesselType'].apply(lambda x: 80 if 80 <= x <= 89 else x)

    predicted_delta = model.predict(new_data_df)

    predicted_delta_lat = predicted_delta[0, 0]
    predicted_delta_lon = predicted_delta[0, 1]

    print("\nPredicted next position:")
    print(f"Predicted new LAT: {predicted_delta_lat}")
    print(f"Predicted new LON: {predicted_delta_lon}")

