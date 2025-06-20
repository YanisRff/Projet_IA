import pickle
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

if __name__ == '__main__':
    # load
    with open('model_3.pkl', 'rb') as f:
        model_pipeline = pickle.load(f)
    
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
    parser.add_argument("--steps", type=int, nargs="?", help="Nombre d'itérations voulues", required=True)
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


    last_position = new_data_df.iloc[-1]
    current_lat = last_position['LAT']
    current_lon = last_position['LON']
    sog = last_position['SOG']
    cog = last_position['COG']
    heading = last_position['Heading']
    vessel_type = last_position['VesselType']
    length = last_position['Length']
    width = last_position['Width']
    draft = last_position['Draft']
    cargo = last_position['Cargo']
    estimated_delta_t = args.time

    if 60 <= vessel_type <= 69:
        vessel_type = 60
    elif 70 <= vessel_type <= 79:
        vessel_type = 70
    elif 80 <= vessel_type <= 89:
        vessel_type = 80

    n_steps = args.steps
    future_positions = []

    for _ in range(n_steps):
        input_data = pd.DataFrame({
            'LAT': [current_lat],
            'LON': [current_lon],
            'SOG': [sog],
            'COG': [cog],
            'Heading': [heading],
            'VesselType': [vessel_type],
            'Length': [length],
            'Width': [width],
            'Draft': [draft],
            'Cargo': [cargo],
            'delta_t': [estimated_delta_t]
        })

        predicted_delta = model_pipeline.predict(input_data)[0]
        delta_lat, delta_lon = predicted_delta

        current_lat = delta_lat
        current_lon = delta_lon
        future_positions.append((current_lat, current_lon))

    future_lats, future_lons = zip(*future_positions)

    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        lat=[last_position['LAT']],
        lon=[last_position['LON']],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=15,
            color='green',
            symbol='star'
        ),
        name='Dernière position connue'
    ))

    fig.add_trace(go.Scattermapbox(
        lat=future_lats,
        lon=future_lons,
        mode='lines+markers',
        marker=go.scattermapbox.Marker(
            size=8,
            color='red'
        ),
        name='Trajectoire future prédite'
    ))

    fig.update_layout(
        mapbox_style="open-street-map",
        hovermode='closest',
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=last_position['LAT'],
                lon=last_position['LON']
            ),
            pitch=0,
            zoom=10
        ),
        title=f'Trajectoire prédite du bateau',
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )
    fig.write_html("prediction_traj_bateau.html")
