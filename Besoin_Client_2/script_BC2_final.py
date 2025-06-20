import pickle
import argparse
import pandas as pd

if __name__ == '__main__':
    # load
    with open('model_2.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scale_2.pkl', 'rb') as f:
        Scaler = pickle.load(f)

    parser = argparse.ArgumentParser(description="Mon programme avec options")
    parser.add_argument("--Length", type=float, nargs="?", help="Length du navire", required=True)
    parser.add_argument("--Width", type=float, nargs="?", help="Width du navire", required=True)
    parser.add_argument("--Draft", type=float, nargs="?", help="Draft du navire", required=True)
    args = parser.parse_args()

    new_data = {}
    new_data['Length'] = args.Length
    new_data['Width'] = args.Width
    new_data['Draft'] = args.Draft

    new_data_df = pd.DataFrame([new_data])

    new_data_df[['Length', 'Width', 'Draft']] = Scaler.transform(new_data_df[['Length', 'Width', 'Draft']])

    predicted_VesselType = model.predict(new_data_df)

    print("Parametres du bateau, Length :",args.Length, "Width :", args.Width, "Draft :", args.Draft)
    print(f"\nPredicted VesselType: {predicted_VesselType}")



#Exemple pour tester
#TP SPIRIT : --Length 182.5 --Width 32.23 --Draft 11.6
#-> doit renvoyer 80
#PAVO J : --Length 139.6 --Width 22.39 --Draft 6.8
#-> doit renvoyer 70
#BROWNSEA ENTERPRISE : --Length 10 --Width 6 --Draft 1.1
#-> doit renvoyer 60