import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sklearn
import pickle

from numpy import sqrt
from numpy import mean

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score as cv
from sklearn.model_selection import cross_val_predict as cvp
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv('../data_clean.csv')


#TRI SELON MMSI
print('\n avant tri \n',data['MMSI'].head(5))
data = data.sort_values('MMSI')
print('\n apres tri \n',data['MMSI'].head(5))


#Suppression des colonnes non désirables et types des variables
data = data.drop(columns=['id','BaseDateTime','LAT','LON','COG','Heading','VesselName','IMO','CallSign','TransceiverClass','Status','Cargo','SOG'])
print('\ncolonnes : ', data.columns)


#Classification des VesselTypes en 3 groupes ("Passager", "Cargo", "Tanker")
data['VesselType'] = data['VesselType'].apply(lambda VT_value: 60 if 60 <= VT_value <= 69 else VT_value)
data['VesselType'] = data['VesselType'].apply(lambda VT_value: 70 if 70 <= VT_value <= 79 else VT_value)
data['VesselType'] = data['VesselType'].apply(lambda VT_value: 80 if 80 <= VT_value <= 89 else VT_value)


#Mise en forme des Donnees
print('\n avant changement de types \n',data.head(5))
data = data.astype({'MMSI' : str,'VesselType' : str, 'Length' : float, 'Width' : float, 'Draft' : float})
print('\n apres changement de types \n',data.head(5))

# Tri des doublons pour éviter l'Overfitting
print("Il y a",data['MMSI'].nunique(),"bateaux dans la base de donnees")
data.drop_duplicates(subset='MMSI', inplace=True)


#Base train et test et deplacement bateau de la fin base de train
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

print('\nRépartition des donnees entre les deux bases')
print('train data : ',len(train_data), 'soit : ', (len(train_data)/len(data))*100,'%')
print('train test : ',len(test_data), 'soit : ', (len(test_data)/len(data))*100,'%\n')

print('avant reorganisation des bases')
print(train_data['MMSI'].tail(1))
print(test_data['MMSI'].head(1))
print('\napres reorganisation des bases')

mmsi_target = test_data['MMSI'].iloc[0]
line_to_move = train_data[train_data['MMSI'] == mmsi_target]
test_data = pd.concat([test_data, line_to_move], ignore_index=True)
train_data = train_data[train_data['MMSI'] != mmsi_target]

print(train_data['MMSI'].tail(1))
print(test_data['MMSI'].head(1))

print('\nRépartition des donnees entre les deux bases')
print('train data : ',len(train_data), 'soit : ', (len(train_data)/len(data))*100,'%')
print('train test : ',len(test_data), 'soit : ', (len(test_data)/len(data))*100,'%')


#Scaler sur les donnees quantitatives
print("\navant preprocessing :")
print("length min:",train_data['Length'].min(),"length max:", train_data['Length'].max())
print("width min:",train_data['Width'].min(),"width max:", train_data['Width'].max())
print("Draft min:",train_data['Draft'].min(),"Draft max:", train_data['Draft'].max())
print('\n',train_data.head(5))

scaler = preprocessing.StandardScaler()
train_data[['Length','Width','Draft']] = scaler.fit_transform(train_data[['Length','Width','Draft']])

print("\napres preprocessing :")
print("length min:",train_data['Length'].min(),"length max:", train_data['Length'].max())
print("width min:",train_data['Width'].min(),"width max:", train_data['Width'].max())
print("Draft min:",train_data['Draft'].min(),"Draft max:", train_data['Draft'].max())
print('\n',train_data.head(5),'\n\n')


#Separation X et Y
X_train = train_data.drop(columns=['MMSI','VesselType'])
Y_train = train_data['VesselType']

X_test = test_data.drop(columns=['MMSI','VesselType'])
Y_test = test_data['VesselType']


#Test SGDC
SGDClassifier = SGDClassifier()
model = SGDClassifier.fit(X_train, Y_train)

pred_SGD = cvp(model, X_train, Y_train, cv=3)

accuracy_SGD = cv(SGDClassifier, X_train, Y_train, cv=3, scoring="accuracy")
mat_SGD = confusion_matrix(Y_train, pred_SGD)

print("accuracy_SGD=", format(accuracy_SGD.mean(), '.2f'))

plt.matshow(mat_SGD, cmap=mpl.cm.Reds)
for (i,j), val in np.ndenumerate(mat_SGD):
  plt.text(j, i, f'{val:.1f}', ha='center', va='center', color='black', fontsize=14)
  plt.xlabel('Prédictions')
  plt.ylabel('Vraies valeurs')
  plt.title('Matrice de Confusion : Méthode SGD')


#Test Random Forest Classifier
RFC = RandomForestClassifier()
model = RFC.fit(X_train, Y_train)

pred_RFC = cvp(model, X_train, Y_train, cv=3)

accuracy_RFC = cv(RFC, X_train, Y_train, cv=3, scoring="accuracy")
mat_RFC = confusion_matrix(Y_train, pred_RFC)

print("accuracy_RFC=", format(accuracy_RFC.mean(), '.2f'))

plt.matshow(mat_RFC, cmap=mpl.cm.Reds)
for (i,j), val in np.ndenumerate(mat_RFC):
  plt.text(j, i, f'{val:.1f}', ha='center', va='center', color='black', fontsize=14)
  plt.xlabel('Prédictions')
  plt.ylabel('Vraies valeurs')
  plt.title('Matrice de Confusion : Méthode RFC')


#Test Logistic Regression
LR = LogisticRegression()
model = LR.fit(X_train, Y_train)

pred_LR = cvp(model, X_train, Y_train, cv=3)

accuracy_LR = cv(LR, X_train, Y_train, cv=3, scoring="accuracy")
mat_LR = confusion_matrix(Y_train, pred_LR)

print("accuracy_LR=", format(accuracy_LR.mean(), '.2f'))

plt.matshow(mat_LR, cmap=mpl.cm.Reds)
for (i,j), val in np.ndenumerate(mat_LR):
  plt.text(j, i, f'{val:.1f}', ha='center', va='center', color='black', fontsize=14)
  plt.xlabel('Prédictions')
  plt.ylabel('Vraies valeurs')
  plt.title('Matrice de Confusion : Méthode LR')


#GridSearch
param_grid = [
    {'n_estimators': [50, 100, 500], 'max_features': ['sqrt'], 'max_depth':[5,10,20]},
]

grid = GridSearchCV(RFC, param_grid=param_grid, cv=3, scoring = "accuracy")
grid.fit(X_train, Y_train)
best_params = grid.best_params_
print(grid.best_params_)
pd.DataFrame(grid.cv_results_)


#Final Model
final_model = grid.best_estimator_
model = final_model.fit(X_train, Y_train)

pred_final_model = cvp(model, X_train, Y_train, cv=3)
accuracy_final_model = cv(final_model, X_train, Y_train, cv=3, scoring="accuracy")

print("accuracy_final_model=", format(accuracy_final_model.mean(), '.2f'))

#Exportation
with open('scale_2.pkl', 'wb') as f:
  pickle.dump(scaler, f)

with open('model_2.pkl', 'wb') as f:
  pickle.dump(final_model, f)