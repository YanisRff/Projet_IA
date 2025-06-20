import pandas as pd
import pickle
import  matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# chargement des données
df = pd.read_csv("../data_clean.csv", delimiter=';')
colonnes_utiles = ['LAT', 'LON', 'SOG', 'COG','Heading'] # selection des colonnes pertinentes pour clustering
df = df[colonnes_utiles].dropna() #supprime les lignes ou mes col pertinentes ont NA
df = df.reset_index(drop=True)

#normalise les données car elles ne son pas a la meme echelle
# LON LAT -180 et +180 heading 0 à 360 SOG 0 à 30+
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


# utilisation de silhouette pour determiner le nbre de cluster



#silhouette_scores = []
#inertias = []
#range_n_clusters = range(2,6)
#for k in range_n_clusters:
   # kmeans = KMeans(n_clusters = k, random_state=42)
    #cluster_labels = kmeans.fit_predict(X_scaled)
    #score = silhouette_score(X_scaled, cluster_labels)
    #inertias.append(kmeans.inertia_)
    #silhouette_scores.append((k,score))
    #print(f"nombre de cluster: {k} - silhouette score : {score:.4f}")
# Trouver le meilleur k
#best_k, best_score = max(silhouette_scores, key=lambda x: x[1])
#print(f"\n>> Meilleur nombre de clusters :", {best_k})
#print(f">> Score de silhouette correspondant : {best_score:.4f}")


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#Reduction des 5 variabes en 2 composantes principales
pca= PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

#clustering(KMeans)
kmeans = KMeans(n_clusters=3, random_state=42) #best_k
labels_kmeans = kmeans.fit_predict(X_scaled)
plt.figure(figsize=(10,6)) #visuialisation
plt.scatter(X_pca[:, 0], X_pca[:, 1],c=labels_kmeans, cmap='tab10', s=10)
plt.title("Clusters KMeans visualisés sur 2 composantes PCA")
plt.xlabel("composante principale 1")
plt.ylabel("composante principale 2")
plt.grid(True)
plt.show()
# Ajout de la légende
import matplotlib.patches as mpatches
cmap = plt.cm.get_cmap('tab10', 3)#best_k
handles = [mpatches.Patch(color=cmap(i), label=f'Cluster {i}') for i in range(3)]#best_k
plt.legend(handles=handles, title="Clusters", loc="best")

plt.show()

#clustering(DBSCAN)
#from sklearn.cluster import DBSCAN
#dbscan = DBSCAN(eps=0.7, min_samples=10)
#labels = dbscan.fit_predict(X_scaled)
#df['cluster_DBSCAN'] = labels
#labels_set = set(labels)#transforme la liste en un ens. unique
#if -1 in labels_set:
#    labels_set.remove(-1) #retire 1 de la liste s'il y a -1
#n_clusters = len(labels_set)
#n_bruit = list(labels).count(-1)
#print(f"Nombre de clusters trouvés hors bruit:",n_clusters)
#print(" Nombre de point considéres comme bruit:",n_bruit)
#visualisation
#plt.figure(figsize=(10,6))
#plt.scatter(X_pca[:, 0], X_pca[:, 1],c=labels, cmap='viridis', s=10)
#plt.title("Clusters DBSCAN visualisés sur 2 composantes PCA")
#plt.xlabel("composante principale 1")
#plt.ylabel("composante principale 2")
#plt.grid(True)
#plt.show()
# Ajout de la légende
#import matplotlib.patches as mpatches
#unique_labels = sorted(set(labels))
#cmap = plt.cm.get_cmap('viridis', len(unique_labels))

#handles = []
#for idx, label in enumerate(unique_labels):
 #   color = 'black' if label == -1 else cmap(idx)
  #  nom = "Bruit" if label == -1 else f"Cluster {label}"
   # handles.append(mpatches.Patch(color=color, label=nom))

#plt.legend(handles=handles, title="Clusters", loc="best")

#plt.show()

# evaluation des cluster
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# Évaluation du clustering KMeans
#print("\nÉvaluation du clustering KMeans :")
#print(f"Silhouette Score : {silhouette_score(X_scaled, labels_kmeans):.4f}")
#print(f"Calinski-Harabasz Index : {calinski_harabasz_score(X_scaled, labels_kmeans):.2f}")
#print(f"Davies-Bouldin Index : {davies_bouldin_score(X_scaled, labels_kmeans):.4f}")

# Caracteristiques des cluster
df['cluster_kmeans'] = labels_kmeans # Ajout du label de cluster KMeans au DataFrame
print("\n==== Analyse statistique des clusters KMeans ====") # Statistiques descriptives par cluster
kmeans_stats = df.groupby('cluster_kmeans')[['SOG', 'COG', 'Heading', 'LAT', 'LON']].agg(['mean', 'std', 'min', 'max'])
print(kmeans_stats)

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5)) # Vitesse (SOG) par cluster
sns.boxplot(x='cluster_kmeans', y='SOG', data=df)
plt.title("Distribution de la vitesse (SOG) par cluster KMeans")
plt.grid()
plt.show()

plt.figure(figsize=(8,5)) # Cap (COG) par cluster
sns.boxplot(x='cluster_kmeans', y='COG', data=df)
plt.title("Distribution du cap (COG) par cluster KMeans")
plt.grid()
plt.show()

plt.figure(figsize=(8,5)) # Direction (Heading) par cluster
sns.boxplot(x='cluster_kmeans', y='Heading', data=df)
plt.title("Distribution de la direction (Heading) par cluster KMeans")
plt.grid()
plt.show()


# Évaluation du clustering DBSCAN (uniquement si ≥ 2 clusters détectés)
#if n_clusters >= 2:
 #   mask = labels != -1
  #  print("\nÉvaluation du clustering DBSCAN :")
   # print(f"Silhouette Score : {silhouette_score(X_scaled[mask], labels[mask]):.4f}")
    #print(f"Calinski-Harabasz Index : {calinski_harabasz_score(X_scaled[mask], labels[mask]):.2f}")
    #print(f"Davies-Bouldin Index : {davies_bouldin_score(X_scaled[mask], labels[mask]):.4f}")
#else:
 #   print("\nÉvaluation du clustering DBSCAN : non applicable (moins de 2 clusters)")

#Création de la carte pour kmeans
import plotly.express as px
px.set_mapbox_access_token("pk.eyJ1IjoibWFyeXNlZSIsImEiOiJjbWMzNWJncm0wMWVrMmtxd3B1YzB6eHVuIn0.Qt5mvwYrebHolFbJzPOipw")
import plotly.graph_objects as go
from sklearn.cluster import KMeans
df_kmeans = df.copy()
df_kmeans["cluster_kmeans"] = labels_kmeans.astype(str)
# Pour garder la carte fluide (>75000 pts → on échantillonne)
def sample_df(dframe, n=75_000):
    return dframe.sample(n=min(len(dframe), n), random_state=42)
data_kmeans  = sample_df(df_kmeans)
fig_k = px.scatter_mapbox(
    data_kmeans,
    lat="LAT", lon="LON",
    color="cluster_kmeans",
    hover_data={"SOG": True, "COG": True, "Heading": True, "cluster_kmeans": True},
    zoom=3,
    height=650,
    title="Trajectoires – Clustering K‑Means",
    mapbox_style="carto-positron",
    color_discrete_sequence=px.colors.qualitative.Plotly
)
fig_k.show()



# Ce que j'utilise pour script
with open("scale_1.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model_1.pkl", "wb") as f:
    pickle.dump(kmeans, f)

#3 Carte DBSCAN
#df_dbscan = df.copy()
#data_dbscan = sample_df(df_dbscan)
#df_dbscan["cluster_DBSCAN"] = labels.astype(str)
#fig_d = px.scatter_mapbox(
 #   data_dbscan,
  #  lat="LAT", lon="LON",
   # color="cluster_DBSCAN",
    #hover_data={"SOG": True, "COG": True, "Heading": True, "cluster_DBSCAN": True},
    #zoom=3,
   # height=650,
    #title="Trajectoires – Clustering DBSCAN",
   # mapbox_style="carto-positron",
   # color_discrete_sequence=px.colors.qualitative.Vivid
#)
#fig_d.show()