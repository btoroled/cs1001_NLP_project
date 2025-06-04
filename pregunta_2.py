#Import
from imports import *

#Carga de CSV
data = pd.read_csv('csv/smogon.csv')
#################################

#Pregunta 2

#################################

#TFID sobre la columna texto
vector = TfidfVectorizer()
x = vector.fit_transform(data['moves'])

#Creacion de Dataframe con los tokens ordenados
header = (sorted(vector.vocabulary_))
tabla = pd.DataFrame(data=x.toarray(),columns=header)

#Clustering con KMeans 
km = KMeans(n_clusters=18,random_state=98)
cluster = km.fit_predict(tabla)
tabla["clusters"] = cluster

#tipos de pokemon que deseamos
types=["fire","water","grass","electric","rock","ground","flying","normal","fighting","psychic","dark","ghost","bug","dragon","ice","fairy","steel","poison"]
print(header)#print de los tokens
print(f"Total de tokens: {len(header)}") #print del total de tokens
#Crear dataframe final 
pokemones_movelist = pd.concat([data["Pokemon"], tabla[types],tabla["clusters"]], axis=1)

pokemones_movelist.to_csv("tfid_moveset.csv", index=False)


#################################
# Graficando los resultados
#################################

#Se copia el dataframe para evitar modificar el original
df = pokemones_movelist.copy()

#################################
# Grafico de Heatmaps
#################################

#Calculamos los promedios de TFID por tipo de cluster
cluster_means = df.groupby('clusters').mean(numeric_only=True)
#Exluir la columna de clusters del dataframe
cluster_means = cluster_means.drop(columns=['clusters'], errors='ignore')

#Creamos heatmap de los promedios por tipo de cluster
plt.figure(figsize=(14, 8))
sns.heatmap(cluster_means, annot=False, cmap="YlGnBu", linewidths=0.5)
plt.title("Mapa de calor de movimientos por tipo de pokemon")
plt.xlabel("Tipos de Movimiento")
plt.ylabel("Clusters")
plt.tight_layout
plt.show()

#################################
# Grafico de Barras de Cantidad de movimientos por Cluster
#################################

#Creamos un DataFrame para los conteos de coicidencia por clusters
cluster_counts = df['clusters'].value_counts().sort_index()

#Creamos un gráfico de barras para visualizar la cantidad de Pokémon por cluster
plt.figure(figsize=(10,6))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')
plt.title("Cantidad de Pokemon por Cluster")
plt.xlabel("Clusters")
plt.ylabel("Cantidad de Pokemon")
plt.tight_layout()
plt.show()
