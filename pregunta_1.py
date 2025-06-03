#Import
from imports import *

#Carga de CSV
data = pd.read_csv('smogon.csv')
#################################

#Pregunta 1

#################################

#TFID sobre la columna texto
vector = TfidfVectorizer()
x = vector.fit_transform(data['texto'])

#Creacion de Dataframe con los tokens ordenados
header = (sorted(vector.vocabulary_))
tabla = pd.DataFrame(data=x.toarray(),columns=header)

#Clustering con KMeans 
km = KMeans(n_clusters=18,random_state=98)
cluster = km.fit_predict(tabla)
tabla["clusters"] = cluster

#tipos de pokemon que deseamos
types=["fire","water","grass","electric","rock","ground","flying","normal","fighting","psychic","dark","ghost","bug","dragon","ice","fairy","steel","poison"]
print(header)
#Crear dataframe final 
pokemones_movelist = pd.concat([data["Pokemon"], tabla[types],tabla["clusters"]], axis=1)

pokemones_movelist.to_csv("tfid_smogon.csv", index=False)
