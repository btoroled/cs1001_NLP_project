#Import
from imports import *

#Carga de CSV
data = pd.read_csv('csv/smogon.csv')
#################################

#Pregunta 2

#################################
#Tipos de pokemon que deseamos
types=["fire","water","grass","electric","rock","ground","flying","normal","fighting","psychic","dark","ghost","bug","dragon","ice","fairy","steel","poison"]
def extract_types(text, types):
    found_types = set()
    words = str(text).lower().split()  # convierte en texto en minusculas y lo separa
    for t in types:
        if any(t in word for word in words):  # Chequea si el tipo existe en alguna palabra
            found_types.add(t)  
    return " ".join(found_types)
data['processed_moves'] = data['moves'].apply(lambda x: extract_types(x, types))
#TFID sobre la columna texto
vector = TfidfVectorizer()
x = vector.fit_transform(data['processed_moves'])

#Creacion de Dataframe con los tokens ordenados
header = (sorted(vector.vocabulary_))
tabla = pd.DataFrame(data=x.toarray(),columns=header)

#Clustering con KMeans 
km = KMeans(n_clusters=18,random_state=98)
cluster = km.fit_predict(tabla)
tabla["clusters"] = cluster

print(header)#print de los tokens
print(f"Total de tokens: {len(header)}") #print del total de tokens
#Crear dataframe final 
available_types = list(set(types) & set(tabla.columns))  # Conseguir las columnas
pokemones_movelist = pd.concat([data["Pokemon"], tabla[available_types], tabla["clusters"]], axis=1)

pokemones_movelist.to_csv("tfid_moveset.csv", index=False)


#################################
# Graficando los resultados
#################################

#Se copia el dataframe para evitar modificar el original
df = pokemones_movelist.copy()

#################################
# Grafico de Heatmaps
#################################

# Agrupar por clusters y calcular promedios
cluster_means = df.groupby('clusters').mean().fillna(0)

# Crear heatmap con etiquetas y diseño mejorado
plt.figure(figsize=(14, 8))
sns.heatmap(cluster_means, annot=True, cmap="YlGnBu", linewidths=0.5, fmt=".2f")

plt.title("Mapa de calor de movimientos por tipo de Pokémon")
plt.xlabel("Tipos de Movimiento")
plt.ylabel("Clusters")
plt.tight_layout()
plt.show()

