#Import
from imports import *

################################
# Pregunta 1
################################

#Leyendo el documento y entrendando el algoritmo
data = pd.read_csv("csv/smogon.csv")
vector = TfidfVectorizer(stop_words= "english",ngram_range= (1,2)) # se utliza todo el vocabulario dentro de la tabla. Se utiliza "ingles" como un stop word y se analiza utilizando unigramas y bigramas
x = vector.fit_transform(data["moves"])
header = (sorted(vector.vocabulary_.keys())) 
table = pd.DataFrame(data=x.toarray(), columns=header) #Se crea la tabla

#Kmeans 
km = KMeans(n_clusters = 18,random_state = 1)
cluster = km.fit_predict(table)
table["cluster"] = cluster #Se agrega los clusters a la tabla
table["Pokemon"] = data["Pokemon"] # se agrega la columna Pokemon

#Realizando el merge
types = pd.read_csv("csv/pokedex.csv") # Se utiliza la base de datos "pokedex.csv descargado de Kaggle"
fusion = pd.merge(types[["Name","Type 1","Type 2"]], table[["Pokemon","cluster"]],left_on="Name",right_on="Pokemon",how = "inner") #Realizmos el merge en esta linea. Poniendo primero la lista de tipos y luego la tabla de smogon.csv. Creando asi una nueva tabla ordenada por numero en la pokedex.
fusion = fusion.dropna(subset=["Pokemon","cluster"]) #se elimina las columnas que tienen datos faltantes
fusion = fusion.drop(columns="Pokemon")
fusion = fusion[fusion["Name"]!=""]
fusion["cluster"]=fusion["cluster"].astype(int)

fusion.to_csv("tabla_final_pregunta1.csv") #se realiza el csv del texto.

#OUTPUT CONSOLA
print("\nResumen de matriz TF-IDF: ")
print(f"\nPokemon procesados:{table.shape[0]}")
print(x)
print()
print("\nTabla de frecuencias")
print(x.toarray())
print(f"\nEl total de tokens es: {len(header)}")
print("\nEl vocabulario del archivo es:")
print(sorted(vector.vocabulary_))
print("\nLas claves del diccionario son: ")
print(sorted(vector.vocabulary_.keys()))
print("\nLa matriz de fercuencias: ")
print(fusion)

# Contar cuántas veces se repite cada tipo por cluster (Type 1)
type1_counts = fusion.groupby("cluster")["Type 1"].value_counts().unstack(fill_value=0)

# Contar cuántas veces se repite cada tipo por cluster (Type 2)
type2_counts = fusion.groupby("cluster")["Type 2"].value_counts().unstack(fill_value=0)

print("\nConteo de tipos (Type 1) por cluster:")
print(type1_counts)

print("\nConteo de tipos (Type 2) por cluster:")
print(type2_counts)

fusion_melted = pd.melt(fusion, id_vars=["cluster"], value_vars=["Type 1", "Type 2"], value_name="Type")
type_total_counts = fusion_melted.groupby(["cluster", "Type"]).size().unstack(fill_value=0)

type_total_counts["Total"] = type_total_counts.sum(axis=1)

print("\nTotal de coincidencias de tipos por cluster:")
print(type_total_counts["Total"])

print("\nConteo total de tipos (Type 1 + Type 2) por cluster:")
print(type_total_counts)
# Guardar los conteos de tipos por cluster en un CSV
type_total_counts.to_csv("conteo_tipos_por_cluster.csv")