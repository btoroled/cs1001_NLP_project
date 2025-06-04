import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

################################
#pregunta 1
################################

#leyendo el documento y entrendando el algoritmo
data = pd.read_csv("smogon.csv")
vector = TfidfVectorizer(stop_words= "english",ngram_range= (1,2)) # se utliza todo el vocabulario ingles como un stop word y se realizan unigramas y bigramas
x = vector.fit_transform(data["moves"])
header = (sorted(vector.vocabulary_.keys())) 
print("El total de tokens es: ", len(header)) # Se imprime la cantidad de tokens
table = pd.DataFrame(data=x.toarray(), columns=header) #se crea la tabla

#haciendo el KMeans

km = KMeans(n_clusters = 18,random_state = 1)
cluster = km.fit_predict(table)
table["cluster"] = cluster

table["Pokemon"] = data["Pokemon"] # se agrega la columna Pokemon


#Realizando el merge
types = pd.read_csv("Pokemon (1).csv") # Se lee el segundo documento de Alberto Barradas
Fusion = pd.merge(table[["Pokemon","cluster"]],types[["Name","Type 1","Type 2"]],left_on="Pokemon",right_on="Name",how = "inner") #Se realiza el merge base a las columnas puestas
Fusion = Fusion.dropna(subset=["Pokemon","cluster"]) #se elimina las columnas que tienen datos faltantes
Fusion = Fusion[Fusion["Pokemon"]!=""]
Fusion["cluster"]=Fusion["cluster"].astype(int)
print(Fusion) # Se Imprime la matriz con los tipos de los pokemones
Fusion.to_csv("Fusion.csv") #se realiza el csv del texto.

print("\nResumen de matriz TF-IDF: ")
print(f"- pokemon procesados:{table.shape[0]}")
print(x)
print()
print("tabla de frecuencias")
print(x.toarray())
print("El total de tokens es: ", len(header))
print("el vocabulario del archivo es:")
print(sorted(vector.vocabulary_))
print("\nLas claves del diccionario son: ")
print(sorted(vector.vocabulary_.keys()))

print("La matriz de fercuencias: ")

print(Fusion)