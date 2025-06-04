import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
data = pd.read_csv("smogon.csv")
vector = TfidfVectorizer(stop_words= "english",ngram_range= (1,2)) # se utliza todo el vocabulario dentro de la tabla. Se utiliza "ingles" como un stop word y se analiza utilizando unigramas y bigramas
x = vector.fit_transform(data["moves"])
header = (sorted(vector.vocabulary_.keys()))
table = pd.DataFrame(data=x.toarray(), columns=header) #Se crea la tabla

#Kmeans
km = KMeans(n_clusters = 18,random_state = 1)
cluster = km.fit_predict(table)

table["cluster"] = cluster

table.to_csv("Table.csv")

#Separador - Pregunta 3
print("------- DataFrame (Con Cluster) -------")
datos3 = pd.read_csv('Table.csv')
datos3.drop(columns=['Unnamed: 0'],inplace=True)
print(datos3)
print(" ! Eliminando la columna de cluster ! ")
datos3.drop(columns=['cluster'],inplace=True) #Elimina la columna "clusters"
print(datos3)

print("------- Separador -------")

print("Aplicando Análisis de Componentes (PCA)")
pca = PCA(n_components=140)
resultado_pca = pca.fit_transform(datos3)
print(resultado_pca)

print("------- Separador: Contadores -------")
print("Número de filas y columnas de la matriz de componentes principales antes:",datos3.shape)
print("Número de filas y columnas de la matriz de componentes principales después:",resultado_pca.shape)


print("------- Separador: Nueva Matriz (PCA1,PCA2,... -------")

cabecera = ["PCA"+ str(i) for i in range(1,141)]
km = KMeans(n_clusters=18, random_state=42)
clusters = km.fit_predict(resultado_pca)
matrizPCA = pd.DataFrame(data=resultado_pca, columns = cabecera)
matrizPCA["Clusters"] =clusters
print(matrizPCA)
print(pca.explained_variance_ratio_) #Componentes de la varianza por separado
print(sum(pca.explained_variance_ratio_)) #Varianza acumulada




