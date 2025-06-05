#import 
from imports import * 

################################
# Pregunta 3
################################
print("------- DataFrame (Con Cluster) -------")
datos3 = pd.read_csv('tfid_pregunta1.csv')
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
cabecera = ["PCA"+ str(i) for i in range(1,141)] #Utilizamos un rango de 1 a 141 para crear la cantidad de columnas que necesitamos para que la varianza sea mayor a 80
km = KMeans(n_clusters=18, random_state=42)
clusters = km.fit_predict(resultado_pca)
matrizPCA = pd.DataFrame(data=resultado_pca, columns = cabecera)
matrizPCA["Clusters"] =clusters
print(matrizPCA)
print(pca.explained_variance_ratio_) #Componentes de la varianza por separado
print(sum(pca.explained_variance_ratio_)) #Varianza acumulada


#TABLA - CLUSTERS

df_tfidf = pd.read_csv('Fusion.csv')
df_pca = pd.read_csv('MatrizGenerada(PCA-Cluster).csv')

# Si es necesario, renombra las columnas para unificarlas
df_tfidf.rename(columns={'Pokemon': 'Nombre'}, inplace=True)
df_pca.rename(columns={'Pokemon': 'Nombre'}, inplace=True)

# Unir ambos DataFrames por el nombre
df_comparacion = pd.merge(df_tfidf[['Nombre', 'cluster']], df_pca[['Nombre', 'Clusters']], on='Nombre', how='inner')
df_comparacion.rename(columns={'cluster': 'Cluster_TFIDF', 'Clusters': 'Cluster_PCA'}, inplace=True)

tabla_comparacion = pd.crosstab(df_comparacion['Cluster_TFIDF'], df_comparacion['Cluster_PCA'])
print(tabla_comparacion)




