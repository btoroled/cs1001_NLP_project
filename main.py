import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

data = pd.read_csv('smogon.csv')
vector = CountVectorizer()
x = vector.fit_transform(data['texto'])
header = (sorted(vector.vocabulary_))
tabla = pd.DataFrame(data=x.toarray(),columns=header)
km = KMeans(n_clusters=20,random_state=98)
cluster = km.fit_predict(tabla)
tabla["clusters"] = cluster
print(tabla)
#tbf = pd.DataFrame(data=x.toarray(), columns=cabecera)

#km = KMeans(n_clusters=20, random_state=98)

#lista_clusters = km.fit_predict(tbf)
#tbf["Cluster"] = lista_clusters

#tbf.to_csv("tfid_smogon.csv")
