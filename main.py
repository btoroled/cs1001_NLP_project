import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

pokemones = pd.read_csv("smogon.csv")

vector = CountVectorizer()

x=vector.fit_transform(pokemones["texto"])
cabecera = sorted(vector.vocabulary_)

tbf = pd.DataFrame(data=x.toarray(), columns=cabecera)

km = KMeans(n_clusters=20, random_state=98)

lista_clusters = km.fit_predict(tbf)
tbf["Cluster"] = lista_clusters

tbf.to_csv("tfid_smogon.csv")