import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

################################
#pregunta 1
################################

data = pd.read_csv("smogon.csv")
vector = TfidfVectorizer(stop_words= "english",ngram_range= (1,2))
x = vector.fit_transform(data["moves"])
header = (sorted(vector.vocabulary_))
print("El total de tokens es: ", len(header))
table = pd.DataFrame(data=x.toarray(), columns=header)

#haciendo el KMeans

km = KMeans(n_clusters = 18,random_state = 1)
cluster = km.fit_predict(table)
table["cluster"] = cluster

table["Pokemon"] = data["Pokemon"]


types = pd.read_csv("Pokemon.csv")
Fusion = pd.merge(table[["Pokemon","cluster"]],types[["Name","Type 1","Type 2"]],left_on="Pokemon",right_on="Name",how = "inner")
Fusion = Fusion.dropna(subset=["Pokemon","cluster"])
Fusion = Fusion[Fusion["Pokemon"]!=""]
Fusion["cluster"]=Fusion["cluster"].astype(int)
print(Fusion)
Fusion.to_csv("Fusion.csv")