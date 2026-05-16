import pandas as pd
from clustering import Clustering

DATA_PATH = "example/Tabular_Playground_Series-Jul2022/data/raw/data.csv"

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["id"]).sample(n=5_000, random_state=42)

clf = Clustering(algorithm="kmeans")
clf.fit(X)

print(f"Optimal k: {clf.n_clusters_}")

print(clf.evaluate().to_string())

clf.plot()
