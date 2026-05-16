import pandas as pd
from clustering import Clustering

DATA_PATH = "example/Tabular_Playground_Series-Jul2022/data/raw/data.csv"

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["id"]).sample(n=5_000, random_state=42)

clf = Clustering(algorithm="dbscan", eps=5.0, min_samples=5)
clf.fit(X)

noise_pct = (clf.labels_ == -1).mean() * 100

print(f"clusters={clf.n_clusters_}  noise={noise_pct:.1f}%")

print(clf.evaluate().to_string())

clf.plot()
