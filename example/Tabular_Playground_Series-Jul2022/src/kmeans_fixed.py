import pandas as pd
from clustering import Clustering

DATA_PATH = "example/Tabular_Playground_Series-Jul2022/data/raw/data.csv"

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["id"]).sample(n=5_000, random_state=42)

clf = Clustering(algorithm="kmeans", n_clusters=5)

clf.fit(X)

print(f"k={clf.n_clusters_}")

X_new = df.drop(columns=["id"]).sample(n=5, random_state=99)
print("\npredict(X_new):", clf.predict(X_new))

clf.plot()
