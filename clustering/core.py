import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as _StandardScaler
from sklearn.cluster import KMeans as _KMeans, DBSCAN as _DBSCAN
from sklearn.metrics import (
    silhouette_score,
)
from sklearn.decomposition import PCA


class Clustering:
    def __init__(
        self,
        algorithm="kmeans",
        reducing="PCA",
        scale=True,
        n_clusters=None,
        seed=42,
        **kwargs,
    ) -> None:
        self.algorithm = algorithm
        self.reducing = reducing
        self.scale = scale
        self.n_clusters = n_clusters
        self.seed = seed
        self.kwargs = kwargs
        self.labels_ = None
        self.n_clusters_ = None
        self._model = None
        self._X_scaled = None

    def fit(self, X):
        self._X_scaled = self._scaler(X)

        if self.algorithm == "kmeans":
            self._fit_kmeans(self._X_scaled)
        elif self.algorithm == "dbscan":
            self._fit_dbscan(self._X_scaled)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm!r}. Use 'kmeans' or 'dbscan'.")

        return self

    def predict(self, X):
        """Assign cluster labels to new data points."""
        if self.labels_ is None:
            raise RuntimeError("Call fit() first.")
        X_scaled = self._scaler(X)
        if self.algorithm == "kmeans":
            return self._model.predict(X_scaled)
        if self.algorithm == "dbscan":
            # DBSCAN has no native predict — assign by nearest cluster centroid
            centroids = np.array([
                self._X_scaled[self.labels_ == k].mean(axis=0)
                for k in range(self.n_clusters_)
            ])
            if len(centroids) == 0:
                return np.full(len(X_scaled), -1)
            dists = np.linalg.norm(X_scaled[:, None] - centroids[None, :], axis=2)
            return dists.argmin(axis=1)
        raise ValueError(f"predict() not supported for algorithm={self.algorithm!r}.")

    def plot(self,  figsize=(8, 6)):
        """Scatter plot of clusters projected to 2D."""
        if self.labels_ is None:
            raise RuntimeError("Call fit() first.")

        X_2d = self._reduce(self._X_scaled)
        df_plot = pd.DataFrame(X_2d, columns=["PC1", "PC2"])
        df_plot["cluster"] = self.labels_.astype(str)

        _, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(
            data=df_plot, x="PC1", y="PC2",
            hue="cluster", palette="tab10",
            s=40, linewidth=0, ax=ax,
        )
        s = self.evaluate()["value"]["Silhouette"]
        score_str = f"{s:.3f}" if not np.isnan(s) else "nan"
        ax.set_title(
            f"{self.algorithm.upper()} · {self.n_clusters_} clusters"
            f" · silhouette={score_str}"
        )
        plt.tight_layout()
        plt.show()
        return ax

    # ------------------------------------------------------------------

    def _scaler(self, X):
        X_arr = X.values if hasattr(X, "values") else np.array(X)
        return _StandardScaler().fit_transform(X_arr) if self.scale else X_arr.astype(float)

    def _reduce(self, X):
        if self.reducing == "PCA":
            return PCA(n_components=2, random_state=self.seed).fit_transform(X)
        raise ValueError(
            f"Unknown reducing: {self.reducing!r}."
        )

    def _fit_kmeans(self, X):
        k = self.n_clusters if self.n_clusters is not None else self._best_k(X)
        self._model = _KMeans(n_clusters=k, random_state=self.seed, **self.kwargs)
        self.labels_ = self._model.fit_predict(X)
        self.n_clusters_ = k

    def _best_k(self, X, k_min=2, k_max=11):
        scores = [
            silhouette_score(X, _KMeans(n_clusters=k, random_state=self.seed).fit_predict(X))
            for k in range(k_min, k_max)
        ]
        return k_min + int(np.argmax(scores))

    def _fit_dbscan(self, X):
        self._model = _DBSCAN(**self.kwargs)
        self.labels_ = self._model.fit_predict(X)
        self.n_clusters_ = len(set(self.labels_) - {-1})


