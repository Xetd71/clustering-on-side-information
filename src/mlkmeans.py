import numpy as np
from sklearn.cluster import KMeans
from metric_learn import MMC


class MKMeans:
    def __init__(self, n_clusters=3, max_iter=300, diagonal=True):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.diagonal = diagonal

    def fit(self, X, y=None, constraints=None):
        mmc = MMC(diagonal=self.diagonal)
        mmc.fit(X, constraints=constraints)
        X_transformed = mmc.transform(X)
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            init='random',
            max_iter=self.max_iter
        )
        kmeans.fit(X_transformed)
        self.labels_ = kmeans.labels_
        return self
