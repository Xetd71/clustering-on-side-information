import numpy as np


class KMeansBase:
    def __init__(self, n_clusters=3, max_iter=30, w=0.001, eps=1e-6, verbose=1, random_seed=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.w = w
        self.eps = eps
        self.verbose = verbose
        self.random_seed = random_seed
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, X, y=None, **kwargs):
        pass

    def fit_predict(self, X, y=None, **kwargs):
        self.fit(X, y=None, **kwargs)
        return self.labels_

    def _init_cluster_centers(self, X, y=None):
        np.random.seed(self.random_seed)
        return X[np.random.choice(X.shape[0], self.n_clusters, replace=False), :]

    def _assign_clusters(self, *args, **kwargs):
        pass

    def _get_cluster_centers(self, X, labels):
        return np.array([
            X[labels == label].mean(axis=0)
            for label in np.arange(self.n_clusters)
        ])