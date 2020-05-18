import numpy as np
from scipy.linalg import eigh
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans


def compute_affinity_matrix(X):
    A = pairwise_distances(X)
    return A


class CSP:
    def __init__(self, n_clusters=3, max_iter=30, betta=1, verbose=1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.betta = betta
        self.verbose = verbose
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, A, constraints):
        n = A.shape[0]
        vol = A.sum()
        D = np.diag(A.sum(axis=0) ** (-0.5))
        L = np.identity(n) - D.dot(A).dot(D)
        Q = D.dot(constraints).dot(D)

        eigvals, eigvects = eigh(L, Q - self.betta/vol * np.identity(n))
        V_ = eigvects.real
        V = (V_.T / np.linalg.norm(V_, ord=2, axis=1)).T * (vol ** 0.5)

        dist = [v.T.dot(L).dot(v) for v in V]
        V_k = V[np.argsort(dist)[:self.n_clusters].tolist()]
        X_embedding = D.dot(V_k.T)
        self.kmeans = KMeans(self.n_clusters)
        self.labels_ = self.kmeans.fit_predict(X_embedding)
        return self












if __name__ == "__main__":
    import os

    os.chdir("/Users/xetd71/Yandex.Disk.localized/Projects/clustering-on-side-information/data")

    with open("X.npy", 'br') as f:
        X = np.load(f).astype(float)
    with open("constrains_01.npy", 'br') as f:
        # constraints = np.ma.masked_values(np.load(f).astype(float), 0.0)
        constraints = np.load(f).astype(float)

    csp = CSP(n_clusters=40, w=0.001)
    pck.fit(X=X, constraints=constraints)
