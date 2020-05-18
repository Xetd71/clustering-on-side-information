import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm.auto import tqdm

from src.kmeans_base import KMeansBase

np.seterr('raise')


class MPCKMeans(KMeansBase):
    "MPCK-Means-S-D that learns only a single (S) diagonal (D) matrix"

    def __init__(self, n_clusters=3, max_iter=30, w=0.001, eps=1e-6, verbose=1):
        super().__init__(n_clusters, max_iter, w, eps, verbose)

    def fit(self, X, y=None, constraints=None):
        # Preprocess constraints

        # Initialize cluster centers
        self.cluster_centers = self._init_cluster_centers(X)

        # Initialize metrics
        A = np.identity(X.shape[1])

        # Repeat until convergence
        for step in np.arange(self.max_iter):
            # Find farthest pair of points according to each metric
            farthest_dist, farthest_index = self._find_farthest_pairs_of_points(X, A)

            # Assign clusters
            labels = self._assign_clusters(X, self.cluster_centers, A, farthest_dist, constraints, self.w)

            # Estimate means
            prev_cluster_centers = self.cluster_centers
            self.cluster_centers = self._get_cluster_centers(X, labels)

            # Update metrics
            A = self._update_metrics(X, labels, self.cluster_centers, farthest_index, constraints, self.w)

            # Check for convergence
            cluster_shift = ((prev_cluster_centers - self.cluster_centers_)**2).sum(axis=1)
            if self.verbose:
                print(f"step = {step}, loss = {cluster_shift.mean():.6f}")
            if (cluster_shift <= self.eps).all():
                break

        return self

    def _find_farthest_pairs_of_points(self, X, A):
        farthest_index = np.unravel_index(pairwise_distances(X).argmax(), (X.shape[0], X.shape[0]))
        farthest_dist = self._dist(X[farthest_index[0]], X[farthest_index[1]], A)
        return farthest_dist, farthest_index

    def _dist(self, x, y, A):
        return np.sqrt((x - y).dot(A).dot((x - y).T))

    def _dist_matrix(self, X, y, A):
        return np.array([np.sqrt((y - x).dot(A).dot((y - x).T)) for x in X])

    def _assign_clusters(self, X, cluster_centers, constraints, w):
        centroid_distances = []
        for centroid in cluster_centers:
            cluster_distances = ((X - centroid)**2).sum(axis=1)
            centroid_distances.append(cluster_distances)
        centroid_distances = np.array(centroid_distances)

        labels = np.full(X.shape[0], fill_value=-1)
        constraints_distances = np.zeros(centroid_distances.shape)
        np.random.seed(self.random_seed)
        x_indexes = np.random.choice(X.shape[0], X.shape[0], replace=False)
        for x_id in x_indexes:
            x_label = (centroid_distances[:, x_id] + constraints_distances[:, x_id]).argmin()
            labels[x_id] = x_label
            constraints_distances[x_label] -= constraints[x_id] * w
        return labels

    def _assign_clusters(self, X, cluster_centers, A, farthest_dist, constraints, w):
        centroid_distances = []
        for centroid in cluster_centers:
            cluster_distances = self._dist_matrix(X, centroid, A)
            centroid_distances.append(cluster_distances)
        centroid_distances = np.array(centroid_distances)

        ML = (constraints > 0).astype(float)
        CL = (constraints < 0).astype(float)

        labels = np.full(X.shape[0], fill_value=-1)
        constraints_distances = np.zeros(centroid_distances.shape)
        np.random.seed(self.random_seed)
        x_indexes = np.random.choice(X.shape[0], X.shape[0], replace=False)
        for x_id in tqdm(x_indexes):
            x_label = (centroid_distances[:, x_id] + constraints_distances[:, x_id]).argmin()
            labels[x_id] = x_label
            x_distances = self._dist_matrix(X, X[x_id], A)
            constraints_distances[x_label] -= ML[x_id] * w * x_distances
            constraints_distances[x_label] += CL[x_id] * w * (farthest_dist - x_distances)
        return labels

    def _update_metrics(self, X, labels, cluster_centers, farthest_index, constraints, w):
        TERM_X = ((X - cluster_centers[labels])**2).sum(axis=0)
        T = (X[farthest_index[0]] - X[farthest_index[1]])**2

        label_similarity = (labels.reshape(-1, 1) == labels.reshape(1, -1)).astype(float)
        ML = (constraints > 0).astype(float) * (1 - label_similarity)
        CL = (constraints < 0).astype(float) * label_similarity

        N, D = X.shape
        A = np.zeros((D, D))
        for d in range(D):
            X_d_diff = pairwise_distances(X[:, d])
            term_x = TERM_X[d]
            term_m = 1 / 2 * (w * ML * X_d_diff)
            term_c = w * CL * (T[d] - X_d_diff)
            A[d, d] = N / max(term_x + term_m + term_c, 1e-9)
        return A


if __name__ == "__main__":
    import os

    os.chdir("/Users/xetd71/Yandex.Disk.localized/Projects/clustering-on-side-information/data")

    with open("X.npy", 'br') as f:
        X = np.load(f).astype(float)
    with open("constrains_01.npy", 'br') as f:
        # constrains = np.ma.masked_values(np.load(f).astype(float), 0.0)
        constraints = np.load(f).astype(float)

    pck = MPCKMeans(n_clusters=40, w=0.001)
    pck.fit(X=X, constraints=constraints)
