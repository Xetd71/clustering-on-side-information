import numpy as np

from src.kmeans_base import KMeansBase


class PCKMeans(KMeansBase):
    def __init__(self, n_clusters=3, max_iter=30, w=0.001, eps=1e-6, verbose=1, random_seed=0):
        super().__init__(n_clusters, max_iter, w, eps, verbose, random_seed=0)

    def fit(self, X, y=None, constraints=None):
        # Initialize centroids; note init not like in PCKMeans
        self.cluster_centers_ = self._init_cluster_centers(X)

        # Repeat until convergence
        for step in np.arange(self.max_iter):
            # Assign clusters
            self.labels_ = self._assign_clusters(X, self.cluster_centers_, constraints, self.w)

            # Estimate means
            prev_cluster_centers = self.cluster_centers_
            self.cluster_centers_ = self._get_cluster_centers(X, self.labels_)

            # Check for convergence
            cluster_shift = ((prev_cluster_centers - self.cluster_centers_)**2).sum(axis=1)
            if self.verbose:
                print(f"step = {step}, loss = {cluster_shift.mean():.6f}")
            if (cluster_shift <= self.eps).all():
               break
        return self

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


if __name__ == "__main__":
    import os

    os.chdir("/Users/xetd71/Yandex.Disk.localized/Projects/clustering-on-side-information/data")

    with open("X.npy", 'br') as f:
        X = np.load(f).astype(float)
    with open("constrains_01.npy", 'br') as f:
        # constraints = np.ma.masked_values(np.load(f).astype(float), 0.0)
        constraints = np.load(f).astype(float)

    pck = PCKMeans(n_clusters=40, w=0.001)
    pck.fit(X=X, constraints=constraints)

