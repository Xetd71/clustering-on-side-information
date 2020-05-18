import numpy as np
from sklearn.preprocessing import MinMaxScaler


def get_constrains_by_quantile(side_dist, quantile):
    ml_th = np.quantile(side_dist, quantile)
    cl_th = 1 - ml_th
    ml_pairs = (side_dist <= ml_th).astype(float)
    cl_pairs = (side_dist >= cl_th).astype(float)
    constrains = ml_pairs - cl_pairs
    return constrains


def labels_to_constraints(labels):
    return (labels.reshape(-1, 1) == labels.reshape(1, -1)).astype(float) * 2 - 1


def get_side_dist(side_information, pair_dist=None, labels=None, side_information_weight=1.0, pair_dist_weight=1.0, labels_weight=1.0):
    if labels is None:
        labels = np.zeros(side_information.shape)

    if pair_dist is None:
        pair_dist = np.zeros(side_information.shape)

    labels_dist = (labels.reshape(-1, 1) != labels.reshape(1, -1)).astype(float)
    side_dist = MinMaxScaler().fit_transform(
        MinMaxScaler().fit_transform(side_information) * side_information_weight +
        MinMaxScaler().fit_transform(pair_dist) * pair_dist_weight +
        MinMaxScaler().fit_transform(labels_dist) * labels_weight
    )

    return side_dist


def get_constraints(constraints_percentage: float, noise_percentage: float, side_dist=None):
    noise = np.random.uniform(-noise_percentage, noise_percentage, side_dist.shape)
    constraints = get_constrains_by_quantile(side_dist + noise, constraints_percentage / 2)
    return constraints
