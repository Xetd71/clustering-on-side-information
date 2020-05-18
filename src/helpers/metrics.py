import os
import pandas as pd
import numpy as np
from sklearn import metrics

from .default_values import DefaultValues


def compute_clustering_metrics(labels_pred, labels_true=None):
    if labels_true is None:
        data_path = os.path.join(DefaultValues.DATA_PATH, "labels.npy")
        with open(data_path, "br") as f:
            labels_true = np.load(f, allow_pickle=True)

    return pd.Series({
        "ARI": metrics.adjusted_rand_score(labels_true, labels_pred),
        "AMI": metrics.adjusted_mutual_info_score(labels_true, labels_pred),
        "V-measure": metrics.v_measure_score(labels_true, labels_pred),
    })


def compute_side_information_metrics(labels_pred, side_information=None):
    if side_information is None:
        data_path = os.path.join(DefaultValues.DATA_PATH, "side_information.npy")
        with open(data_path, "br") as f:
            side_information = np.load(f).astype(float)

    return pd.Series({
        "Silhouette": metrics.silhouette_score(side_information, labels_pred)
    })
