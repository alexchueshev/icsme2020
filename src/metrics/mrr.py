import numpy as np


def mean_reciprocal_rank(y_true, y_pred):
    ranks = []

    for _y_true, _y_pred in zip(y_true, y_pred):
        _, _, y_pred_ind = np.intersect1d(_y_true, _y_pred, return_indices=True)
        if y_pred_ind.size:
            ranks.append(y_pred_ind[0] + 1)

    return np.sum(1 / np.array(ranks)) / len(y_true)
