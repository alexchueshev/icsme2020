import numpy as np


def accuracy(y_true, y_pred):
    intersects = []
    for _y_true, _y_pred in zip(y_true, y_pred):
        intersect = np.intersect1d(_y_true, _y_pred)
        intersects.append(intersect.size > 0)

    return np.sum(intersects) / len(y_true)
