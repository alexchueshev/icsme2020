import numpy as np


def mpr_scorer(y_true, y_pred):
    score1 = .0
    score2 = .0
    n_samples = y_true.shape[1]

    for u in range(n_samples):
        col_y_true, col_y_score = y_true[:, u], y_pred[:, u]
        if col_y_true.nnz:
            s1, s2 = _mpr_score(col_y_true
                                .toarray()
                                .reshape((-1,)),
                                col_y_score)
            score1 += s1
            score2 += s2

    return score1 / score2


def _mpr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    ranks = np.sort(np.argsort(y_true)) / (len(y_true) - 1)

    return np.dot(y_true, ranks), np.sum(y_true)
