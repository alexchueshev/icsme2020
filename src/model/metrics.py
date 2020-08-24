import numpy as np
import pandas as pd

from metrics import accuracy
from metrics import mean_reciprocal_rank

_KWARG_TOP_N = 'top_n'
_TOP_N_DEFAULT = [5]


def metrics(*args, **kwargs):
    model_recommender, x_df, cols = args
    top_n = kwargs.get(_KWARG_TOP_N, _TOP_N_DEFAULT)

    y_true = []
    y_pred = {n: [] for n in top_n}
    top_n_max = np.max(top_n)
    for _, (_, files, reviewers) in x_df.iterrows():
        _y_pred = model_recommender.recommend((files, files), N=top_n_max)

        y_true.append(reviewers)
        for n in top_n:
            y_pred[n].append(_y_pred[:n] if not pd.isna(_y_pred).any() else [])

    acc = [accuracy(y_true, _y_pred) for n, _y_pred in y_pred.items()]
    mrr = [mean_reciprocal_rank(y_true, _y_pred) for n, _y_pred in y_pred.items()]

    metrics_df = pd.DataFrame([acc, mrr], index=['acc', 'mrr'], columns=[f'top-{n}' for n in top_n])

    return metrics_df

