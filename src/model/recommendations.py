import numpy as np
import pandas as pd

_KWARG_TOP_N = 'top_n'
_TOP_N_DEFAULT = [5]


def recommend(*args, **kwargs):
    model_recommender, x_df, cols = args
    top_n = kwargs.get(_KWARG_TOP_N, _TOP_N_DEFAULT)

    y_pred = []
    top_n_max = np.max(top_n)
    for _, (number, files) in x_df.iterrows():
        _y_pred = model_recommender.recommend((files, files), N=top_n_max)
        y_pred.append([number, *[_y_pred[:n] for n in top_n]])

    y_df = pd.DataFrame(y_pred, columns=['number', *[f'top-{n}' for n in top_n]])

    return y_df
