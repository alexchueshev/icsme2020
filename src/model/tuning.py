from hopts import GridSearchALS, BayesSearchALS
from transformer import MatrixTransformer, MatrixLinearTransformer
from estimator import ALSEstimator
from scorer import mpr_scorer

from .utils import pipe_hyperparameters

_STEP_TRANSFORM = 'transform'
_STEP_ALS = 'als'


def _factory(*args, **kwargs):
    typ, x_df, cols, hyperparameters = args

    col_item, col_user, _ = cols
    hyperparameters = pipe_hyperparameters(hyperparameters, [_STEP_TRANSFORM, _STEP_ALS])

    shape = (len(x_df[col_item].unique()), len(x_df[col_user].unique()))
    groups = (x_df.groupby(col_item).ngroup(), x_df.groupby(col_user).ngroup())
    transform_y = MatrixTransformer(cols, shape, groups).fit_transform
    pipe = [
        (_STEP_TRANSFORM, MatrixLinearTransformer(cols, shape, groups)),
        (_STEP_ALS, ALSEstimator())
    ]

    if typ == 'grid':
        return GridSearchALS(pipe, hyperparameters, mpr_scorer, **kwargs), transform_y
    elif typ == 'bayes':
        return BayesSearchALS(pipe, hyperparameters, mpr_scorer, **kwargs), transform_y
    else:
        raise NotImplementedError()


def grid_search_als(*args, **kwargs):
    x_df, cols, hyperparameters = args
    y_df = x_df.copy()

    gs, transform_y = _factory('grid', x_df, cols, hyperparameters, **kwargs)

    return gs.fit(x_df, y_df, cols, transform_y=transform_y)


def bayes_search_als(*args, **kwargs):
    x_df, cols, hyperparameters = args
    y_df = x_df.copy()

    bs, transform_y = _factory('bayes', x_df, cols, hyperparameters, **kwargs)

    return bs.fit(x_df, y_df, cols, transform_y=transform_y)
