from typing import TYPE_CHECKING

from sklearn.pipeline import Pipeline

from transformer import MatrixLinearTransformer
from estimator import ALSEstimator

from .utils import pipe_hyperparameters

if TYPE_CHECKING:
    from pandas import DataFrame

_ARG_PIPE_HYPERPARAMETERS = 'pipe_hyperparameters'

_STEP_TRANSFORM = 'transform'
_STEP_ALS = 'als'

MAPPINGS_USER_TO_ID = 0x1
MAPPINGS_ITEM_TO_ID = 0x2

MAPPINGS_ID_TO_ITEM = 0x3
MAPPINGS_ID_TO_USER = 0x4


def train_als(*args, **kwargs):
    x_df, cols, hyperparameters = args

    col_item, col_user, col_values = cols
    shape = (len(x_df[col_item].unique()), len(x_df[col_user].unique()))
    groups = (x_df.groupby(col_item).ngroup(), x_df.groupby(col_user).ngroup())
    mappings = _mappings(x_df, cols, groups)

    pipe = Pipeline([
        (_STEP_TRANSFORM, MatrixLinearTransformer(cols, shape, groups)),
        (_STEP_ALS, ALSEstimator())
    ])

    hyperparameters = pipe_hyperparameters(hyperparameters, [_STEP_TRANSFORM, _STEP_ALS])

    pipe \
        .set_params(**hyperparameters) \
        .fit(x_df)

    return pipe.named_steps[_STEP_ALS], mappings


def _mappings(x_df: 'DataFrame', cols: tuple, groups: tuple):
    col_item, col_user, _ = cols
    item_group, user_group = groups

    id_to_item = {item_group[ind]: row[col_item] for ind, row in x_df.iterrows()}
    id_to_user = {user_group[ind]: row[col_user] for ind, row in x_df.iterrows()}

    item_to_id = dict(zip(id_to_item.values(), id_to_item.keys()))
    user_to_id = dict(zip(id_to_user.values(), id_to_user.keys()))

    return {
        MAPPINGS_ID_TO_ITEM: id_to_item,
        MAPPINGS_ID_TO_USER: id_to_user,
        MAPPINGS_ITEM_TO_ID: item_to_id,
        MAPPINGS_USER_TO_ID: user_to_id
    }
