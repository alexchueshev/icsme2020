import logging as log

from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from tempfile import TemporaryDirectory

from .base_search import BaseSearchALS

_KWARG_TRANSFORM = 'transform_y'

_KWARG_N_ITER = 'n_iter'
_N_ITER_DEFAULT = 20


class BayesSearchALS(BaseSearchALS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_iter = kwargs.get(_KWARG_N_ITER, _N_ITER_DEFAULT)

    def fit(self, *args, **kwargs):
        x_df, y_df, cols = args
        transform_y = kwargs.get(_KWARG_TRANSFORM, None)

        cv = self._make_cv(cols)
        scorer = self._make_scorer(transform_y=transform_y)

        with TemporaryDirectory(prefix='recsys_') as tmpdir:
            bs = BayesSearchCV(Pipeline(self.pipe, memory=tmpdir),
                               self.hyperparameters,
                               cv=cv,
                               scoring=scorer,
                               return_train_score=self.with_train_score,
                               n_jobs=self.n_jobs,
                               n_iter=self.n_iter,
                               verbose=self.verbose)
            log.info(f'Total iterations: {bs.total_iterations}')
            bs.fit(x_df, y=y_df)

        return bs
