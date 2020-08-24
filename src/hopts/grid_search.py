from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .base_search import BaseSearchALS

_KWARG_TRANSFORM = 'transform_y'


class GridSearchALS(BaseSearchALS):

    def fit(self, *args, **kwargs):
        x_df, y_df, cols = args
        transform_y = kwargs.get(_KWARG_TRANSFORM, None)

        cv = self._make_cv(cols)
        scorer = self._make_scorer(transform_y=transform_y)

        gs = GridSearchCV(Pipeline(self.pipe),
                          self.hyperparameters,
                          cv=cv,
                          scoring=scorer,
                          return_train_score=self.with_train_score,
                          n_jobs=self.n_jobs,
                          verbose=self.verbose)
        gs.fit(x_df, y=y_df)

        return gs
