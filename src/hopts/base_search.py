from abc import abstractmethod

from sklearn.metrics import make_scorer
from pandas import DataFrame

from selection import ShuffleSplitInGroups

_KWARG_N_SPLITS = 'n_splits'
_N_SPLITS_DEFAULT = 2

_KWARG_MIN_SAMPLES = 'min_samples'
_MIN_SAMPLES_DEFAULT = 5

_KWARG_TEST_SIZE = 'test_size'
_TEST_SIZE_DEFAULT = .1

_KWARG_N_JOBS = 'n_jobs'
_N_JOBS_DEFAULT = -1

_KWARG_VERBOSE = 'verbose'
_VERBOSE_DEFAULT = 1

_KWARG_WITH_TRAIN_SCORE = 'with_train_score'
_WITH_TRAIN_SCORE_DEFAULT = False

_KWARG_RANDOM_STATE = 'random_state'
_RANDOM_STATE_DEFAULT = None

_KWARG_TRANSFORM = 'transform_y'
_TRANSFORM_DEFAULT = None


class BaseSearchALS:
    def __init__(self, *args, **kwargs):
        _pipe, self.hyperparameters, self.metric = args

        self.pipe = list(_pipe)

        self.n_splits = kwargs.get(_KWARG_N_SPLITS, _N_SPLITS_DEFAULT)
        self.min_samples = kwargs.get(_KWARG_MIN_SAMPLES, _MIN_SAMPLES_DEFAULT)
        self.test_size = kwargs.get(_KWARG_TEST_SIZE, _TEST_SIZE_DEFAULT)
        self.n_jobs = kwargs.get(_KWARG_N_JOBS, _N_JOBS_DEFAULT)
        self.verbose = kwargs.get(_KWARG_VERBOSE, _VERBOSE_DEFAULT)
        self.with_train_score = kwargs.get(_KWARG_WITH_TRAIN_SCORE, _WITH_TRAIN_SCORE_DEFAULT)
        self.random_state = kwargs.get(_KWARG_RANDOM_STATE, _RANDOM_STATE_DEFAULT)

    def _make_cv(self, cols):
        col_item, _, _ = cols
        cv = ShuffleSplitInGroups(by=col_item,
                                  n_splits=self.n_splits,
                                  min_samples=self.min_samples,
                                  test_size=self.test_size,
                                  random_state=self.random_state)
        return cv

    def _make_scorer(self, transform_y=None):
        def _wrap_scorer(y_true, y_pred, **kwargs):
            transform_y = kwargs.get(_KWARG_TRANSFORM, _TRANSFORM_DEFAULT)
            if isinstance(y_true, DataFrame) and transform_y:
                y_true = transform_y(y_true)

            return self.metric(y_true, y_pred)

        scorer = make_scorer(_wrap_scorer,
                             greater_is_better=False,
                             transform_y=transform_y)

        return scorer

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass
