from typing import TYPE_CHECKING

from utils import save_pickle, load_pickle
from utils import save_json

if TYPE_CHECKING:
    from sklearn.model_selection._search import BaseSearchCV

_KEY_GRID = 'grid'
_KEY_MEAN_SCORE = 'mean_score'
_KEY_STD_SCORE = 'std_score'
_KEY_BEST_MEAN_SCORE = 'best_mean_score'
_KEY_BEST_ESTIMATOR = 'best_estimator'


def serialize(path: str, data):
    save_pickle(path, data)


def deserialize(path: str, format=None):
    return load_pickle(path)


def save_tuning_results(bs: 'BaseSearchCV', parameters: dict):
    def _save_grid(bs: 'BaseSearchCV', values: dict):
        save_json(values['out'], bs.cv_results_['params'], mode='w')

    def _save_mean_score(bs: 'BaseSearchCV', values: dict):
        output = {'mean_test_score': bs.cv_results_['mean_test_score']}
        save_json(values['out'], output, mode='w')

    def _save_std_score(bs: 'BaseSearchCV', values: dict):
        output = {'std_test_score': bs.cv_results_['std_test_score']}
        save_json(values['out'], output, mode='w')

    def _save_best_mean_score(bs: 'BaseSearchCV', values: dict):
        output = {
            'best_params': bs.best_params_,
            'best_score': bs.best_score_,
            'best_index': bs.best_index_,
        }
        save_json(values['out'], output, mode='w')

    def _save_best_estimator(bs: 'BaseSearchCV', values: dict):
        save_pickle(values['out'], bs.best_estimator_)

    _funcs = {
        _KEY_GRID: _save_grid,
        _KEY_MEAN_SCORE: _save_mean_score,
        _KEY_STD_SCORE: _save_std_score,
        _KEY_BEST_MEAN_SCORE: _save_best_mean_score,
        _KEY_BEST_ESTIMATOR: _save_best_estimator,
    }

    for key, values in parameters.items():
        if key in _funcs and values.get('save', True):
            _funcs[key](bs, values)
