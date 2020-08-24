from typing import TYPE_CHECKING
from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import coo_matrix

if TYPE_CHECKING:
    from pandas import DataFrame


class _MatrixTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, dims: tuple, shape: tuple, groups: tuple):
        self.dims = dims
        self.shape = shape
        self.groups = groups

        self.rows = None
        self.cols = None
        self.values = None

    def fit(self, X: 'DataFrame', y=None):
        row, col, value = self.dims
        ngroups_row, ngroups_col = self.groups

        indices = np.array([[ngroups_row[ind], ngroups_col[ind]]
                            for ind, _ in X.iterrows()])

        self.rows = indices[:, 0]
        self.cols = indices[:, 1]
        self.values = self._transform(X[value].values)

        return self

    def transform(self, X: 'DataFrame', y=None):
        return coo_matrix((self.values, (self.rows, self.cols)), shape=self.shape) \
            .tocsr()

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)

    def _indices(self, ngroups, X):
        return [ngroups[ind] for ind, _ in X.iterrows()]

    @abstractmethod
    def _transform(self, values):
        pass


class MatrixTransformer(_MatrixTransformer):
    def __init__(self, dims: tuple, shape: tuple, groups: tuple):
        super().__init__(dims, shape, groups)

    def _transform(self, values):
        return values


class MatrixLogTransformer(_MatrixTransformer):
    def __init__(self, dims: tuple, shape: tuple, groups: tuple,
                 alpha: float = 25, eps: float = 0.01):
        super().__init__(dims, shape, groups)

        self.alpha = alpha
        self.eps = eps

    def _transform(self, values):
        return 1 + self.alpha * np.log(1 + values / self.eps)


class MatrixLinearTransformer(_MatrixTransformer):
    def __init__(self, dims: tuple, shape: tuple, groups: tuple, alpha: float = 25):
        super().__init__(dims, shape, groups)

        self.alpha = alpha

    def _transform(self, values):
        return 1 + self.alpha * values
