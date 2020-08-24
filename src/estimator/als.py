import logging

import numpy as np
from implicit.als import AlternatingLeastSquares
from sklearn.base import BaseEstimator

log = logging.getLogger("implicit")


class ALSEstimator(BaseEstimator, AlternatingLeastSquares):
    def __init__(self,
                 factors=100,
                 regularization=.01,
                 iterations=15,
                 filter_already_used=True,
                 calculate_training_loss=True):

        super().__init__(factors=factors, regularization=regularization, iterations=iterations,
                         calculate_training_loss=calculate_training_loss, use_gpu=False)

        self.x_train_nonzero = None
        self.filter_already_used = filter_already_used

    def fit(self, X, y=None):
        super().fit(X)

        if self.filter_already_used:
            self.x_train_nonzero = X.nonzero()

        return self

    def predict(self, X=None, y=None):
        predictions = np.dot(self.item_factors, self.user_factors.T)

        if self.filter_already_used:
            predictions[self.x_train_nonzero] = -99

        return predictions
