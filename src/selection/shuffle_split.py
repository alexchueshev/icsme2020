import numpy as np
import multiprocessing as mp
from sklearn.model_selection._split import BaseShuffleSplit
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.model_selection._split import check_random_state


class ShuffleSplitInGroups(BaseShuffleSplit):
    def __init__(self, by: str, min_samples=5, n_splits=10, test_size="default",
                 train_size=None, random_state=None):
        super().__init__(n_splits, test_size, train_size, random_state)

        self.by = by
        self.min_samples = min_samples

    def _iter_indices(self, X, y=None, groups=None):
        x_grouped = X.groupby(self.by)
        rng = check_random_state(self.random_state)

        with mp.Pool(mp.cpu_count()) as pool:
            indices = [pool.apply(self._indices, args=(rng, x_grouped))
                       for _ in range(self.n_splits)]

        for ind_train, ind_test in indices:
            yield ind_train, ind_test

    def _indices(self, rng, x_grouped):
        shift = 0
        ind_test = []
        ind_train = []

        for _, group in x_grouped:
            n_samples = len(group)

            if n_samples < self.min_samples:
                ind_train.extend(np.arange(n_samples) + shift)
                shift += n_samples
                continue

            n_train, n_test = _validate_shuffle_split(n_samples, self.test_size, self.train_size)
            permutation = rng.permutation(n_samples)

            ind_test.extend(permutation[:n_test] + shift)
            ind_train.extend(permutation[n_test:(n_test + n_train)] + shift)

            shift += n_samples

        return np.array(ind_train), np.array(ind_test)
