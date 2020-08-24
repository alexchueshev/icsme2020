import numpy as np

from layers import BaseRecommenderLayer
from layers import BaseTransformerLayer


class _Model:
    def __init__(self, *args):
        self.pipe = list(args)

    @property
    def layers(self):
        return len(self.pipe)

    def add_layers(self, *args):
        self.pipe.extend(args)
        return self

    def recommend(self, X, **kwargs):
        x = np.copy(X)
        for fn in self.pipe:
            if isinstance(fn, BaseTransformerLayer):
                x = fn.transform(x, **kwargs)
            elif isinstance(fn, BaseRecommenderLayer):
                x = fn.recommend(x, **kwargs)
        return x


def build(*args) -> _Model:
    return _Model(*args)
