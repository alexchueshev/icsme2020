from collections import Counter
from collections import deque
from pathlib import Path

import numpy as np

from .base import BaseTransformerLayer

_MISSED_ID = -999


class MappingsValueToId(BaseTransformerLayer):
    def __init__(self, mappings: dict):
        self.mappings = mappings

    def transform(self, X, y=None, **kwargs):
        ids = np.asarray([self.mappings[value] if value in self.mappings else _MISSED_ID for value in X])
        return np.ma.masked_where(ids == _MISSED_ID, ids, copy=False)


class MappingsValueToIdFallback(BaseTransformerLayer):
    def __init__(self, mappings: dict):
        self.mappings = mappings

    def transform(self, X, y=None, **kwargs):
        ids = []
        values = list(self.mappings.keys())
        for value in X:
            if value in self.mappings:
                ids.append(self.mappings[value])
            else:  # fallback
                fallback = self._lc_tokens(value, values)
                if len(fallback):
                    ids.extend([self.mappings[f] for f in fallback
                                if self.mappings[f] not in ids])
                else:
                    ids.append(_MISSED_ID)

        ids = np.array(ids)
        return np.ma.masked_where(ids == _MISSED_ID, ids, copy=False)

    @staticmethod
    def _lc_tokens(target: str, paths: list):
        parts = [Path(path).parts for path in paths]
        parts_target = Path(target).parts

        # longest common, longest prefix, index
        longest_common_tokens = deque([(_MISSED_ID, _MISSED_ID, _MISSED_ID)], maxlen=2)
        for ind, _parts in enumerate(parts):
            len_common_tokens = len(Counter(_parts) & Counter(parts_target))
            prefix = []
            for p1, p2 in zip(parts_target, _parts):
                if p1 != p2:
                    break
                prefix.append(p1)
            len_common_prefix = len(prefix)

            if len_common_tokens > 0 and len_common_tokens >= longest_common_tokens[0][0]\
                    and len_common_prefix >= longest_common_tokens[0][1]:
                longest_common_tokens.appendleft((len_common_tokens, len_common_prefix, ind))

        return [paths[ind] for _, _, ind in longest_common_tokens if ind != _MISSED_ID]


class MappingsValueToIdPair(BaseTransformerLayer):
    def __init__(self, mappings: tuple):
        self.value_to_id_1 = MappingsValueToId(mappings[0])
        self.value_to_id_2 = MappingsValueToId(mappings[1])

    def transform(self, X, y=None, **kwargs):
        ids_1 = self.value_to_id_1.transform(X[0])
        ids_2 = self.value_to_id_2.transform(X[1])

        return ids_1, ids_2


class MappingsValueToIdPairFallback(BaseTransformerLayer):
    def __init__(self, mappings: tuple):
        self.value_to_id_1 = MappingsValueToIdFallback(mappings[0])
        self.value_to_id_2 = MappingsValueToIdFallback(mappings[1])

    def transform(self, X, y=None, **kwargs):
        ids_1 = self.value_to_id_1.transform(X[0])
        ids_2 = self.value_to_id_2.transform(X[1])

        return ids_1, ids_2


class MappingsIdToValue(BaseTransformerLayer):
    def __init__(self, mappings: dict):
        self.mappings = mappings

    def transform(self, X, y=None, **kwargs):
        values = np.asarray([self.mappings[_id]
                             if _id in self.mappings
                             else np.nan
                             for _id in X])
        return values


class MappingsIdToEmbedding(BaseTransformerLayer):
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings

    def transform(self, X, y=None, **kwargs):
        embeddings = [self.embeddings[_id] if not np.ma.is_masked(_id)
                      else self._empty_embedding(self.embeddings.shape[1])
                      for _id in X]
        return np.asarray(embeddings)

    @staticmethod
    def _empty_embedding(size):
        embedding = np.empty(size)
        embedding[:] = np.nan
        return embedding


class MappingsIdToEmbeddingPair(BaseTransformerLayer):
    def __init__(self, embeddings: tuple):
        self.id_to_embedding_1 = MappingsIdToEmbedding(embeddings[0])
        self.id_to_embedding_2 = MappingsIdToEmbedding(embeddings[1])

    def transform(self, X, y=None, **kwargs):
        embeddings_1 = self.id_to_embedding_1.transform(X[0])
        embeddings_2 = self.id_to_embedding_2.transform(X[1])

        return embeddings_1, embeddings_2

