from itertools import islice

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRecommenderLayer
from .mappings import MappingsIdToValue, MappingsValueToId

_K = 25
_MISSING_VALUE = -999

_KWARG_FILTER_USERS = 'filter_users'
_KWARG_N = 'N'
_N_DEFAULT = 10


class CandidateRecommender(BaseRecommenderLayer):
    def __init__(self, users_factors: tuple, mappings_review: tuple, mappings_commits: tuple):
        self.users_factors_1, self.users_factors_2 = users_factors

        self.id_to_user_1 = MappingsIdToValue(mappings_review[0])
        self.user_to_id_1 = MappingsValueToId(mappings_review[1])

        self.id_to_user_2 = MappingsIdToValue(mappings_commits[0])
        self.user_to_id_2 = MappingsValueToId(mappings_commits[1])

    def recommend(self, items_factors: tuple, **kwargs):
        items_factors_1, items_factors_2 = items_factors
        filter_users = kwargs.get(_KWARG_FILTER_USERS, None)

        _filter_users = set(filter_users) \
            if filter_users is not None \
            else set()

        if np.all(np.isnan(items_factors_1)):
            return ([np.nan], [np.nan]), ([np.nan], [np.nan]), ([np.nan], [np.nan]), [np.nan]

        users_1 = self._recommend(items_factors_1, self.users_factors_1, _filter_users)
        scores_11 = self._users_scores(users_1, items_factors_1, self.users_factors_1)

        users_12 = self.user_to_id_2.transform(
            self.id_to_user_1.transform(users_1))
        users_common_1 = users_1[users_12.mask == False].reshape([-1])
        users_common_2 = users_12[users_12.mask == False].reshape([-1])

        if len(users_common_2):
            users_sim = self._similar_users(users_common_2, self.users_factors_2)
            users_sim_1 = self.user_to_id_1.transform(
                self.id_to_user_2.transform(users_sim))
            users_2 = users_sim[users_sim_1.mask == True]
            scores_22 = self._users_scores(users_2, items_factors_2, self.users_factors_2)
        else:
            users_2 = []
            scores_22 = []

        return (np.array(users_1, dtype='int', copy=False),
                np.array(users_2, dtype='int', copy=False)), \
               (np.array(users_common_1, dtype='int', copy=False),
                np.array(users_common_2, dtype='int', copy=False)), \
               (scores_11, scores_22), \
               items_factors_1[~np.isnan(items_factors_1).any(axis=1)],

    def _recommend(self, items_factors, users_factors, filter_users):
        ids = ~np.isnan(items_factors).any(axis=1)

        users_all = [self._recommend_users(users_factors, item_factors,
                                           weight=1, N=_K,
                                           filter_users=filter_users,
                                           without_score=True)
                     for item_factors in items_factors[ids]]

        users = np.unique(np.array(users_all, copy=False))

        return users

    def _recommend_users(self, users_factors: np.ndarray, item_factors: np.ndarray, weight: float, N: int,
                         filter_users: set, without_score=False):
        scores = users_factors.dot(item_factors) * weight

        count = N + len(filter_users)
        if count < len(scores):
            ids = np.argpartition(scores, -count)[-count:]
            best = sorted(zip(ids, scores[ids]), key=lambda x: -x[1])
        else:
            best = sorted(enumerate(scores), key=lambda x: -x[1])

        if without_score:
            best = list(islice((rec[0] for rec in best if rec[0] not in filter_users), N))
        else:
            best = list(islice((rec for rec in best if rec[0] not in filter_users), N))

        return np.asarray(best, dtype=object)

    def _users_scores(self, users, items_factors: np.ndarray, users_factors: np.ndarray):
        ids = ~np.isnan(items_factors).any(axis=1)
        scores = [items_factors[ids].dot(users_factors[user])
                  if not np.ma.is_masked(user)
                  else self._empty_recommendations((ids.sum(),))
                  for user in users]

        return np.array(scores)

    def _similar_users(self, users, users_factors: np.ndarray, K=10):
        users_factors_target = users_factors[users]
        sim_users = cosine_similarity(users_factors_target, users_factors)
        sim_users[:, users] = _MISSING_VALUE

        users_similar = []
        for _sim_users in sim_users:
            if K < len(_sim_users):
                ids = np.argpartition(_sim_users, -K)[-K:]
                best = list(zip(ids, _sim_users[ids]))
            else:
                best = list(enumerate(_sim_users))
            users_similar.extend([_id for _id, _ in best if _id != _MISSING_VALUE])

        return np.unique(users_similar)

    def _empty_recommendations(self, size):
        recommendations = np.empty(size)
        recommendations[:] = np.nan
        return recommendations


class RankingRecommender(BaseRecommenderLayer):
    def __init__(self, mappings, profiles):
        self.id_to_user_1 = MappingsIdToValue(mappings[0])
        self.id_to_user_2 = MappingsIdToValue(mappings[1])

        self.profiles_1 = profiles[0]
        self.profiles_2 = profiles[1]

    def recommend(self, X, **kwargs):
        N = kwargs.get(_KWARG_N, _N_DEFAULT)

        (_candidates_rev, _candidates_pdev), \
        (_candidates_rev_devrev, _candidates_dev_devrev), \
        (_scores_rev, _scores_dev), items_profiles = X
        if pd.isna(_candidates_rev).all():
            return self._empty_recommendations(N)

        sim_items = cosine_similarity(items_profiles)

        if not pd.isna(_candidates_pdev).any() and len(_candidates_pdev):
            # Scores
            _, ind_candidates_rev_devrev, _ = np.intersect1d(_candidates_rev, _candidates_rev_devrev,
                                                             return_indices=True)
            _scores_rev_devrev = _scores_rev[ind_candidates_rev_devrev]

            # Profiles
            _profiles_pdev = self.profiles_2[_candidates_pdev]
            _profiles_dev_devrev = self.profiles_2[_candidates_dev_devrev]

            # Similarities of development preferences
            _sim_dev_pdev_devrev = np.dot(_profiles_pdev, _profiles_dev_devrev.T)

            # Restore scores for plain developers
            _sim_dev_pdev_devrev_sum = np.absolute(_sim_dev_pdev_devrev).sum(axis=1).reshape([-1, 1])
            _scores_rev_pdev = np.dot(_sim_dev_pdev_devrev, _scores_rev_devrev) / _sim_dev_pdev_devrev_sum

            # Concatenate candidates
            _scores_rev_all = np.concatenate((_scores_rev, _scores_rev_pdev), axis=0)
            _users_all = np.concatenate((
                self.id_to_user_1.transform(_candidates_rev),
                self.id_to_user_2.transform(_candidates_pdev)))

            _scores_all = _scores_rev_all
        else:
            _scores_all = _scores_rev
            _users_all = self.id_to_user_1.transform(_candidates_rev)

        scores_weighted = [np.sum(sim_items * _scores, axis=1) / np.sum(sim_items, axis=1)
                           for _scores in _scores_all]
        _scores_all = np.sum(scores_weighted, axis=1)

        best = np.argsort(_scores_all)[::-1][:N]
        _y_pred = _users_all[best]

        return _y_pred

    def _empty_recommendations(self, size):
        recommendations = np.empty(size)
        recommendations[:] = np.nan
        return recommendations
