from abc import abstractmethod


class BaseRecommenderLayer:
    @abstractmethod
    def recommend(self, *args, **kwargs):
        pass


class BaseTransformerLayer:
    @abstractmethod
    def transform(self, *args, **kwargs):
        pass
