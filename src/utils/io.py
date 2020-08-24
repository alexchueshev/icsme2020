import json
from pathlib import Path

import pickle
import numpy as np
import pandas as pd


class __Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)

        return json.JSONEncoder.default(self, obj)


def save_json(filename: str, data: dict or list, mode: str = 'w'):
    _create_not_exist(filename)

    # TODO add appending strategy

    with open(filename, mode='w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, cls=__Encoder)


def save_csv(filename: str, df: pd.DataFrame, **kwargs):
    _create_not_exist(filename)

    parameters = {
        'index': False,
        'sep': '|'
    }
    parameters.update(kwargs)

    df.to_csv(filename, **parameters)


def save_pickle(filename: str, model: any):
    _create_not_exist(filename)

    with open(filename, mode='wb') as f:
        pickle.dump(model, f)


def load_pickle(filename):
    with open(filename, mode='rb') as f:
        model = pickle.load(f)

    return model


def read_csv(filename: str, **kwargs):
    return pd.read_csv(filename, sep='|', **kwargs)


def _create_not_exist(filename: str):
    path = Path(filename)

    if not Path(path.parent).is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
