import re
from functools import reduce

import yaml


def _merge1(d1, d2):
    """return new merged dict of dicts"""
    for k, v in d1.items():  # in Python 2, use .iteritems()!
        if k in d2:
            d2[k] = _merge1(v, d2[k])
    d3 = d1.copy()
    d3.update(d2)
    return d3


def _merge2(d1, d2):
    """update first dict with second recursively"""
    for k, v in d1.items():  # in Python 2, use .iteritems()!
        if k in d2:
            d2[k] = _merge2(v, d2[k])
    d1.update(d2)
    return d1


def load(*files: str):
    cfg = []

    loader = yaml.FullLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    for file in files:
        with open(file, mode='r') as config:
            cfg.append(yaml.load(config, Loader=loader))

    return reduce(_merge1, cfg)
