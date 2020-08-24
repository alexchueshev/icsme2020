import argparse
import logging as log

import config
import model
import layers
import utils

_ARG_PROJECTS = 'projects'
_ARG_PROJECTS_SHORT = 'prj'

_ARG_INI = 'ini'
_ARG_INI_SHORT = 'i'

_ARG_GRID_SEARCH = 'grid'
_ARG_GRID_SEARCH_SHORT = 'gs'

_ARG_BAYES_SEARCH = 'bayes'
_ARG_BAYES_SEARCH_SHORT = 'bs'

_ARG_TRAIN = 'train'
_ARG_TRAIN_SHORT = 't'

_ARG_BUILD = 'build'
_ARG_BUILD_SHORT = 'b'

_ARG_RECOMMEND = 'recommend'
_ARG_RECOMMEND_SHORT = 'r'

_ARG_METRICS = 'metrics'
_ARG_METRICS_SHORT = 'm'

log.basicConfig(level=log.INFO)


def _grid_search(cfg: dict):
    parameters, hyperparameters = cfg['parameters'], cfg['hyperparameters']
    cfg_dataset, cfg_results = cfg['dataset'], cfg['results']

    x_df = utils.read_csv(cfg_dataset['path'], usecols=cfg_dataset['cols'])
    gs = model.grid_search_als(x_df, cfg_dataset['cols'], hyperparameters, **parameters)
    model.save_tuning_results(gs, cfg_results)


def _bayes_search(cfg: dict):
    parameters, hyperparameters = cfg['parameters'], cfg['hyperparameters']
    cfg_dataset, cfg_results = cfg['dataset'], cfg['results']

    x_df = utils.read_csv(cfg_dataset['path'], usecols=cfg_dataset['cols'])
    bs = model.bayes_search_als(x_df, cfg_dataset['cols'], hyperparameters, **parameters)
    model.save_tuning_results(bs, cfg_results)


def _train(cfg: dict):
    hyperparameters, cfg_dataset = cfg['hyperparameters'], cfg['dataset']
    cfg_model, cfg_mappings = cfg['model'], cfg['mappings']

    x_df = utils.read_csv(cfg_dataset['path'], usecols=cfg_dataset['cols'])
    model_als, mappings = model.train_als(x_df, cfg_dataset['cols'], hyperparameters)

    if cfg_model.get('save', True):
        model.serialize(cfg_model['out'], model_als)
    if cfg_mappings.get('save', True):
        model.serialize(cfg_mappings['out'], mappings)


def _build(cfg: dict):
    def _deserialize_model(cfg_model: dict):
        model_als = model.deserialize(cfg_model['als'])
        mappings = model.deserialize(cfg_model['mappings'])
        return model_als, mappings

    cfg_model_reviews, cfg_model_commits = cfg['models']['reviews'], cfg['models']['commits']
    cfg_results = cfg['results']

    model_reviews_als, mappings_reviews = _deserialize_model(cfg_model_reviews)
    model_commits_als, mappings_commits = _deserialize_model(cfg_model_commits)

    model_recommender = model.build(
        layers.MappingsValueToIdPairFallback(
            (mappings_reviews[model.MAPPINGS_ITEM_TO_ID], mappings_commits[model.MAPPINGS_ITEM_TO_ID])
        ),
        layers.MappingsIdToEmbeddingPair(
            (model_reviews_als.item_factors, model_commits_als.item_factors)
        ),
        layers.CandidateRecommender(
            (model_reviews_als.user_factors, model_commits_als.user_factors),
            (mappings_reviews[model.MAPPINGS_ID_TO_USER], mappings_reviews[model.MAPPINGS_USER_TO_ID]),
            (mappings_commits[model.MAPPINGS_ID_TO_USER], mappings_commits[model.MAPPINGS_USER_TO_ID]),
        ),
        layers.RankingRecommender(
            (mappings_reviews[model.MAPPINGS_ID_TO_USER], mappings_commits[model.MAPPINGS_ID_TO_USER]),
            (model_reviews_als.user_factors, model_commits_als.user_factors),
        )
    )

    if cfg_results.get('save', True):
        model.serialize(cfg_results['out'], model_recommender)


def _recommend(cfg: dict):
    top_n = cfg['top_n']
    cfg_model, cfg_dataset = cfg['model'], cfg['dataset']
    cfg_results = cfg['results']

    _, col_files = cfg_dataset['cols']
    x_df = utils \
        .read_csv(cfg_dataset['path'], usecols=cfg_dataset['cols']) \
        .pipe(utils.to_list_of_strings, col=col_files)
    model_recommender = model.deserialize(cfg_model['path'])

    y_df = model.recommend(model_recommender, x_df, cfg_dataset['cols'], top_n=top_n)

    if cfg_results.get('save', True):
        utils.save_csv(cfg_results['out'], y_df)


def _metrics(cfg: dict):
    top_n = cfg['top_n']
    cfg_model, cfg_dataset = cfg['model'], cfg['dataset']
    cfg_results = cfg['results']

    _, col_files, col_reviewers = cfg_dataset['cols']
    x_df = utils \
        .read_csv(cfg_dataset['path'], usecols=cfg_dataset['cols']) \
        .pipe(utils.to_list_of_strings, col=col_files) \
        .pipe(utils.to_list_of_strings, col=col_reviewers)
    model_recommender = model.deserialize(cfg_model['path'])

    metrics_df = model.metrics(model_recommender, x_df, cfg_dataset['cols'], top_n=top_n)

    if cfg_results.get('save', True):
        utils.save_csv(cfg_results['out'], metrics_df, index=True)


_arguments = {
    (_ARG_INI_SHORT, _ARG_INI): dict(required=True, nargs='+'),
    (_ARG_PROJECTS_SHORT, _ARG_PROJECTS): dict(required=True, nargs='+'),
    (_ARG_GRID_SEARCH_SHORT, _ARG_GRID_SEARCH): dict(action='store_true'),
    (_ARG_BAYES_SEARCH_SHORT, _ARG_BAYES_SEARCH): dict(action='store_true'),
    (_ARG_TRAIN_SHORT, _ARG_TRAIN): dict(action='store_true'),
    (_ARG_BUILD_SHORT, _ARG_BUILD): dict(action='store_true'),
    (_ARG_RECOMMEND_SHORT, _ARG_RECOMMEND): dict(action='store_true'),
    (_ARG_METRICS_SHORT, _ARG_METRICS): dict(action='store_true'),
}

_funcs = {
    _ARG_GRID_SEARCH: _grid_search,
    _ARG_BAYES_SEARCH: _bayes_search,
    _ARG_TRAIN: _train,
    _ARG_BUILD: _build,
    _ARG_RECOMMEND: _recommend,
    _ARG_METRICS: _metrics,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for (arg_short, arg), param in _arguments.items():
        parser.add_argument(f'-{arg_short}', f'--{arg}', **param)
    args = parser.parse_args()

    cfg = config.load(*getattr(args, _ARG_INI))
    cfg_projects = cfg['projects']

    for project in args.projects:
        if project not in cfg_projects:
            log.warning(f'Project "{project}" not found')
            continue

        log.info(f'=== Project: {project} ===')
        cfg_project = cfg_projects[project]
        for _, arg in _arguments.keys():
            if getattr(args, arg) and arg in cfg_project:
                log.info(f'Command: {arg}')
                _funcs[arg](cfg_project[arg])
