def pipe_hyperparameters(hyperparameters: dict, steps: list):
    params = {}
    for step in steps:
        params.update({f'{step}__{param}': val for param, val in hyperparameters[step].items()})

    return params
