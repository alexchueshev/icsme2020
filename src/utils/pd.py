from typing import TYPE_CHECKING
from ast import literal_eval

if TYPE_CHECKING:
    from pandas import DataFrame


def to_list_of_strings(df: 'DataFrame', col: str):
    rows = []
    for values in df[col]:
        values_list = literal_eval(values)
        rows.append([str(r) for r in values_list])
    df[col] = rows

    return df
