"""Split data into validation and estimation samples"""

from typing import NamedTuple
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EstimationValidationIndices(NamedTuple):
    estimation: pd.Index
    validation: pd.Index


def split(
    dataframe: pd.DataFrame, slices: int, groups: str | None = None
) -> list[EstimationValidationIndices]:
    """
    Splits a DataFrame into multiple training and validation index sets for cross-validation.

    This function returns a list of `EstimationValidationIndices` named tuples, each containing
    the indices for an estimation (training) set and a validation set. If a grouping column
    is specified, the split ensures that all entries with the same group ID remain in the
    same fold.

    :param dataframe: The full dataset to split.
    :param slices: The number of folds/slices. Must be >= 2.
    :param groups: Optional name of the column containing group identifiers.
                   If provided, all rows with the same group ID are kept in the same fold.

    :return: A list of EstimationValidationIndices tuples containing index sets, one per fold.
    :raises ValueError: If `slices` is less than 2.
    """
    if slices < 2:
        raise ValueError(f'The number of slices is {slices}. It must be at least 2.')

    if groups is None:
        shuffled_data = dataframe.sample(frac=1)
        fold_data = np.array_split(shuffled_data.index, slices)
    else:
        group_ids = dataframe[groups].unique()
        np.random.shuffle(group_ids)
        grouped_ids = np.array_split(group_ids, slices)
        fold_data = [
            dataframe[dataframe[groups].isin(group)].index for group in grouped_ids
        ]

    return [
        EstimationValidationIndices(
            estimation=dataframe.index.difference(fold_data[i]),
            validation=fold_data[i],
        )
        for i in range(slices)
    ]
