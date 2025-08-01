"""
This module provides utility functions for performing sampling operations
on pandas DataFrames, including standard bootstrapping and panel-based sampling.

Michel Bierlaire
Wed Mar 26 19:39:21 2025
"""

import pandas as pd
import numpy as np
from biogeme.exceptions import BiogemeError


def sample_with_replacement(df: pd.DataFrame, size: int | None = None) -> pd.DataFrame:
    if size is None:
        size = len(df)
    indices = np.random.randint(0, len(df), size=size)
    return df.iloc[indices].reset_index(drop=True)


def sample_panel_with_replacement(
    df: pd.DataFrame, individual_map: pd.DataFrame, size: int | None = None
) -> pd.DataFrame:
    """
    Draws a sample of individuals with replacement from a panel dataset.

    :param df: The input DataFrame representing the full dataset.
    :param individual_map: A DataFrame mapping each individual ID to (start, end) row indices.
    :param size: The number of individuals to sample. Defaults to the number of individuals in the map.
    :return: A new DataFrame with the sampled individuals' rows, with reset index.
    :raises BiogemeError: if the individual_map is missing or empty.
    """
    if individual_map is None or individual_map.empty:
        raise BiogemeError("Panel individual map is missing or empty.")
    if size is None:
        size = len(individual_map)
    sampled_rows = []
    sampled_ids = np.random.choice(individual_map.index, size=size, replace=True)
    for individual_id in sampled_ids:
        start_idx, end_idx = individual_map.loc[individual_id]
        rows = df.loc[start_idx:end_idx]
        sampled_rows.append(rows)
    return pd.concat(sampled_rows, ignore_index=True)


def split_validation_sets(
    df: pd.DataFrame, slices: int, group_column: str | None = None
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Splits a DataFrame into multiple (estimation, validation) pairs for cross-validation.

    :param df: The input DataFrame to split.
    :param slices: The number of folds (must be >= 2).
    :param group_column: Optional column name used to group rows (e.g., individual ID).
                         If provided, groups are kept together in folds.
    :return: A list of (estimation, validation) DataFrame tuples.
    :raises BiogemeError: if the number of slices is less than 2 or group column is not found.
    """
    if slices < 2:
        raise BiogemeError("Validation requires at least 2 slices.")
    if group_column is None:
        shuffled = df.sample(frac=1)
        folds = np.array_split(shuffled, slices)
    else:
        if group_column not in df.columns:
            raise BiogemeError(f"Grouping column '{group_column}' not found.")
        ids = df[group_column].unique()
        np.random.shuffle(ids)
        folds = [
            df[df[group_column].isin(group.tolist())]
            for group in np.array_split(ids, slices)
        ]
    estimation_sets = []
    validation_sets = []
    for i, validation in enumerate(folds):
        estimation = pd.concat(folds[:i] + folds[i + 1 :])
        estimation_sets.append(estimation.reset_index(drop=True))
        validation_sets.append(validation.reset_index(drop=True))
    return list(zip(estimation_sets, validation_sets))
