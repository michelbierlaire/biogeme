"""
MDCEV Utilities

This module provides helper functions for preparing and transforming
data specifically for MDCEV (Multiple Discrete Continuous Extreme Value) models.

Michel Bierlaire
Wed Mar 26 19:43:21 2025
"""

from typing import Iterable
import pandas as pd
from biogeme.database import Database


def mdcev_count(
    df: pd.DataFrame, list_of_columns: list[str], new_column: str
) -> pd.DataFrame:
    """
    Computes the number of non-zero entries across specified columns,
    corresponding to the number of goods consumed in MDCEV.

    :param df: DataFrame containing the MDCEV data.
    :param list_of_columns: Columns representing quantities of each good.
    :param new_column: Name of the output column to store the count.
    :return: Modified DataFrame with the count column added.
    """
    df[new_column] = df[list_of_columns].apply(lambda row: (row != 0).sum(), axis=1)
    return df


def mdcev_row_split(
    df: pd.DataFrame, a_range: Iterable[int] | None = None
) -> list[Database]:
    """
    Splits a DataFrame into a list of Database objects, one for each row,
    useful for row-level MDCEV processing.

    :param df: DataFrame to split.
    :param a_range: Optional subset of row indices to extract.
    :return: List of Database objects.
    """
    if a_range is None:
        a_range = range(len(df))
    else:
        max_index = len(df) - 1
        if any(i < 0 or i > max_index for i in a_range):
            raise IndexError("One or more indices in a_range are out of bounds.")

    return [Database(name=f'row_{i}', pandas_database=df.iloc[[i]]) for i in a_range]
