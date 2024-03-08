"""
Obtain sampling weights using local-sensitivity hashing

:author: Nicola Ortelli
:date: Fri Aug 11 18:25:39 2023

"""

import numpy as np
import pandas as pd


def get_lsh_weights(
    df: pd.DataFrame, w: float, a: np.ndarray, max_weight: int | None
) -> np.ndarray:
    """Compute weights using Locality-Sensitive Hashing (LSH) on input data.

    This function applies LSH to the input data frame, generating weights
    based on bucketing of the data. It also provides an option to limit
    the maximum weight assigned to a group of data points.

    :param df: The input data frame containing the data to compute weights for.
        The DataFrame should have at least one target column and one
        weight column.
    :type df: pandas DataFrame

    :param w: The width of the LSH buckets.
    :type w: float

    :param a: The LSH hash functions as a 2D array. Each row of this array
        represents an LSH hash function.
    :type a: numpy.ndarray

    :param max_weight: The maximum weight allowed for a group of data points. If not
        provided, no maximum weight constraint is applied.
    :type max_weight: int, optional

    :return: An array of weights corresponding to the input data frame.
    :rtype: numpy.ndarray
    """

    # Normalize the explanatory variables
    df_expla = df.drop(
        columns=['target', 'weight']
    )  # need to drop the intercept column...
    df_expla_norm = (df_expla - df_expla.min()) / (df_expla.max() - df_expla.min())

    # hashing into buckets according to LSH
    b = np.random.rand(a.shape[1]) * w
    buckets = np.floor((df_expla_norm.dot(a) + b) / w).astype(int)

    # if buckets depend on the target, uncomment this line...
    # buckets['target'] = df['target']

    # saving names of colummns storing buckets
    groupby_cols = list(buckets.columns)

    # Randomize the order of buckets to avoid bias
    buckets = buckets.sample(frac=1).reset_index()

    # adding a column that guarantees max_weight is never exceeded
    if max_weight:
        group_counts = buckets.groupby(groupby_cols, sort=False)[0].transform(
            'cumcount'
        )

        buckets['sieve'] = (group_counts / max_weight).astype(int)

        groupby_cols.append('sieve')

    # adding a column to store weights
    buckets['weight'] = 1

    # preparing aggregation dictionary for final grouping
    agg_dict = {'index': 'first', 'weight': 'sum'}

    # final grouping
    selected = buckets.groupby(groupby_cols, sort=False).agg(agg_dict)

    # building vector of weights
    weights = np.zeros(len(df))
    np.put(weights, selected['index'], selected['weight'])

    return weights
