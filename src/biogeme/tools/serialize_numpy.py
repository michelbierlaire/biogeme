"""
Tools to transform NaN in numpy arrays in order to be serialized

Michel Bierlaire
Mon May 19 2025, 11:57:38
"""

import numpy as np


def safe_serialize_array(
    array: np.ndarray,
) -> list[float | None] | list[list[float | None]]:
    """
    Convert a NumPy array with potential NaN values into a nested or flat list
    with `None` in place of `np.nan`, making it safe for YAML or JSON serialization.

    :param array: A NumPy array that may contain np.nan values.
    :return: A list (1D or 2D) with None in place of np.nan, suitable for serialization.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f'Input must be a NumPy array, not {type(array)}.')
    if array.ndim == 1:
        return [None if np.isnan(val) else float(val) for val in array]
    elif array.ndim == 2:
        return [[None if np.isnan(val) else float(val) for val in row] for row in array]
    else:
        raise ValueError("Only 1D and 2D arrays are supported.")


def safe_deserialize_array(
    serialized: list[float | None] | list[list[float | None]],
) -> list[float] | list[list[float]]:
    """
    Convert a flat or nested list with None values (as parsed from YAML or JSON)
    into a list with None replaced by float('nan').

    :param serialized: A list (1D or 2D) containing float or None values.
    :return: A list (1D or 2D) with None replaced by float('nan').
    """
    if not isinstance(serialized, list):
        raise TypeError("Input must be a list.")

    if all(isinstance(val, (float, int, type(None))) for val in serialized):
        # 1D case
        return [float('nan') if val is None else float(val) for val in serialized]
    elif all(
        isinstance(row, list)
        and all(isinstance(val, (float, int, type(None))) for val in row)
        for row in serialized
    ):
        # 2D case
        return [
            [float('nan') if val is None else float(val) for val in row]
            for row in serialized
        ]
    else:
        raise TypeError("Input must be a 1D or 2D list of numbers or None.")
