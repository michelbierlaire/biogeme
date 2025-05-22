"""Audit the dataframe"""

import numpy as np
import pandas as pd

from biogeme.audit_tuple import AuditTuple
from biogeme.tools import count_number_of_groups


def audit_dataframe(data: pd.DataFrame) -> AuditTuple:
    """
    Performs a series of checks and reports warnings and errors for a pandas DataFrame.

    :param data: The DataFrame to audit.
    :return: the list of errors.
    """
    list_of_warnings = []
    list_of_errors = []

    for col, dtype in data.dtypes.items():
        if not np.issubdtype(dtype, np.number):
            list_of_errors.append(
                f'Column {col} in the database contains non-numeric type: {dtype}'
            )

    if data.isnull().values.any():
        nan_locations = data.isnull()
        rows_with_nan = data.index[nan_locations.any(axis=1)].tolist()
        cols_with_nan = data.columns[nan_locations.any(axis=0)].tolist()
        list_of_errors.append(
            f"The database contains NaN value(s).\n"
            f"Columns with NaN: {cols_with_nan}\n"
            f"Rows with NaN: {rows_with_nan}\n"
            f"Use database.dataframe.isna() to inspect further."
        )

    return AuditTuple(errors=list_of_errors, warnings=list_of_warnings)


def audit_panel_dataframe(
    data: pd.DataFrame, id_column: str
) -> tuple[list[str], list[str]]:
    """
    Performs panel-specific checks on a pandas DataFrame, ensuring entries for
    the same individual are contiguous.

    :param data: The DataFrame to audit.
    :param id_column: The name of the column identifying individuals.
    :return: A tuple (list_of_errors, list_of_warnings).
    """
    list_of_errors = []
    list_of_warnings = []

    if id_column not in data.columns:
        list_of_errors.append(
            f"The column '{id_column:d}' is missing from the dataset."
        )
        return list_of_errors, list_of_warnings

    original_groups = count_number_of_groups(data, id_column)
    sorted_data = data.sort_values(by=id_column).reset_index(drop=True)
    sorted_groups = count_number_of_groups(sorted_data, id_column)

    if original_groups != sorted_groups:
        list_of_errors.append(
            f"The data must be sorted so that entries for the same individual "
            f"are contiguous. Found {original_groups} original groups, "
            f"but {sorted_groups} after sorting."
        )

    return list_of_errors, list_of_warnings
