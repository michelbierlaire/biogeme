"""Audit the dataframe"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from biogeme.audit_tuple import AuditTuple
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Expression
from biogeme.tools import count_number_of_groups
from .container import Database


class ChosenAvailable(NamedTuple):
    chosen: int
    available: int


def check_availability_of_chosen_alt(
    database: Database, avail: dict[int:Expression], choice: Expression
) -> pd.Series:
    """Check if the chosen alternative is available for each entry
    in the database.

    :param database: object containing the data
    :param avail: list of expressions to evaluate the
                  availability conditions for each alternative.
    :param choice: expression for the chosen alternative.

    :return: numpy series of bool, long as the number of entries
             in the database, containing True is the chosen alternative is
             available, False otherwise.

    :raise BiogemeError: if the chosen alternative does not appear
        in the availability dict
    """
    from biogeme.jax_calculator import evaluate_expression

    choice_array = evaluate_expression(
        expression=choice, numerically_safe=False, database=database, use_jit=True
    )
    calculated_avail = {}
    for key, expression in avail.items():
        calculated_avail[key] = evaluate_expression(
            expression=expression,
            numerically_safe=False,
            database=database,
            use_jit=True,
        )
    try:
        avail_chosen = np.array(
            [calculated_avail[c][i] for i, c in enumerate(choice_array)]
        )
        return avail_chosen != 0
    except KeyError as exc:
        for c in choice_array:
            if c not in calculated_avail:
                err_msg = (
                    f'Chosen alternative {c} does not appear in '
                    f'availability dict: {calculated_avail.keys()}'
                )
                raise BiogemeError(err_msg) from exc


def choice_availability_statistics(
    database: Database, avail: dict[int:Expression], choice: Expression
) -> dict[int, ChosenAvailable]:
    """Calculates the number of times an alternative is chosen and available

    :param database: object containing the data
    :param avail: list of expressions to evaluate the
                  availability conditions for each alternative.
    :param choice: expression for the chosen alternative.

    :return: for each alternative, a tuple containing the number of time
        it is chosen, and the number of time it is available.

    :raise BiogemeError: if the database is empty.
    """
    from biogeme.jax_calculator import evaluate_expression

    choice_array = evaluate_expression(
        expression=choice, numerically_safe=False, database=database, use_jit=True
    )
    calculated_avail = {}
    for key, expression in avail.items():
        calculated_avail[key] = evaluate_expression(
            expression=expression,
            numerically_safe=False,
            database=database,
            use_jit=True,
        )
    unique = np.unique(choice_array, return_counts=True)
    choice_stat = {alt: int(unique[1][i]) for i, alt in enumerate(list(unique[0]))}
    avail_stat = {k: sum(a) for k, a in calculated_avail.items()}
    the_results = {
        alt: ChosenAvailable(chosen=c, available=avail_stat[alt])
        for alt, c in choice_stat.items()
    }
    return the_results


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
