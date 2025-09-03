"""
Generates the estimation results in Pandas

Michel Bierlaire
Wed Oct 2 06:43:33 2024
"""

import logging

import numpy as np
import pandas as pd

from .estimation_results import (
    EstimateVarianceCovariance,
    EstimationResults,
    calc_p_value,
    calculates_correlation_matrix,
)
from ..exceptions import BiogemeError

logger = logging.getLogger(__name__)


def get_pandas_one_parameter(
    estimation_results: EstimationResults,
    parameter_index: int,
    variance_covariance_type: EstimateVarianceCovariance,
    parameter_number=None,
    parameter_name=None,
) -> dict[str, float | int | str]:
    """Generate one row of the Pandas table of the estimated parameters.

    :param estimation_results: estimation results.
    :param parameter_index: index of the parameter
    :param variance_covariance_type: type of variance-covariance estimate to be used.
    :param parameter_number: number of the parameter to report. If None, it is the index.
    :param parameter_name: name of the parameter to report. If None, taken from estimation results.
    :return: one row of the table
    """
    if parameter_index < 0 or parameter_index >= len(estimation_results.beta_names):
        error_msg = f'Invalid parameter index {parameter_index}. Valid range: 0- {len(estimation_results.beta_names)-1}'
        raise ValueError(error_msg)
    if parameter_number is None:
        parameter_number = parameter_index
    if parameter_name is None:
        parameter_name = estimation_results.beta_names[parameter_index]

    covar_header = covar_header = str(variance_covariance_type)

    value = estimation_results.get_parameter_value_from_index(
        parameter_index=parameter_index
    )
    std_err = (
        estimation_results.get_parameter_std_err_from_index(
            parameter_index=parameter_index, estimate_var_covar=variance_covariance_type
        )
        if estimation_results.are_derivatives_available
        else np.nan
    )
    t_test = (
        estimation_results.get_parameter_t_test_from_index(
            parameter_index=parameter_index,
            estimate_var_covar=variance_covariance_type,
            target=0,
        )
        if estimation_results.are_derivatives_available
        else np.nan
    )
    p_value = (
        estimation_results.get_parameter_p_value_from_index(
            parameter_index=parameter_index,
            estimate_var_covar=variance_covariance_type,
            target=0,
        )
        if estimation_results.are_derivatives_available
        else np.nan
    )

    the_row = {
        '#': parameter_number,
        'Name': parameter_name,
        'Value': value,
        f'{covar_header} std err.': std_err,
        f'{covar_header} t-stat.': t_test,
        f'{covar_header} p-value': p_value,
    }
    if estimation_results.is_any_bound_active():
        the_row['Active bound'] = estimation_results.is_bound_active(
            parameter_name=estimation_results.beta_names[parameter_index]
        )
    return the_row


def get_pandas_estimated_parameters(
    estimation_results: EstimationResults,
    variance_covariance_type: EstimateVarianceCovariance | None = None,
    renumbering_parameters: dict[int, int] | None = None,
    renaming_parameters: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Get the estimated parameters as a pandas data frame

    :param estimation_results: estimation results.
    :param variance_covariance_type: type of variance-covariance estimate to be used.
    :param renumbering_parameters: a dict that suggests new numbers for parameters
    :param renaming_parameters: a dict that suggests new names for some or all parameters.
    :param variance_covariance_type: select which type of variance-covariance matrix is used to generate the
        statistics. If None, the bootstrap one is used if available. If not available, the robust one.
    :return: a Pandas data frame
    """
    if variance_covariance_type is None:
        variance_covariance_type = (
            estimation_results.get_default_variance_covariance_matrix()
        )
    if (
        variance_covariance_type == EstimateVarianceCovariance.BOOTSTRAP
        and estimation_results.bootstrap_time is None
    ):
        logger.warning(
            f'No bootstrap data is available. The robust variance-covariance matrix is used instead.'
        )
        variance_covariance_type = EstimateVarianceCovariance.ROBUST
    if renumbering_parameters is not None:
        # Verify that the numbering is well defined
        number_values = list(renumbering_parameters.values())
        if len(number_values) != len(set(number_values)):
            error_msg = f'The new numbering cannot assign the same number to two different parameters.'
            raise BiogemeError(error_msg)

    if renaming_parameters is not None:
        # Verify that the renaming is well defined.
        name_values = list(renaming_parameters.values())
        if len(name_values) != len(set(name_values)):
            warning_msg = f'The new renaming assigns the same name for multiple parameters. It may not be the desired action.'
            logger.warning(warning_msg)
    all_rows = {}
    for parameter_index, parameter_name in enumerate(estimation_results.beta_names):
        new_number = (
            renumbering_parameters.get(parameter_index)
            if renumbering_parameters is not None
            else parameter_index
        )
        new_name = (
            renaming_parameters.get(parameter_name)
            if renaming_parameters is not None
            else estimation_results.beta_names[parameter_index]
        )

        the_row = get_pandas_one_parameter(
            estimation_results=estimation_results,
            parameter_index=parameter_index,
            variance_covariance_type=variance_covariance_type,
            parameter_number=new_number,
            parameter_name=new_name,
        )
        all_rows[new_number] = the_row

    list_of_all_rows = [all_rows[a_row_number] for a_row_number in sorted(all_rows)]
    the_frame = pd.DataFrame(list_of_all_rows)
    the_frame.set_index('#', inplace=True)
    the_frame.index.name = None

    return the_frame


def get_pandas_one_pair_of_parameters(
    estimation_results: EstimationResults,
    first_parameter_index: int,
    second_parameter_index: int,
    variance_covariance_type: EstimateVarianceCovariance,
    first_parameter_name=None,
    second_parameter_name=None,
) -> dict[str, float | int | str]:
    """Generate one row of the Pandas table of the correlation data for estimated parameters.

    :param estimation_results: estimation results.
    :param first_parameter_index: index of the first parameter
    :param second_parameter_index: index of the second parameter
    :param variance_covariance_type: type of variance-covariance estimate to be used.
    :param first_parameter_name: name of the parameter to report. If None, taken from estimation results.
    :param second_parameter_name: name of the parameter to report. If None, taken from estimation results.
    :return: one row of the table
    """
    if first_parameter_index < 0 or first_parameter_index >= len(
        estimation_results.beta_names
    ):
        error_msg = (
            f'Invalid parameter index {first_parameter_index}. Valid range: 0-'
            f' {len(estimation_results.beta_names)-1}'
        )
        raise ValueError(error_msg)
    if second_parameter_index < 0 or second_parameter_index >= len(
        estimation_results.beta_names
    ):
        error_msg = (
            f'Invalid parameter index {second_parameter_index}. Valid range: 0-'
            f' {len(estimation_results.beta_names)-1}'
        )
        raise ValueError(error_msg)

    if first_parameter_name is None:
        first_parameter_name = estimation_results.beta_names[first_parameter_index]
    if second_parameter_name is None:
        second_parameter_name = estimation_results.beta_names[second_parameter_index]

    covar_header = str(variance_covariance_type)

    covariance_matrix = estimation_results.get_variance_covariance_matrix(
        variance_covariance_type=variance_covariance_type
    )
    correlation_matrix = calculates_correlation_matrix(covariance=covariance_matrix)
    covariance = covariance_matrix[first_parameter_index, second_parameter_index]
    correlation = correlation_matrix[first_parameter_index, second_parameter_index]
    t_test = estimation_results.calculate_test(
        first_parameter_index, second_parameter_index, covariance_matrix
    )
    p_value = calc_p_value(t_test)
    the_row = {
        'First parameter': first_parameter_name,
        'Second parameter': second_parameter_name,
        f'{covar_header} covariance': covariance,
        f'{covar_header} correlation': correlation,
        f'{covar_header} t-stat.': t_test,
        f'{covar_header} p-value': p_value,
    }
    return the_row


def get_pandas_correlation_results(
    estimation_results: EstimationResults,
    variance_covariance_type: EstimateVarianceCovariance | None = None,
    involved_parameters: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Get the correlation results in a Pandas data frame

    :param estimation_results: estimation results.
    :param variance_covariance_type: type of variance-covariance estimate to be used.
    :param involved_parameters: a dict that identifies the parameters to involve, as assign them with a name for the
        reporting.
    :return: a Pandas data frame
    """
    if variance_covariance_type is None:
        variance_covariance_type = (
            estimation_results.get_default_variance_covariance_matrix()
        )
    if involved_parameters is None:
        list_of_parameters = {
            index: name for index, name in enumerate(estimation_results.beta_names)
        }
    else:
        list_of_parameters = {
            estimation_results.get_parameter_index(orig_name): new_name
            for orig_name, new_name in involved_parameters.items()
        }
    list_of_rows = []
    for first_parameter_index, first_parameter_name in list_of_parameters.items():
        for second_parameter_index, second_parameter_name in list_of_parameters.items():
            if first_parameter_index > second_parameter_index:
                the_row = get_pandas_one_pair_of_parameters(
                    estimation_results=estimation_results,
                    first_parameter_index=first_parameter_index,
                    second_parameter_index=second_parameter_index,
                    variance_covariance_type=variance_covariance_type,
                    first_parameter_name=first_parameter_name,
                    second_parameter_name=second_parameter_name,
                )
                list_of_rows.append(the_row)
    the_frame = pd.DataFrame(list_of_rows)
    return the_frame
