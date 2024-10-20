"""
Generates a LaTeX output

Michel Bierlaire
Mon Sep 30 17:50:30 2024
"""

import logging
from datetime import datetime

import numpy as np

from biogeme.version import get_version, versionDate, get_latex
from .estimation_results import (
    EstimationResults,
    EstimateVarianceCovariance,
)
from .. import version
from ..exceptions import BiogemeError
from ..parameters import Parameters

logger = logging.getLogger(__name__)

PARAMETERS_TABLE_HEADER = r"""
\begin{tabular}{rlr@{.}lr@{.}lr@{.}lr@{.}l}
          &              &   \multicolumn{2}{l}{}         & \multicolumn{2}{l}{__VARCOVAR__}  &  \multicolumn{4}{l}{}  \\
Parameter &              &   \multicolumn{2}{l}{Coeff.}   & \multicolumn{2}{l}{Asympt.}       & \multicolumn{4}{l}{}   \\
number    &  Description &   \multicolumn{2}{l}{estimate} & \multicolumn{2}{l}{std. error}    & \multicolumn{2}{l}{$t$-stat}  &  \multicolumn{2}{l}{$p$-value} \\
\hline
"""

PARAMETERS_TABLE_HEADER_ACTIVE = r"""
\begin{tabular}{rlr@{.}lr@{.}lr@{.}lr@{.}ll}
          &              &   \multicolumn{2}{l}{}         & \multicolumn{2}{l}{__VARCOVAR__}  &  \multicolumn{4}{l}{} & \\
Parameter &              &   \multicolumn{2}{l}{Coeff.}   & \multicolumn{2}{l}{Asympt.}       & \multicolumn{4}{l}{}  & \\
number    &  Description &   \multicolumn{2}{l}{estimate} & \multicolumn{2}{l}{std. error}    & \multicolumn{2}{l}{
$t$-stat}  &  \multicolumn{2}{l}{$p$-value} & \\
\hline
"""

PARAMETERS_TABLE_FOOTER = r"""
\end{tabular}
"""


def add_trailing_zero(formatted_number: str) -> str:
    """If the formatted number does not contain a period, we add at the end a period and a zero.

    :param formatted_number: number already formatted
    :return: process strings
    """
    if not formatted_number:
        return '0.0'
    if '.' in formatted_number:
        if formatted_number[-1] == '.':
            return f'{formatted_number}0'
        return formatted_number
    return f'{formatted_number}.0'


def format_real_number(value: float) -> str:
    """Format a real number to be included in the LaTeX table"""

    formatted_value = f'{value:.3g}'

    # If the scientific notation has been used, we need to cancel it.
    if 'e' in formatted_value:
        left, right = formatted_value.split('e')
        return f'{add_trailing_zero(left)}e{right}'
    if 'E' in formatted_value:
        left, right = formatted_value.split('E')
        return f'{add_trailing_zero(left)}E{right}'
    return add_trailing_zero(formatted_value)


def _get_latex_header(estimation_results: EstimationResults) -> str:
    """Prepare the header for the LaTeX file, containing comments and the
    version of Biogeme.

    :return: string containing the header.
    """
    header = ''
    header += '%% This file is designed to be included into a LaTeX document\n'
    header += '%% See http://www.latex-project.org for information about LaTeX\n'
    if estimation_results.raw_estimation_results is None:
        header += 'No estimation result is available.'
        return header
    header += (
        f'%% {estimation_results.model_name} - Report from '
        f'biogeme {get_version()} '
        f'[{versionDate}]\n'
    )

    header += get_latex()
    return header


def get_latex_preamble(estimation_results: EstimationResults, file_name: str) -> str:
    """Generates the first part of the LaTeX file, with the preamble information.

    :param estimation_results: estimation results
    :param file_name: name of the LaTeX file (used only for reporting)
    :return: HTML code
    """
    now = datetime.now()
    latex = version.get_version()
    latex += f'\n% This file has automatically been generated on {now}\n'
    if estimation_results.raw_estimation_results is None:
        latex += '% No estimation result is available.'
        return latex
    latex += f'% Report file: {file_name}\n'
    latex += f'% Database name: {estimation_results.raw_estimation_results.data_name}\n'

    if not estimation_results.algorithm_has_converged:
        latex += r'\section{Algorithm failed to converge.}\n'
        latex += (
            'It seems that the optimization algorithm did not converge. '
            'Therefore, the results below do not correspond to the maximum '
            'likelihood estimator. Check the specification of the model, '
            'or the criteria for convergence of the algorithm.'
        )
    identification_threshold = Parameters().get_value('identification_threshold')
    if np.abs(estimation_results.smallest_eigenvalue) <= identification_threshold:
        latex += r'\section{Warning: identification issue}\n'
        latex += (
            f'The second derivatives matrix is close to singularity. '
            f'The smallest eigenvalue is '
            f'{np.abs(estimation_results.smallest_eigenvalue):.3g}. This warning is '
            f'triggered when it is smaller than the parameter '
            f'\\texttt{{identification_threshold}}='
            f'{identification_threshold}.\n\n'
            f'Variables involved:'
        )
        latex += r'\begin{tabular}{l@{*}l}'
        for i, ev in enumerate(estimation_results.smallest_eigenvector):
            if np.abs(ev) > identification_threshold:
                latex += f'{ev:.3g}' f' & ' f'{estimation_results.beta_names[i]}\\\\ \n'
        latex += r'\end{tabular}'

    if estimation_results.user_notes is not None:
        # User notes
        latex += f'User notes: {estimation_results.user_notes}'
    return latex


def get_latex_general_statistics(estimation_results: EstimationResults) -> str:
    """Get the general statistics coded in LaTeX

    :return: LaTeX code
    """
    latex = '\n%% General statistics\n'
    latex += '\\section{General statistics}\n'
    statistics = estimation_results.get_general_statistics()
    latex += '\\begin{tabular}{ll}\n'
    for name, value in statistics.items():
        value = value.replace('_', '\\_')
        latex += f'{name} & {value} \\\\\n'
    for key, value in estimation_results.optimization_messages.items():
        if key in (
            'Relative projected gradient',
            'Relative change',
            'Relative gradient',
        ):
            latex += f'{key} & \\verb${value:.4g}$ \\\\\n'
        else:
            latex += f'{key} & \\verb${value}$ \\\\\n'
    latex += '\\end{tabular}\n'

    return latex


def get_latex_one_parameter(
    estimation_results: EstimationResults,
    parameter_index: int,
    variance_covariance_type: EstimateVarianceCovariance,
    parameter_number=None,
    parameter_name=None,
):
    """Generate the LaTeX code for one row of the table of the estimated parameters.

    :param estimation_results: estimation results.
    :param parameter_index: index of the parameter
    :param variance_covariance_type: type of variance-covariance estimate to be used.
    :param parameter_number: number of the parameter to report. If None, it is the index.
    :param parameter_name: name of the parameter to report. If None, taken from estimation results.
    :return: LaTeX code for the row
    """
    if parameter_index < 0 or parameter_index >= len(estimation_results.beta_names):
        error_msg = f'Invalid parameter index {parameter_index}. Valid range: 0- {len(estimation_results.beta_names)-1}'
        raise ValueError(error_msg)
    if parameter_number is None:
        parameter_number = parameter_index
    if parameter_name is None:
        parameter_name = estimation_results.beta_names[parameter_index]

    output = f'{parameter_number}'
    output += ' & '
    output += parameter_name.replace('_', r'\_')
    output += ' & '
    # Value
    value = estimation_results.get_parameter_value(parameter_index=parameter_index)
    output += format_real_number(value).replace('.', '&')
    output += ' & '
    # std err
    std_err = estimation_results.get_parameter_std_err(
        parameter_index=parameter_index, estimate_var_covar=variance_covariance_type
    )
    output += format_real_number(std_err).replace('.', '&')
    output += ' & '
    # t-test against 0
    t_test = estimation_results.get_parameter_t_test(
        parameter_index=parameter_index,
        estimate_var_covar=variance_covariance_type,
        target=0,
    )
    output += format_real_number(t_test).replace('.', '&')
    output += ' & '
    # p-value against 0
    p_value = estimation_results.get_parameter_p_value(
        parameter_index=parameter_index,
        estimate_var_covar=variance_covariance_type,
        target=0,
    )
    output += format_real_number(p_value).replace('.', '&')
    if estimation_results.is_bound_active(
        parameter_name=estimation_results.beta_names[parameter_index]
    ):
        output += '& Active bound'
    output += r' \\ '
    return output


def get_latex_estimated_parameters(
    estimation_results: EstimationResults,
    variance_covariance_type: EstimateVarianceCovariance = EstimateVarianceCovariance.ROBUST,
    renumbering_parameters: dict[int, int] | None = None,
    renaming_parameters: dict[str, str] | None = None,
) -> str:
    """Get the estimated parameters coded in LaTeX

    :param estimation_results: estimation results.
    :param variance_covariance_type: type of variance-covariance estimate to be used.
    :param renumbering_parameters: a dict that suggests new numbers for parameters
    :param renaming_parameters: a dict that suggests new names for some or all parameters.
    :return: LaTeX code
    """
    if renumbering_parameters is not None:
        # Verify that the numbering is well defined
        number_values = list(renumbering_parameters.values())
        if len(number_values) != len(set(number_values)):
            error_msg = f'The new numbering cannot assign the same number to two different parameters: {renumbering_parameters}'
            raise BiogemeError(error_msg)

    if renaming_parameters is not None:
        # Verify that the renaming is well defined.
        name_values = list(renaming_parameters.values())
        if len(name_values) != len(set(name_values)):
            warning_msg = f'The new renaming assigns the same name for multiple parameters. It may not be the desired action: {renaming_parameters}'
            logger.warning(warning_msg)

    covar_header = {
        EstimateVarianceCovariance.RAO_CRAMER: 'Rao-Cramer',
        EstimateVarianceCovariance.ROBUST: 'Robust',
        EstimateVarianceCovariance.BOOTSTRAP: 'Bootstrap',
    }
    the_header = (
        PARAMETERS_TABLE_HEADER_ACTIVE
        if estimation_results.is_any_bound_active()
        else PARAMETERS_TABLE_HEADER
    )
    output = the_header.replace('__VARCOVAR__', covar_header[variance_covariance_type])
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

        the_row = (
            get_latex_one_parameter(
                estimation_results=estimation_results,
                parameter_index=parameter_index,
                variance_covariance_type=variance_covariance_type,
                parameter_number=new_number,
                parameter_name=new_name,
            )
            + '\n'
        )
        all_rows[new_number] = the_row

    for a_row_number in sorted(all_rows):
        output += all_rows[a_row_number]
    output += PARAMETERS_TABLE_FOOTER
    return output
