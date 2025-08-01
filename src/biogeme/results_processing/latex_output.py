"""
Generates a LaTeX output

Michel Bierlaire
Mon Sep 30 17:50:30 2024
"""

import logging
import os
from collections import Counter
from datetime import datetime

import numpy as np

from biogeme.version import get_latex, get_version, versionDate
from .estimation_results import EstimateVarianceCovariance, EstimationResults
from ..exceptions import BiogemeError
from ..parameters import Parameters
from ..tools.ellipse import Ellipse

logger = logging.getLogger(__name__)

LATEX_FILE_HEADER = r"""
\documentclass{article}
\begin{document}
"""

LATEX_FILE_FOOTER = r"""
\end{document}
"""

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

COMPARAISON_PREAMBLE = r"""
\usepackage{longtable}
\usepackage{siunitx}
\sisetup{
  parse-numbers=false,      % Prevents automatic parsing (needed for parentheses & superscripts)
  detect-inline-weight=math,% Ensures proper formatting in tables
  tight-spacing=true        % Keeps spacing consistent
}
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


def format_with_period(value):
    """
    Formats a number using the '.3g' format while ensuring that a decimal point is always present.

    :param value: The number to format (int or float).
    :return: A string representation of the number in '.3g' format, always containing a period.

    Example:
    >>> format_with_period(1)
    '1.0'
    >>> format_with_period(1.23)
    '1.23'
    >>> format_with_period(1234)
    '1230'
    >>> format_with_period(1000000)
    '1e+06'
    >>> format_with_period(0.00456)
    '0.00456'
    """
    formatted = f'{value:.3g}'  # Use .3g format
    return formatted if '.' in formatted or 'e' in formatted else f'{formatted}.0'


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
    latex = ''
    latex += f'\n% This file has automatically been generated on {now}\n'
    if estimation_results.raw_estimation_results is None:
        latex += '% No estimation result is available.'
        return latex
    latex += f'% Report file: {file_name}\n'
    latex += f'% Database name: {estimation_results.raw_estimation_results.data_name}\n'

    latex += _get_latex_header(estimation_results=estimation_results)

    if not estimation_results.algorithm_has_converged:
        latex += r'\section{Algorithm failed to converge.}\n'
        latex += (
            'It seems that the optimization algorithm did not converge. '
            'Therefore, the results below do not correspond to the maximum '
            'likelihood estimator. Check the specification of the model, '
            'or the criteria for convergence of the algorithm.'
        )
    identification_threshold = Parameters().get_value('identification_threshold')
    try:
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
                    latex += (
                        f'{ev:.3g}' f' & ' f'{estimation_results.beta_names[i]}\\\\ \n'
                    )
            latex += r'\end{tabular}'
    except BiogemeError:
        latex += r'\section{Warning: second derivatives matrix not available}\n'
        latex += 'The second derivatives matrix has not been calculated. The statistics requiring it are not generated.'

    if estimation_results.user_notes is not None:
        # User notes
        latex += f'%% User notes: {estimation_results.user_notes}'
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
) -> str:
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
    value = estimation_results.get_parameter_value_from_index(
        parameter_index=parameter_index
    )
    output += format_real_number(value).replace('.', '&')
    output += ' & '
    # std err
    std_err = estimation_results.get_parameter_std_err_from_index(
        parameter_index=parameter_index, estimate_var_covar=variance_covariance_type
    )
    output += format_real_number(std_err).replace('.', '&')
    output += ' & '
    # t-test against 0
    t_test = estimation_results.get_parameter_t_test_from_index(
        parameter_index=parameter_index,
        estimate_var_covar=variance_covariance_type,
        target=0,
    )
    output += format_real_number(t_test).replace('.', '&')
    output += ' & '
    # p-value against 0
    p_value = estimation_results.get_parameter_p_value_from_index(
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


def rename_and_renumber(
    names: list[str],
    renumbering_parameters: dict[int, int] | None = None,
    renaming_parameters: dict[str, str] | None = None,
) -> dict[int, tuple[int, str]]:
    """Rename and renumber parameters according to instructions from the user

    :param names: list of existing names
    :param renumbering_parameters: dict mapping old numbers to new numbers.
    :param renaming_parameters: dict mapping old names to new names.
    :return: dict mapping the original parameter index with the new number and the new name.
    """
    if renumbering_parameters is None:
        renumbering_parameters = {}
    if renaming_parameters is None:
        renaming_parameters = {}
    else:
        # Verify that the renaming is well-defined.
        name_values = list(renaming_parameters.values())
        if len(name_values) != len(set(name_values)):
            warning_msg = f'The new renaming assigns the same name for multiple parameters. It may not be the desired action: {renaming_parameters}'
            logger.warning(warning_msg)

    updated_items = {}
    for old_key, name in enumerate(names):
        new_key = renumbering_parameters.get(
            old_key, old_key
        )  # Default to original key
        updated_items[old_key] = (new_key, renaming_parameters.get(name, name))

    list_of_new_keys = [val[0] for val in updated_items.values()]
    element_counts = Counter(list_of_new_keys)
    repeated_elements = {
        key: value for key, value in element_counts.items() if value > 1
    }
    if repeated_elements:
        error_msg = f'The following new indices appear more than once: {repeated_elements.keys()}'
        raise BiogemeError(error_msg)
    return updated_items


def get_latex_estimated_parameters(
    estimation_results: EstimationResults,
    variance_covariance_type: EstimateVarianceCovariance | None = None,
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
    if variance_covariance_type is None:
        variance_covariance_type = estimation_results.get_default_variance_covariance_matrix()
    if renumbering_parameters is not None:
        # Verify that the numbering is well-defined
        number_values = list(renumbering_parameters.values())
        if len(number_values) != len(estimation_results.beta_names):
            error_msg = (
                f'The new numbering involves {len(number_values)} values while there are '
                f'{len(estimation_results.beta_names)} parameters'
            )
            raise BiogemeError(error_msg)

        if len(number_values) != len(set(number_values)):
            error_msg = f'The new numbering cannot assign the same number to two different parameters: {renumbering_parameters}'
            raise BiogemeError(error_msg)

    the_header = (
        PARAMETERS_TABLE_HEADER_ACTIVE
        if estimation_results.is_any_bound_active()
        else PARAMETERS_TABLE_HEADER
    )
    output = the_header.replace('__VARCOVAR__', str(variance_covariance_type))

    renamed_parameters: dict[int, tuple[int, str]] = rename_and_renumber(
        names=estimation_results.beta_names,
        renumbering_parameters=renumbering_parameters,
        renaming_parameters=renaming_parameters,
    )
    all_rows = {}
    for old_index, user_defined in renamed_parameters.items():
        new_index, new_name = user_defined
        the_row = (
            get_latex_one_parameter(
                estimation_results=estimation_results,
                parameter_index=old_index,
                variance_covariance_type=variance_covariance_type,
                parameter_number=new_index,
                parameter_name=new_name,
            )
            + '\n'
        )
        all_rows[new_index] = the_row
    for a_row_number in sorted(all_rows):
        output += all_rows[a_row_number]
    output += PARAMETERS_TABLE_FOOTER
    return output


def get_sign_for_p_value(p_value: float, p_thresholds: list[tuple[float, str]]) -> str:
    """

    :param p_value: p-value to be treated
    :param p_thresholds: list of tuple establishing the coding convention for the p-value threshold. Assume that the
    list is composed of pairs (t_i, sign_i), and that the p-value of a parameter is p. Among all i such that p <=
    t_i, select index k associated with the minimum t_k. Then the p-value is coded using sign_k.
    :return: sign to be used
    """
    try:
        the_p_value = float(p_value)
    except (ValueError, TypeError) as e:
        raise TypeError(f'Cannot convert {p_value} to float') from e

    try:
        candidates_thresholds = [
            (float(t), str(sign)) for t, sign in p_thresholds if the_p_value <= float(t)
        ]
    except (ValueError, TypeError) as e:
        raise TypeError(f'Non float thresholds: {p_thresholds}') from e

    if not candidates_thresholds:
        return ''

    _, selected_sign = min(candidates_thresholds, key=lambda x: x[0])
    return selected_sign


def compare_parameters(
    estimation_results: dict[str, EstimationResults],
    p_thresholds: list[tuple[float, str]] | None = None,
    renumbering_parameters: dict[int, int] | None = None,
    renaming_parameters: dict[str, str] | None = None,
) -> str:
    """

    :param estimation_results: dict mapping model names with estimation results
    :param p_thresholds: list of tuple establishing the coding convention for the p-value threshold. Assume that the
    list is composed of pairs (t_i, sign_i), and that the p-value of a parameter is p. Among all i such that p <=
    t_i, select index k associated with the minimum t_k. Then the p-value is coded using sign_k.
    :param renumbering_parameters: a dict that suggests new numbers for parameters
    :param renaming_parameters: a dict that suggests new names for some or all parameters.
    :return: the LaTeX code of a tabular object compile the estimation results.
    """
    if p_thresholds is None:
        p_thresholds = [
            (0.01, r'\textsuperscript{***}'),
            (0.05, r'\textsuperscript{**}'),
            (0.1, r'\textsuperscript{*}'),
        ]

    latex_code = COMPARAISON_PREAMBLE
    # Opening the longtable environment
    number_of_columns = len(estimation_results)
    s_columns = 'S' * number_of_columns
    latex_code += r'\begin{longtable}{rl' + s_columns + r'}'
    latex_code += '\n'
    # First row
    latex_code += '& & '
    latex_code += ' & '.join(
        [
            fr'\multicolumn{{1}}{{c}}{{{model_name}}}'
            for model_name in estimation_results.keys()
        ]
    )
    latex_code += r' \\'
    latex_code += '\n'

    # Second row
    latex_code += ' & Parameter name & '
    latex_code += ' & '.join(
        [r' \multicolumn{1}{c}{Coef./(SE)}' for _ in estimation_results.keys()]
    )
    latex_code += r' \\'
    latex_code += '\n'

    # Separation line
    latex_code += r'\hline'
    latex_code += '\n'
    # Generate the list of all parameters
    all_beta_names: set[str] = set()
    for estimation_result in estimation_results.values():
        all_beta_names.update(estimation_result.beta_names)

    renamed_parameters: dict[int, tuple[int, str]] = rename_and_renumber(
        names=sorted(all_beta_names),
        renumbering_parameters=renumbering_parameters,
        renaming_parameters=renaming_parameters,
    )

    rows_1 = []
    rows_2 = []
    for new_id, name in renamed_parameters.values():
        row_1 = ''
        row_2 = ''
        row_1 += f'{new_id} &'
        row_2 += ' & '
        row_1 += name.replace('_', r'\_')
        for estimation_result in estimation_results.values():
            if name in estimation_result.beta_names:
                value = estimation_result.get_parameter_value(parameter_name=name)
                std_err = estimation_result.get_parameter_std_err(parameter_name=name)
                p_value = estimation_result.get_parameter_p_value(parameter_name=name)
                sign = get_sign_for_p_value(p_value, p_thresholds)
                formatted_value = format_with_period(value)
                row_1 += f'& {formatted_value}{sign} '
                row_2 += f'&({std_err:.3g})'
            else:
                row_1 += ' & '
                row_2 += ' & '
        row_1 += r' \\'
        row_2 += r' \\'
        row_1 += '\n'
        row_2 += '\n'
        rows_1.append(row_1)
        rows_2.append(row_2)

    for row_1, row_2 in zip(rows_1, rows_2):
        latex_code += row_1
        latex_code += row_2

    # Separation line
    latex_code += r'\hline'
    latex_code += '\n'

    # Observations
    latex_code += r'\multicolumn{2}{l}{Number of observations} &'
    latex_code += ' & '.join(
        [
            f'{estimation_results.sample_size}'
            for estimation_results in estimation_results.values()
        ]
    )
    latex_code += r' \\'
    latex_code += '\n'

    # Parameters
    latex_code += r'\multicolumn{2}{l}{Number of parameters} &'
    latex_code += ' & '.join(
        [
            f'{estimation_results.number_of_free_parameters}'
            for estimation_results in estimation_results.values()
        ]
    )
    latex_code += r' \\'
    latex_code += '\n'

    # AIC
    latex_code += r'\multicolumn{2}{l}{Akaike Information Criterion} &'
    latex_code += ' & '.join(
        [
            f'{estimation_results.akaike_information_criterion:.1f}'
            for estimation_results in estimation_results.values()
        ]
    )
    latex_code += r' \\'
    latex_code += '\n'

    # BIC
    latex_code += r'\multicolumn{2}{l}{Bayesian Information Criterion} &'
    latex_code += ' & '.join(
        [
            f'{estimation_results.bayesian_information_criterion:.1f}'
            for estimation_results in estimation_results.values()
        ]
    )
    latex_code += r' \\'
    latex_code += '\n'

    # Separation line
    latex_code += r'\hline'
    latex_code += '\n'

    latex_code += r'\multicolumn{4}{l}{\footnotesize Standard errors: '
    footnote = [f'{mark}: $p < {threshold}$' for (threshold, mark) in p_thresholds]
    latex_code += ', '.join(footnote)
    latex_code += r'}'
    latex_code += '\n'

    # Closing the longtable environment
    latex_code += r'\end{longtable}'
    latex_code += '\n'
    return latex_code


def draw_confidence_ellipse(
    ellipse: Ellipse,
    first_reporting_name: str,
    second_reporting_name: str,
) -> str:
    """Provides a Tikz picture of the confidence ellipsis for two parameters
    :param ellipse: ellipse to be drawn
    :param first_reporting_name: name to use in the reporting for the first parameter
    :param second_reporting_name: name to use in the reporting for the second parameter
    """
    # Define axis ranges
    xmin, xmax = (
        ellipse.center_x - 1.5 * ellipse.axis_one,
        ellipse.center_x + 1.5 * ellipse.axis_one,
    )
    ymin, ymax = (
        ellipse.center_y - 1.5 * ellipse.axis_two,
        ellipse.center_y + 1.5 * ellipse.axis_two,
    )

    # Generate TikZ code
    tikz_code = f"""
    \\begin{{tikzpicture}}
        \\begin{{axis}}[
            width=13cm, height=9cm,
            grid=major,
            axis x line=middle,
            axis y line=middle,
            xlabel={{{first_reporting_name}}},
            ylabel={{{second_reporting_name}}},
            xmin={xmin}, xmax={xmax},
            ymin={ymin}, ymax={ymax},
            axis line style={{thick}},
            ticks=both
        ]
            % Ellipse parameters
            \\pgfmathsetmacro{{\\centerx}}{{{ellipse.center_x:.4f}}}
            \\pgfmathsetmacro{{\\centery}}{{{ellipse.center_y:.4f}}}
            \\pgfmathsetmacro{{\\cosphi}}{{{ellipse.cos_phi:.6f}}}
            \\pgfmathsetmacro{{\\sinphi}}{{{ellipse.sin_phi:.6f}}}
            \\pgfmathsetmacro{{\\axisone}}{{{ellipse.axis_one:.6f}}}
            \\pgfmathsetmacro{{\\axistwo}}{{{ellipse.axis_two:.6f}}}

            % Center of the ellipse
            \\coordinate (center) at (\\centerx, \\centery);
            \\node at (center) {{$\\bullet$}};

            % Plot ellipse
            \\addplot[domain=0:360, samples=100, smooth, thick] (
                {{\\centerx + \\axisone * cos(x) * \\cosphi - \\axistwo * sin(x) * \\sinphi}},
                {{\\centery + \\axisone * cos(x) * \\sinphi + \\axistwo * sin(x) * \\cosphi}}
            );
        \\end{{axis}}
    \\end{{tikzpicture}}
    """
    return tikz_code


def generate_latex_file_content(
    estimation_results: EstimationResults,
    filename: str,
    variance_covariance_type: EstimateVarianceCovariance,
    include_begin_document=False,
) -> str:
    """
    Generate the full content of a LaTeX document summarizing the estimation results,
    and prepare it for export to a file.

    This function assembles a complete LaTeX document including the document header,
    general statistics, and the estimated parameter table. It is intended to be used
    to produce a full report from Biogeme estimation results.

    :param estimation_results: The estimation results object containing the model outputs.
    :param filename: The name of the LaTeX file, used only for documentation within the file.
    :param include_begin_document: if True, the LaTeX file can be directly compiled. If False, it must be included
        in another document.
    :param variance_covariance_type: select which type of variance-covariance matrix is used to generate the
        statistics.
    :return: None. The LaTeX content is generated as a string but not written to a file.
    """
    latex = ''
    if include_begin_document:
        latex += LATEX_FILE_HEADER
    latex += get_latex_preamble(
        estimation_results=estimation_results, file_name=filename
    )
    latex += get_latex_general_statistics(estimation_results=estimation_results)
    latex += get_latex_estimated_parameters(
        estimation_results=estimation_results,
        variance_covariance_type=variance_covariance_type,
    )
    if include_begin_document:
        latex += LATEX_FILE_FOOTER
    return latex


def generate_latex_file(
    estimation_results: EstimationResults,
    filename: str,
    include_begin_document=False,
    overwrite=False,
    variance_covariance_type: EstimateVarianceCovariance | None = None,
) -> None:
    """
    Generate and save a LaTeX document that summarizes the model estimation results.

    This function calls `generate_latex_file_content` to build the full LaTeX document,
    and writes it to a file. If the file already exists and `overwrite` is False,
    a `FileExistsError` is raised.

    :param estimation_results: Estimation results to be documented.
    :param include_begin_document: if True, the LaTeX file can be directly compiled. If False, it must be included
        in another document.
    :param filename: Path to the LaTeX file to be created.
    :param overwrite: If True, overwrite the file if it already exists. Defaults to False.
    :param variance_covariance_type: select which type of variance-covariance matrix is used to generate the
        statistics. If None, the bootstrap one is used if available. If not available, the robust one.
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

    if not overwrite and os.path.exists(filename):
        raise FileExistsError(f"The file '{filename}' already exists.")

    with open(filename, 'w') as file:
        content = generate_latex_file_content(
            estimation_results=estimation_results,
            filename=filename,
            include_begin_document=include_begin_document,
            variance_covariance_type=variance_covariance_type,
        )
        print(content, file=file)
    logger.info(f'File {filename} has been generated.')
