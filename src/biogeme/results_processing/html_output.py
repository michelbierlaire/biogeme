"""
Generates an HTML output

Michel Bierlaire
Wed Oct 2 16:10:54 2024
"""

import logging
import os
from datetime import datetime

import numpy as np

import biogeme.version as version
from biogeme.exceptions import BiogemeError
from .estimation_results import (
    EstimateVarianceCovariance,
    EstimationResults,
    calc_p_value,
    calculates_correlation_matrix,
)
from ..parameters import Parameters

logger = logging.getLogger(__name__)


def format_real_number(value: float) -> str:
    """Format a real number to be included in the HTML table"""

    formatted_value = f'{value:.3g}'
    return formatted_value


def get_html_header(estimation_results: EstimationResults) -> str:
    """Prepare the header for the HTML file, containing comments and the
    version of Biogeme.

    :return: string containing the header.
    """
    html = ''
    html += '<html>\n'
    html += '<head>\n'
    html += '<script src="http://transp-or.epfl.ch/biogeme/sorttable.js">' '</script>\n'
    html += '<meta http-equiv="Content-Type" content="text/html; ' 'charset=utf-8" />\n'
    if estimation_results.raw_estimation_results is None:
        html += '<p>No estimation result is available.</p>'
        return html
    html += (
        f'<title>{estimation_results.raw_estimation_results.model_name} - Report from '
        f'biogeme {version.get_version()} '
        f'[{version.versionDate}]</title>\n'
    )
    html += (
        '<meta name="keywords" content="biogeme, discrete choice, ' 'random utility">\n'
    )
    html += (
        f'<meta name="description" content="Report from '
        f'biogeme {version.get_version()} '
        f'[{version.versionDate}]">\n'
    )
    html += '<meta name="author" content="{bv.author}">\n'
    html += '<style type=text/css>\n'
    html += '.biostyle\n'
    html += '	{font-size:10.0pt;\n'
    html += '	font-weight:400;\n'
    html += '	font-style:normal;\n'
    html += '	font-family:Courier;}\n'
    html += '.boundstyle\n'
    html += '	{font-size:10.0pt;\n'
    html += '	font-weight:400;\n'
    html += '	font-style:normal;\n'
    html += '	font-family:Courier;\n'
    html += '        color:red}\n'
    html += '</style>\n'
    html += '</head>\n'
    html += '<body bgcolor="#ffffff">\n'
    return html


def get_html_footer() -> str:
    """Prepare the footer for the HTML file, containing comments and the
    version of Biogeme.
    """
    html = '</body>\n</html>'
    return html


def get_html_preamble(estimation_results: EstimationResults, file_name: str) -> str:
    """Generates the first part of the HTML, with the preamble information.

    :param estimation_results: estimation results
    :param file_name: name of the HTML file (used only for reporting)
    :return: HTML code
    """
    now = datetime.now()
    html = version.get_html()
    html += f'<p>This file has automatically been generated on {now}</p>\n'
    if estimation_results.raw_estimation_results is None:
        html += '<p>No estimation result is available.</p>'
        return html
    html += '<table>\n'
    html += (
        f'<tr class=biostyle><td align=right>'
        f'<strong>Report file</strong>:	</td>'
        f'<td>{file_name}</td></tr>\n'
    )
    html += (
        f'<tr class=biostyle><td align=right>'
        f'<strong>Database name</strong>:	</td>'
        f'<td>{estimation_results.raw_estimation_results.data_name}</td></tr>\n'
    )
    html += '</table>\n'

    if not estimation_results.algorithm_has_converged:
        html += '<h2>Algorithm failed to converge</h2>\n'
        html += (
            '<p>It seems that the optimization algorithm did not converge. '
            'Therefore, the results below do not correspond to the maximum '
            'likelihood estimator. Check the specification of the model, '
            'or the criteria for convergence of the algorithm. </p>'
        )
    identification_threshold = Parameters().get_value('identification_threshold')
    try:
        if np.abs(estimation_results.smallest_eigenvalue) <= identification_threshold:
            html += '<h2>Warning: identification issue</h2>\n'
            html += (
                f'<p>The second derivatives matrix is close to singularity. '
                f'The smallest eigenvalue is '
                f'{np.abs(estimation_results.smallest_eigenvalue):.3g}. This warning is '
                f'triggered when it is smaller than the parameter '
                f'<code>identification_threshold</code>='
                f'{identification_threshold}.</p>'
                f'<p>Variables involved:'
            )
            html += '<table>'
            for i, ev in enumerate(estimation_results.smallest_eigenvector):
                if np.abs(ev) > identification_threshold:
                    html += (
                        f'<tr><td>{ev:.3g}</td>'
                        f'<td> *</td>'
                        f'<td> {estimation_results.beta_names[i]}</td></tr>\n'
                    )
            html += '</table>'
            html += '</p>\n'
    except BiogemeError:
        html += '<h2>Warning: second derivatives matrix not available.</h2>\n'
        html += '<p>The second derivatives matrix has not been calculated. The statistics requiring '
        html += 'it have been generated using the BHHH matrix or the bootstrap samples (if available).</p>'

    if estimation_results.user_notes is not None:
        # User notes
        html += (
            f'<blockquote style="border: 2px solid #666; '
            f'padding: 10px; background-color:'
            f' #ccc;">{estimation_results.user_notes}</blockquote>'
        )
    return html


def get_html_general_statistics(estimation_results: EstimationResults) -> str:
    """Get the general statistics coded in HTML

    :return: HTML code
    """

    html = '<table border="0">\n'
    statistics = estimation_results.get_general_statistics()
    for description, value in statistics.items():
        if value is not None:
            html += (
                f'<tr class=biostyle><td align=right >'
                f'<strong>{description}</strong>: </td> '
                f'<td>{value}</td></tr>\n'
            )
    for key, value in estimation_results.optimization_messages.items():
        if key in (
            'Relative projected gradient',
            'Relative change',
            'Relative gradient',
        ):
            html += (
                f'<tr class=biostyle><td align=right >'
                f'<strong>{key}</strong>: </td> '
                f'<td>{value:.7g}</td></tr>\n'
            )
        else:
            html += (
                f'<tr class=biostyle><td align=right >'
                f'<strong>{key}</strong>: </td> '
                f'<td>{value}</td></tr>\n'
            )

    html += '</table>\n'
    return html


def get_html_one_parameter(
    estimation_results: EstimationResults,
    parameter_index: int,
    variance_covariance_type: EstimateVarianceCovariance,
    parameter_number=None,
    parameter_name=None,
):
    """Generate the HTML code for one row of the table of the estimated parameters.

    :param estimation_results: estimation results.
    :param parameter_index: index of the parameter
    :param variance_covariance_type: type of variance-covariance estimate to be used.
    :param parameter_number: number of the parameter to report. If None, it is the index.
    :param parameter_name: name of the parameter to report. If None, taken from estimation results.
    :return: HTML code for the row
    """
    if parameter_index < 0 or parameter_index >= len(estimation_results.beta_names):
        error_msg = f'Invalid parameter index {parameter_index}. Valid range: 0- {len(estimation_results.beta_names)-1}'
        raise ValueError(error_msg)
    if parameter_number is None:
        parameter_number = parameter_index
    if parameter_name is None:
        parameter_name = estimation_results.beta_names[parameter_index]

    output = '<tr class=biostyle>'
    output += f'<td>{parameter_number}</td>'
    output += f'<td>{parameter_name}</td>'
    # Value
    value = estimation_results.get_parameter_value_from_index(
        parameter_index=parameter_index
    )
    output += f'<td>{format_real_number(value)}</td>'
    # std err
    std_err = estimation_results.get_parameter_std_err_from_index(
        parameter_index=parameter_index, estimate_var_covar=variance_covariance_type
    )
    output += f'<td>{format_real_number(std_err)}</td>'
    # t-test against 0
    t_test = estimation_results.get_parameter_t_test_from_index(
        parameter_index=parameter_index,
        estimate_var_covar=variance_covariance_type,
        target=0,
    )
    output += f'<td>{format_real_number(t_test)}</td>'
    # p-value against 0
    p_value = estimation_results.get_parameter_p_value_from_index(
        parameter_index=parameter_index,
        estimate_var_covar=variance_covariance_type,
        target=0,
    )
    output += f'<td>{format_real_number(p_value)}</td>'
    if estimation_results.is_bound_active(
        parameter_name=estimation_results.beta_names[parameter_index]
    ):
        output += '<td>Active bound</td>'
    output += '</tr>'
    return output


def get_html_estimated_parameters(
    estimation_results: EstimationResults,
    variance_covariance_type: EstimateVarianceCovariance | None = None,
    renaming_parameters: dict[str, str] | None = None,
    sort_by_name: bool = False,
) -> str:
    """Get the estimated parameters coded in HTML

    :param estimation_results: estimation results.
    :param variance_covariance_type: type of variance-covariance estimate to be used.
    :param renaming_parameters: a dict that suggests new names for some or all parameters.
    :param sort_by_name: if True, parameters are sorted alphabetically by name.
    :return: HTML code
    """
    if variance_covariance_type is None:
        variance_covariance_type = (
            estimation_results.get_default_variance_covariance_matrix()
        )
    if renaming_parameters is not None:
        # Verify that the renaming is well-defined.
        name_values = list(renaming_parameters.values())
        if len(name_values) != len(set(name_values)):
            warning_msg = (
                f'The new renaming assigns the same name for multiple parameters. It may not be the '
                f'desired action: {renaming_parameters}'
            )
            logger.warning(warning_msg)

    covar_header = str(variance_covariance_type)
    html = '<table border="1">\n'
    html += '<tr class=biostyle>'
    html += '<th>Id</th>'
    html += '<th>Name</th>'
    html += '<th>Value</th>'
    html += f'<th>{covar_header} std err.</th>'
    html += f'<th>{covar_header} t-stat.</th>'
    html += f'<th>{covar_header} p-value</th>'
    if estimation_results.is_any_bound_active():
        html += '<th></th>'
    html += '</tr>\n'

    rows = []
    for parameter_index, parameter_name in enumerate(estimation_results.beta_names):
        name = (
            renaming_parameters.get(parameter_name)
            if renaming_parameters is not None
            else parameter_name
        )
        row_html = (
            get_html_one_parameter(
                estimation_results=estimation_results,
                parameter_index=parameter_index,
                variance_covariance_type=variance_covariance_type,
                parameter_number=parameter_index,
                parameter_name=name,
            )
            + '\n'
        )
        rows.append((parameter_index, name, row_html))

    if sort_by_name:
        rows.sort(key=lambda x: x[1])

    for _, _, row_html in rows:
        html += row_html
    html += '</table>'
    return html


def get_html_one_pair_of_parameters(
    estimation_results: EstimationResults,
    first_parameter_index: int,
    second_parameter_index: int,
    variance_covariance_type: EstimateVarianceCovariance,
    first_parameter_name=None,
    second_parameter_name=None,
) -> str:
    """Generate one row of the HTML table of the correlation data for estimated parameters.

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

    covariance_matrix = estimation_results.get_variance_covariance_matrix(
        variance_covariance_type=variance_covariance_type
    )
    correlation_matrix = calculates_correlation_matrix(covariance=covariance_matrix)
    covariance = float(covariance_matrix[first_parameter_index, second_parameter_index])
    correlation = float(
        correlation_matrix[first_parameter_index, second_parameter_index]
    )
    t_test = estimation_results.calculate_test(
        first_parameter_index, second_parameter_index, covariance_matrix
    )
    p_value = calc_p_value(t_test)
    the_row = '<tr class=biostyle>'
    the_row += f'<td>{first_parameter_name}</td>'
    the_row += f'<td>{second_parameter_name}</td>'
    the_row += f'<td>{format_real_number(covariance)}</td>'
    the_row += f'<td>{format_real_number(correlation)}</td>'
    the_row += f'<td>{format_real_number(t_test)}</td>'
    the_row += f'<td>{format_real_number(p_value)}</td>'
    the_row += '</tr>'
    return the_row


def get_html_correlation_results(
    estimation_results: EstimationResults,
    variance_covariance_type: EstimateVarianceCovariance | None = None,
    involved_parameters: dict[str, str] | None = None,
) -> str:
    """Get the correlation results in an HTML format

    :param estimation_results: estimation results.
    :param variance_covariance_type: type of variance-covariance estimate to be used.
    :param involved_parameters: a dict that identifies the parameters to involve, as assign them with a name for the
        reporting.
    :return: HTML code
    """
    if variance_covariance_type is None:
        variance_covariance_type = (
            estimation_results.get_default_variance_covariance_matrix()
        )
    covar_header = str(variance_covariance_type)

    if involved_parameters is None:
        list_of_parameters = {
            index: name for index, name in enumerate(estimation_results.beta_names)
        }
    else:
        list_of_parameters = {
            estimation_results.get_parameter_index(orig_name): new_name
            for orig_name, new_name in involved_parameters.items()
        }

    html = '<table border="1">\n'
    html += '<tr class=biostyle>'
    html += '<th>Coefficient 1</th>'
    html += '<th>Coefficient 2</th>'
    html += f'<th>{covar_header} covariance</th>'
    html += f'<th>{covar_header} correlation</th>'
    html += f'<th>{covar_header} t-test</th>'
    html += f'<th>{covar_header} p-value</th>'
    html += '</tr>'
    for first_parameter_index, first_parameter_name in list_of_parameters.items():
        for second_parameter_index, second_parameter_name in list_of_parameters.items():
            if first_parameter_index > second_parameter_index:
                the_row = get_html_one_pair_of_parameters(
                    estimation_results=estimation_results,
                    first_parameter_index=first_parameter_index,
                    second_parameter_index=second_parameter_index,
                    variance_covariance_type=variance_covariance_type,
                    first_parameter_name=first_parameter_name,
                    second_parameter_name=second_parameter_name,
                )
                html += the_row + '\n'
    html += '</table>'
    return html


def get_html_condition_number(estimation_results: EstimationResults) -> str:
    """Report the smallest and largest eigenvalues, and the condition number.

    :param estimation_results: estimation results
    :return: HTML code
    """
    html = (
        f'<p>Smallest eigenvalue: '
        f'{estimation_results.smallest_eigenvalue:.6g}</p>\n'
    )
    html += (
        f'<p>Largest eigenvalue: ' f'{estimation_results.largest_eigenvalue:.6g}</p>\n'
    )
    html += f'<p>Condition number: ' f'{estimation_results.condition_number:.6g}</p>\n'
    return html


def generate_html_file(
    estimation_results: EstimationResults,
    filename: str,
    overwrite=False,
    variance_covariance_type: EstimateVarianceCovariance | None = None,
) -> None:
    """Generate an HTML file with the estimation results

    :param estimation_results: estimation results
    :param filename: name of the file
    :param overwrite: if True and the file exists, it is overwritten
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
        header = get_html_header(estimation_results=estimation_results)
        preamble = get_html_preamble(
            estimation_results=estimation_results, file_name=filename
        )
        general_statistics = get_html_general_statistics(
            estimation_results=estimation_results
        )
        parameters = get_html_estimated_parameters(
            estimation_results=estimation_results,
            sort_by_name=True,
            variance_covariance_type=variance_covariance_type,
        )
        correlation_results = get_html_correlation_results(
            estimation_results=estimation_results,
            variance_covariance_type=variance_covariance_type,
        )

        footer = get_html_footer()

        print(header, file=file)
        print(preamble, file=file)
        print('<h1>Estimation report</h1>', file=file)
        print(general_statistics, file=file)
        print('<h1>Estimated parameters</h1>', file=file)
        print(parameters, file=file)
        print('<h2>Correlation of coefficients</h2>', file=file)
        print(correlation_results, file=file)
        try:
            condition_number = get_html_condition_number(
                estimation_results=estimation_results
            )
            print(condition_number, file=file)
        except BiogemeError:
            ...
        print(footer, file=file)
    logger.info(f'File {filename} has been generated.')
