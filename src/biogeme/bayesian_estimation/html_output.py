"""Generate an HTML report from a Bayesian results summary.

This module generates HTML reports exclusively from
:class:`BayesianResultsSummary`, without requiring posterior draws or access to
NetCDF files. All diagnostics included in the report must therefore already be
stored in the summary object.

Michel Bierlaire
Mon Oct 20 2025, 19:27:28
"""

import html
import logging
import os
from datetime import datetime

from biogeme import version
from biogeme.exceptions import BiogemeError

from .bayesian_results import BayesianResults
from .bayesian_results_summary import BayesianResultsSummary, EstimatedBetaSummary

logger = logging.getLogger(__name__)


class EmptyListOfParameters(BiogemeError): ...


def format_real_number(value: float) -> str:
    """
    Format a real number for inclusion in an HTML table.

    :param value: Number to format.
    :return: A short string representation (currently using ``.3g`` formatting).
    """
    formatted_value = f'{value:.3g}'
    return formatted_value


def get_html_arviz_diagnostics(
    estimation_results: BayesianResultsSummary,
    html_filename: str,
    var_names: list[str] | None = None,
) -> str:
    """Generate an HTML snippet embedding pre-rendered diagnostic figures.

    The figures must already have been generated elsewhere and referenced in the
    summary object. This function only embeds them in the HTML report.

    :param estimation_results: Bayesian results summary.
    :param html_filename: Target HTML filename.
    :param var_names: Unused, kept for API compatibility.
    :return: HTML snippet containing embedded diagnostic figures, or an empty
        string if no figure references are available.
    """
    _ = html_filename
    _ = var_names

    figures = estimation_results.get_diagnostic_figure_references()
    if not figures:
        return ""

    explanations = {
        "trace": (
            "<strong>Trace</strong>: per-chain draws vs iteration and marginal "
            "density. Good: chains overlap, no trends or stickiness, rapid "
            "mixing. Suspicious: chains at different levels, strong drifts, "
            "long flat stretches, sudden jumps."
        ),
        "rank": (
            "<strong>Rank plot</strong>: rank-normalized samples by chain. "
            "Good: chains produce nearly uniform, overlapping ranks. "
            "Suspicious: U-shapes, spikes, or chains with very different rank "
            "distributions."
        ),
        "energy": (
            "<strong>Energy</strong>: HMC energy diagnostics and BFMI. Good: "
            "similar energy distributions across chains, no extreme tails; BFMI "
            "not flagged. Suspicious: clearly separated energy histograms across "
            "chains or very low BFMI."
        ),
        "autocorr": (
            "<strong>Autocorrelation</strong>: lag correlation within chains. "
            "Good: autocorrelation decays quickly toward 0 within tens of lags. "
            "Suspicious: long positive tails, high values at large lags, or "
            "periodic patterns."
        ),
    }

    titles = {
        "trace": "Trace",
        "rank": "Rank plot",
        "energy": "Energy",
        "autocorr": "Autocorrelation",
    }

    html_output = '<h1>Diagnostics</h1>\n'
    html_output += (
        '<p class="biostyle">The plots below summarize MCMC diagnostics. '
        'Look for well-mixed chains, agreement across chains, and weak lag '
        'dependence.</p>'
    )
    html_output += '<table border="0">\n'

    for key in ("trace", "rank", "energy", "autocorr"):
        figure_ref = figures.get(key)
        if figure_ref is None:
            continue
        html_output += (
            '<tr class=biostyle><td colspan=2>'
            f'<p>{explanations[key]}</p>'
            '</td></tr>\n'
        )
        html_output += (
            f'<tr class=biostyle><td><strong>{titles[key]}</strong></td>'
            f'<td><img style="max-width:840px;height:auto;display:block;" '
            f'src="{html.escape(figure_ref)}" alt="{html.escape(key)}"></td></tr>\n'
        )

    html_output += '</table>\n'
    return html_output


# --- ArviZ reproduction instructions section ---


def _html_code_block(code: str) -> str:
    """Wrap Python code in an HTML ``<pre><code>`` block.

    :param code: Python code to display.
    :return: HTML fragment containing the escaped code.
    """
    escaped = html.escape(code)
    return (
        '<pre style="background-color:#f4f4f4;padding:12px;border:1px solid #ccc;'
        'overflow-x:auto;"><code>'
        f'{escaped}'
        '</code></pre>'
    )


def get_html_arviz_reproduction_instructions(
    estimation_results: BayesianResultsSummary,
    html_filename: str,
    var_names: list[str] | None = None,
) -> str:
    """Generate HTML instructions describing the role of stored diagnostics.

    Since the report is generated from a lightweight summary without posterior
    draws, ArviZ diagnostics must have been generated beforehand and stored as
    figure references.

    :param estimation_results: Bayesian results summary.
    :param html_filename: Target HTML filename.
    :param var_names: Unused, kept for API compatibility.
    :return: HTML snippet explaining how diagnostics are handled.
    """
    _ = estimation_results
    _ = html_filename
    _ = var_names

    html_output = '<h1>Diagnostic figures</h1>\n'
    html_output += (
        '<p class="biostyle">This report was generated from a '
        '<code>BayesianResultsSummary</code> object. Therefore, posterior draws '
        'are not available here, and ArviZ figures cannot be regenerated at '
        'report-generation time. Any diagnostic figures shown above were '
        'generated beforehand and referenced in the summary object.</p>\n'
    )
    return html_output


def get_html_header(estimation_results: BayesianResultsSummary) -> str:
    """Prepare the header for the HTML file, containing comments and the
    version of Biogeme.

    :return: string containing the header.
    """
    html = ''
    html += '<html>\n'
    html += '<head>\n'
    html += '<script src="http://transp-or.epfl.ch/biogeme/sorttable.js">' '</script>\n'
    html += '<meta http-equiv="Content-Type" content="text/html; ' 'charset=utf-8" />\n'
    html += (
        f'<title>{estimation_results.model_name} - Report from '
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
    html += '<meta name="author" content="biogeme">\n'
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


def get_html_preamble(
    estimation_results: BayesianResultsSummary, file_name: str
) -> str:
    """Generates the first part of the HTML, with the preamble information.

    :param estimation_results: estimation results
    :param file_name: name of the HTML file (used only for reporting)
    :return: HTML code
    """
    now = datetime.now()
    html = version.get_html()
    html += f'<p>This file has automatically been generated on {now}</p>\n'
    html += '<table>\n'
    html += (
        f'<tr class=biostyle><td align=right>'
        f'<strong>Bayesian estimation report file</strong>:	</td>'
        f'<td>{file_name}</td></tr>\n'
    )
    html += (
        f'<tr class=biostyle><td align=right>'
        f'<strong>Database name</strong>:	</td>'
        f'<td>{estimation_results.data_name}</td></tr>\n'
    )
    html += '</table>\n'

    user_notes = estimation_results.get_user_notes()
    if user_notes:
        notes_html = '<br/>'.join(html.escape(note) for note in user_notes)
        html += (
            '<blockquote style="border: 2px solid #666; '
            'padding: 10px; background-color: #ccc;">'
            f'{notes_html}</blockquote>'
        )
    return html


def generate_one_row(description: str, value: str) -> str:
    """
    Generate a single HTML table row with a description/value pair.

    :param description: Label shown in the left column.
    :param value: Value shown in the right column.
    :return: HTML code for one ``<tr>`` row.
    """
    return (
        f'<tr class=biostyle><td align=right >'
        f'<strong>{description}</strong>: </td> '
        f'<td>{value}</td></tr>\n'
    )


def get_html_general_statistics(estimation_results: BayesianResultsSummary) -> str:
    """Get the general statistics coded in HTML

    :return: HTML code
    """

    html = '<table border="0">\n'
    for description, value in estimation_results.generate_general_information().items():
        html += generate_one_row(description=description, value=f'{value}')
    html += '</table>\n'
    return html


# --- Identification diagnostics section ---


def get_html_identification_diagnostics(
    estimation_results: BayesianResultsSummary,
) -> str:
    """Generate an HTML section reporting stored identification diagnostics.

    The diagnostics must already have been computed and stored in the summary
    object.

    :param estimation_results: Bayesian results summary.
    :return: HTML code for the identification diagnostics section, or an empty
        string if unavailable.
    """
    diag = estimation_results.get_identification_diagnostics_summary()
    if not isinstance(diag, dict) or not diag:
        return ""

    html_output = '<h1>Identification diagnostics</h1>\n'
    html_output += (
        '<p class="biostyle">This section reports precomputed checks for '
        '<em>non-identification</em> or <em>weak identification</em>. The '
        'diagnostics were computed previously from posterior draws and stored '
        'in the summary object.</p>\n'
    )

    flags = diag.get("flags")
    if flags:
        html_output += (
            '<p class="biostyle"><strong>Flags:</strong> '
            + ", ".join([str(f) for f in flags])
            + '</p>\n'
        )

    def _fmt(v: object) -> str:
        if isinstance(v, list) and v and isinstance(v[0], tuple) and len(v[0]) == 2:
            return ", ".join(
                f'{format_real_number(float(coef))}·{name}' for name, coef in v
            )
        if isinstance(v, dict):
            items = list(v.items())
            try:
                items.sort(key=lambda t: abs(float(t[1])), reverse=True)
            except (TypeError, ValueError):
                pass
            parts: list[str] = []
            for name, coef in items[:10]:
                try:
                    coef_str = format_real_number(float(coef))
                except (TypeError, ValueError):
                    coef_str = str(coef)
                parts.append(f'{coef_str}·{name}')
            return ", ".join(parts)
        if isinstance(v, float):
            return format_real_number(v)
        return str(v)

    def _render_kv_table(title: str, dct: dict) -> str:
        if not dct:
            return ""
        out = f'<h2>{title}</h2>\n'
        out += '<table border="0">\n'
        for key, value in dct.items():
            out += generate_one_row(description=str(key), value=_fmt(value))
        out += '</table>\n'
        return out

    html_output += _render_kv_table(
        'Posterior covariance diagnostics', diag.get('posterior', {})
    )
    html_output += _render_kv_table(
        'Prior covariance diagnostics', diag.get('prior', {})
    )

    per_param = diag.get('per_parameter')
    if isinstance(per_param, list) and per_param and isinstance(per_param[0], dict):
        html_output += '<h2>Per-parameter prior/posterior dispersion</h2>\n'
        html_output += (
            '<p class="biostyle">The table below compares posterior and prior '
            'standard deviations when prior draws were available.</p>\n'
        )
        cols = list(per_param[0].keys())
        html_output += '<table border="1">\n'
        html_output += '<tr class=biostyle>'
        for col in cols:
            html_output += f'<th>{col}</th>'
        html_output += '</tr>\n'
        for row in per_param:
            html_output += '<tr class=biostyle>'
            for col in cols:
                html_output += f'<td>{row.get(col, "")}</td>'
            html_output += '</tr>\n'
        html_output += '</table>\n'

    return html_output


def get_html_one_parameter(
    estimation_results: BayesianResultsSummary,
    parameter_number: int,
    parameter_name: str,
    display_name: str,
) -> str:
    """Generate the HTML code for one row of the table of the estimated parameters.

    :param estimation_results: estimation results.
    :param parameter_number: index of the parameter
    :param parameter_name: real key of the parameter in the stored summary.
    :param display_name: name to display in the table.
    :return: HTML code for the row
    """
    estimated_value = estimation_results.parameters.get(parameter_name)
    if estimated_value is None:
        return ''
    output = '<tr class=biostyle>'
    output += f'<td>{parameter_number}</td>'
    output += f'<td>{display_name}</td>'
    # : str
    # estimate: float
    # std_err: float
    # z_value: float | None
    # p_value: float | None
    # hdi_low: float | None
    # hdi_high: float | None

    # Value (mean)
    value = estimated_value.mean
    output += f'<td>{format_real_number(value)}</td>'
    # Value (median)
    value = estimated_value.median
    output += f'<td>{format_real_number(value)}</td>'
    # Value (mode)
    value = estimated_value.mode
    output += f'<td>{format_real_number(value)}</td>'
    # std err
    std_err = estimated_value.std_err
    output += f'<td>{format_real_number(std_err)}</td>'
    # t-test against 0
    t_test = estimated_value.z_value
    output += f'<td>{format_real_number(t_test)}</td>'
    # p-value against 0
    p_value = estimated_value.p_value
    output += f'<td>{format_real_number(p_value)}</td>'
    # hdi low
    hdi_low = estimated_value.hdi_low
    output += f'<td>{format_real_number(hdi_low)}</td>'
    # hdi high
    hdi_high = estimated_value.hdi_high
    output += f'<td>{format_real_number(hdi_high)}</td>'
    # rhat
    rhat = estimated_value.rhat
    output += f'<td>{format_real_number(rhat)}</td>'
    # Effective sample size
    ess_bulk = estimated_value.effective_sample_size_bulk
    output += f'<td>{format_real_number(ess_bulk)}</td>'
    ess_tail = estimated_value.effective_sample_size_tail
    output += f'<td>{format_real_number(ess_tail)}</td>'
    output += '</tr>'
    return output


def get_html_estimated_parameters(
    estimation_results: BayesianResultsSummary,
    estimated_parameters: bool = True,
    renaming_parameters: dict[str, str] | None = None,
    sort_by_name: bool = False,
) -> str:
    """Get the estimated parameters coded in HTML

    :param estimation_results: estimation results.
    :param estimated_parameters: if True, only the estimated parameters are generated. If False, only the other variables are generated.
    :param renaming_parameters: a dict that suggests new names for some or all parameters.
    :param sort_by_name: if True, parameters are sorted alphabetically by name.

    :return: HTML code
    """
    if renaming_parameters is not None:
        # Verify that the renaming is well-defined.
        name_values = list(renaming_parameters.values())
        if len(name_values) != len(set(name_values)):
            warning_msg = (
                f'The new renaming assigns the same name for multiple parameters. It may not be the '
                f'desired action: {renaming_parameters}'
            )
            logger.warning(warning_msg)

    html = '<table border="1">\n'
    html += '<tr class=biostyle>'
    html += '<th>Id</th>'
    html += '<th>Name</th>'
    html += '<th>Value (mean)</th>'
    html += '<th>Value (median)</th>'
    html += '<th>Value (mode)</th>'
    html += '<th>std err.</th>'
    html += '<th>z-value</th>'
    html += '<th>p-value</th>'
    html += '<th>HDI low</th>'
    html += '<th>HDI high</th>'
    html += '<th>R hat</th>'
    html += '<th>ESS (bulk)</th>'
    html += '<th>ESS (tail)</th>'
    html += '</tr>\n'

    to_be_reported = (
        estimation_results.parameter_estimates()
        if estimated_parameters
        else estimation_results.other_variables()
    )

    rows = []
    for parameter_index, parameter_name in enumerate(to_be_reported):
        name = (
            renaming_parameters.get(parameter_name)
            if renaming_parameters is not None
            else parameter_name
        )
        row_html = (
            get_html_one_parameter(
                estimation_results=estimation_results,
                parameter_number=parameter_index,
                parameter_name=parameter_name,
                display_name=name,
            )
            + '\n'
        )
        rows.append((parameter_index, name, row_html))

    if not rows:
        error_msg = 'Empty list of parameters'
        raise EmptyListOfParameters(error_msg)

    if sort_by_name:
        rows.sort(key=lambda x: x[1])

    for _, _, row_html in rows:
        html += row_html
    html += '</table>\n'

    if estimated_parameters:
        html += '<table border="0">\n'
        for column, doc in EstimatedBetaSummary.documentation.items():
            html += '<tr class=biostyle>'
            html += f'<td>{column}:</td>'
            html += f'<td>{doc}</td>'
            html += '</tr>\n'
        html += '</table>\n'
    return html


def generate_html_simulated_data(estimation_results: BayesianResultsSummary) -> str:
    simulated_data = estimation_results.report_stored_variables()
    return simulated_data.to_html(
        index=False,
        border=0,
        classes=["table", "table-striped", "table-sm"],
        justify="left",
    )


def generate_html_file(
    estimation_results: BayesianResultsSummary,
    filename: str,
    overwrite: bool = False,
) -> None:
    """Generate an HTML file with the estimation results

    :param estimation_results: estimation results
    :param filename: name of the file
    :param overwrite: if True and the file exists, it is overwritten
    """
    if not overwrite and os.path.exists(filename):
        raise FileExistsError(f"The file '{filename}' already exists.")

    with open(filename, 'w') as file:
        header = get_html_header(estimation_results=estimation_results)
        print(header, file=file)

        preamble = get_html_preamble(
            estimation_results=estimation_results, file_name=filename
        )
        print(preamble, file=file)
        print('<h1>Bayesian estimation report</h1>', file=file)
        general_statistics = get_html_general_statistics(
            estimation_results=estimation_results
        )
        print(general_statistics, file=file)
        try:
            parameters = get_html_estimated_parameters(
                estimation_results=estimation_results,
                estimated_parameters=True,
                sort_by_name=True,
            )
            print('<h1>Estimated parameters</h1>', file=file)
            print(parameters, file=file)
        except EmptyListOfParameters:
            logger.warning('No parameter to report.')

        identification_section = get_html_identification_diagnostics(
            estimation_results=estimation_results
        )
        if identification_section:
            print(identification_section, file=file)

        print('<h1>Simulated quantities</h1>', file=file)
        simulated_quantities = generate_html_simulated_data(
            estimation_results=estimation_results
        )
        print(simulated_quantities, file=file)

        diagnostics = get_html_arviz_diagnostics(
            estimation_results=estimation_results,
            html_filename=filename,
        )
        if diagnostics:
            print(diagnostics, file=file)

        instructions = get_html_arviz_reproduction_instructions(
            estimation_results=estimation_results,
            html_filename=filename,
        )
        if instructions:
            print(instructions, file=file)

        footer = get_html_footer()
        print(footer, file=file)
    logger.info(f'File {filename} has been generated.')


def generate_html_file_from_results(
    estimation_results: BayesianResults,
    filename: str,
    overwrite: bool = False,
) -> None:
    """Generate an HTML report from full Bayesian results.

    The full results are first converted into a
    :class:`BayesianResultsSummary`, and the HTML report is then generated from
    that summary object.

    :param estimation_results: Full Bayesian estimation results.
    :param filename: Name of the HTML file.
    :param overwrite: If True and the file exists, it is overwritten.
    """
    summary = estimation_results.to_summary()
    generate_html_file(summary, filename, overwrite=overwrite)
