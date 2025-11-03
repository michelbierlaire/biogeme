"""
Generates an HTML output for Bayesian results

Michel Bierlaire
Mon Oct 20 2025, 19:27:28
"""

import logging
import os
from datetime import datetime
from enum import Enum

import arviz as az
import biogeme.version as version
import matplotlib
import matplotlib.pyplot as plt

from .bayesian_results import BayesianResults, EstimatedBeta

logger = logging.getLogger(__name__)
matplotlib.use("Agg")  # headless backend


def format_real_number(value: float) -> str:
    """Format a real number to be included in the HTML table"""
    formatted_value = f'{value:.3g}'
    return formatted_value


# --- Helper to save matplotlib figure safely ---
def _save_fig(fig, out_path: str) -> None:
    dir_ = os.path.dirname(out_path)
    if dir_ and not os.path.exists(dir_):
        os.makedirs(dir_, exist_ok=True)
    try:
        fig.tight_layout()
    except (RuntimeError, ValueError):
        pass
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


# --- FigureSize enum and size helper ---
class FigureSize(Enum):
    """Discrete size scale for diagnostic figures and embedded images."""

    NONE = "none"  # Do not render any diagnostic figures
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    VERY_LARGE = "very_large"


def _get_size_params(size: FigureSize) -> dict:
    """Return plotting and HTML sizing parameters for a given size level.

    Returns a dict with:
        - trace_figsize: tuple[float, float]
        - other_figsize: tuple[float, float]
        - max_width_px: int (for the <img> max-width CSS)
    """
    if size == FigureSize.SMALL:
        return {
            "trace_figsize": (7.0, 5.0),
            "other_figsize": (6.4, 4.0),
            "max_width_px": 520,
        }
    if size == FigureSize.MEDIUM:
        return {
            "trace_figsize": (9.0, 6.0),
            "other_figsize": (7.5, 5.0),
            "max_width_px": 680,
        }
    if size == FigureSize.LARGE:
        return {
            "trace_figsize": (11.0, 7.0),
            "other_figsize": (9.0, 5.8),
            "max_width_px": 840,
        }
    if size == FigureSize.VERY_LARGE:
        return {
            "trace_figsize": (13.0, 8.0),
            "other_figsize": (10.5, 6.6),
            "max_width_px": 980,
        }
    # Default / fallback (also used for NONE guard upstream)
    return {
        "trace_figsize": (9.0, 5.0),
        "other_figsize": (7.5, 5.0),
        "max_width_px": 680,
    }


def get_html_arviz_diagnostics(
    estimation_results: BayesianResults,
    html_filename: str,
    var_names: list[str] | None = None,
    figure_size: FigureSize = FigureSize.MEDIUM,
) -> str:
    """Generate ArviZ diagnostic figures (trace, rank, energy, autocorr) and
    return an HTML snippet embedding them as <img> tags.

    Figures are saved next to the HTML file in a 'figs' subfolder.
    """
    # Early exit if diagnostics should not be rendered
    if figure_size == FigureSize.NONE:
        return ""

    params = _get_size_params(figure_size)

    # Where to store figures
    base, _ = os.path.splitext(html_filename)
    figs_dir = base + "_figs"

    # Load posterior draws
    idata = estimation_results.idata

    # Default var_names: use keys from results.parameters up to max_vars
    if var_names is None:
        var_names = list(estimation_results.parameters.keys())

    html = '<h1>Diagnostics</h1>\n'
    # General note
    html += (
        '<p class="biostyle">The plots below summarize MCMC diagnostics. '
        'Look for well-mixed chains, agreement across chains, and weak lag dependence. </p>'
    )
    html += '<table border="0">\n'

    # 1) Trace plot
    try:
        fig = az.plot_trace(
            idata,
            var_names=var_names,
            compact=True,
            figsize=params["trace_figsize"],
        )
        # ArviZ usually returns a Figure for plot_trace; if not, fall back
        if not hasattr(fig, "savefig"):
            fig = plt.gcf()
        trace_path = os.path.join(figs_dir, "trace.png")
        _save_fig(fig, trace_path)
        # Explanation
        html += (
            '<tr class=biostyle><td colspan=2>'
            '<p><strong>Trace</strong>: per-chain draws vs iteration and marginal density. '
            'Good: chains overlap, no trends or stickiness, rapid mixing. '
            'Suspicious: chains at different levels, strong drifts, long flat stretches, sudden jumps.</p>'
            '</td></tr>\n'
        )
        html += (
            f'<tr class=biostyle><td><strong>Trace</strong></td>'
            f'<td><img style="max-width:{params["max_width_px"]}px;height:auto;display:block;" '
            f'src="{os.path.basename(figs_dir)}/trace.png" alt="trace"></td></tr>\n'
        )
    except (ValueError, TypeError) as e:
        logger.warning(f"Trace plot failed: {e}")

    # 2) Rank plot
    try:
        fig = az.plot_rank(idata, var_names=var_names, figsize=params["other_figsize"])
        if not hasattr(fig, "savefig"):
            fig = plt.gcf()
        rank_path = os.path.join(figs_dir, "rank.png")
        _save_fig(fig, rank_path)
        html += (
            '<tr class=biostyle><td colspan=2>'
            '<p><strong>Rank plot</strong>: rank-normalized samples by chain. '
            'Good: chains produce nearly uniform, overlapping ranks. '
            'Suspicious: U-shapes, spikes, or chains with very different rank distributions (poor mixing or non-stationarity).</p>'
            '</td></tr>\n'
        )
        html += (
            f'<tr class=biostyle><td><strong>Rank plot</strong></td>'
            f'<td><img style="max-width:{params["max_width_px"]}px;height:auto;display:block;" '
            f'src="{os.path.basename(figs_dir)}/rank.png" alt="rank"></td></tr>\n'
        )
    except (ValueError, TypeError) as e:
        logger.warning(f"Rank plot failed: {e}")

    # 3) Energy plot (NUTS diagnostics)
    try:
        energy_figsize = (
            params["other_figsize"][0] * 0.75,
            params["other_figsize"][1] * 0.75,
        )
        fig = az.plot_energy(idata, figsize=energy_figsize)
        if not hasattr(fig, "savefig"):
            fig = plt.gcf()
        energy_path = os.path.join(figs_dir, "energy.png")
        _save_fig(fig, energy_path)
        html += (
            '<tr class=biostyle><td colspan=2>'
            '<p><strong>Energy</strong>: HMC energy diagnostics and BFMI. '
            'Good: similar energy distributions across chains, no extreme tails; BFMI not flagged. '
            'Suspicious: clearly separated energy histograms across chains or very low BFMI (e.g., &lt; 0.3) indicating poor exploration.</p>'
            '</td></tr>\n'
        )
        html += (
            f'<tr class=biostyle><td><strong>Energy</strong></td>'
            f'<td><img style="max-width:{params["max_width_px"]}px;height:auto;display:block;" '
            f'src="{os.path.basename(figs_dir)}/energy.png" alt="energy"></td></tr>\n'
        )
    except (ValueError, TypeError) as e:
        logger.warning(f"Energy plot failed: {e}")

    # 4) Autocorrelation (limited lags)
    try:
        autocorr_figsize = (
            params["other_figsize"][0] * 1,
            params["other_figsize"][1] * 1.5,
        )
        fig = az.plot_autocorr(
            idata, var_names=var_names, max_lag=60, figsize=autocorr_figsize
        )
        if not hasattr(fig, "savefig"):
            fig = plt.gcf()
        acf_path = os.path.join(figs_dir, "autocorr.png")
        _save_fig(fig, acf_path)
        html += (
            '<tr class=biostyle><td colspan=2>'
            '<p><strong>Autocorrelation</strong>: lag correlation within chains. '
            'Good: autocorrelation decays quickly toward 0 within tens of lags. '
            'Suspicious: long positive tails (slow decay), high values at large lags, or periodic patterns (slow mixing).</p>'
            '</td></tr>\n'
        )
        html += (
            f'<tr class=biostyle><td><strong>Autocorrelation</strong></td>'
            f'<td><img style="max-width:{params["max_width_px"]}px;height:auto;display:block;" '
            f'src="{os.path.basename(figs_dir)}/autocorr.png" alt="autocorr"></td></tr>\n'
        )
    except (ValueError, TypeError) as e:
        logger.warning(f"Autocorr plot failed: {e}")

    html += '</table>\n'
    return html


def get_html_header(estimation_results: BayesianResults) -> str:
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


def get_html_preamble(estimation_results: BayesianResults, file_name: str) -> str:
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

    if estimation_results.user_notes is not None:
        # User notes
        html += (
            f'<blockquote style="border: 2px solid #666; '
            f'padding: 10px; background-color:'
            f' #ccc;">{estimation_results.user_notes}</blockquote>'
        )
    return html


def generate_one_row(description: str, value: str) -> str:
    return (
        f'<tr class=biostyle><td align=right >'
        f'<strong>{description}</strong>: </td> '
        f'<td>{value}</td></tr>\n'
    )


def get_html_general_statistics(estimation_results: BayesianResults) -> str:
    """Get the general statistics coded in HTML

    :return: HTML code
    """

    html = '<table border="0">\n'
    for description, value in estimation_results.generate_general_information().items():
        html += generate_one_row(description=description, value=f'{value}')
    html += '</table>\n'
    return html


def get_html_one_parameter(
    estimation_results: BayesianResults,
    parameter_number: int,
    parameter_name: str,
) -> str:
    """Generate the HTML code for one row of the table of the estimated parameters.

    :param estimation_results: estimation results.
    :param parameter_number: index of the parameter
    :param parameter_name: name of the parameter to report. If None, taken from estimation results.
    :return: HTML code for the row
    """
    estimated_value = estimation_results.parameters.get(parameter_name)
    if estimated_value is None:
        return ''
    output = '<tr class=biostyle>'
    output += f'<td>{parameter_number}</td>'
    output += f'<td>{parameter_name}</td>'
    # : str
    # estimate: float
    # std_err: float
    # z_value: float | None
    # p_value: float | None
    # hdi_low: float | None
    # hdi_high: float | None

    # Value
    value = estimated_value.estimate
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
    estimation_results: BayesianResults,
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
    html += '<th>Value</th>'
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
                parameter_name=name,
            )
            + '\n'
        )
        rows.append((parameter_index, name, row_html))

    if sort_by_name:
        rows.sort(key=lambda x: x[1])

    for _, _, row_html in rows:
        html += row_html
    html += '</table>\n'

    if estimated_parameters:
        html += '<table border="0">\n'
        for column, doc in EstimatedBeta.documentation.items():
            html += '<tr class=biostyle>'
            html += f'<td>{column}:</td>'
            html += f'<td>{doc}</td>'
            html += '</tr>\n'
        html += '</table>\n'
    return html


def generate_html_file(
    estimation_results: BayesianResults,
    filename: str,
    overwrite=False,
    figure_size: FigureSize = FigureSize.MEDIUM,
) -> None:
    """Generate an HTML file with the estimation results

    :param estimation_results: estimation results
    :param filename: name of the file
    :param overwrite: if True and the file exists, it is overwritten
    :param figure_size: FigureSize for diagnostics (default: MEDIUM)
    """
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
            estimated_parameters=True,
            sort_by_name=True,
        )
        other_variables = get_html_estimated_parameters(
            estimation_results=estimation_results,
            estimated_parameters=False,
            sort_by_name=True,
        )
        diagnostics = get_html_arviz_diagnostics(
            estimation_results=estimation_results,
            html_filename=filename,
            figure_size=figure_size,
        )

        footer = get_html_footer()

        print(header, file=file)
        print(preamble, file=file)
        print('<h1>Bayesian estimation report</h1>', file=file)
        print(general_statistics, file=file)
        print('<h1>Estimated parameters</h1>', file=file)
        print(parameters, file=file)
        print('<h1>Other simulated variables</h1>', file=file)
        print(other_variables, file=file)
        if diagnostics:
            print(diagnostics, file=file)
        print(footer, file=file)
    logger.info(f'File {filename} has been generated.')
