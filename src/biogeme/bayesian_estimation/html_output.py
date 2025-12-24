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
import matplotlib.pyplot as plt
import numpy as np
from biogeme import version
from biogeme.exceptions import BiogemeError

from .bayesian_results import BayesianResults, EstimatedBeta
from ..parameters import Parameters

logger = logging.getLogger(__name__)


class EmptyListOfParameters(BiogemeError): ...


# --- Identification diagnostics section ---
def _flatten_draws_to_matrix(idata_group, var_names: list[str]) -> np.ndarray:
    """
    Return a 2D array of shape (n_draws_total, n_params) from an ArviZ group.

    The group is expected to be an xarray Dataset (for example, ``idata.posterior`` or
    ``idata.prior``) containing scalar parameters stored with dimensions
    ``(chain, draw)``.

    Missing variables are ignored.

    :param idata_group: ArviZ/xarray dataset (for example, ``idata.posterior`` or ``idata.prior``).
    :param var_names: Names of the scalar parameters to extract.
    :return: A 2D NumPy array with one column per parameter and one row per draw.
    """
    cols: list[np.ndarray] = []
    for name in var_names:
        if name not in idata_group:
            continue
        da = idata_group[name]
        # Expected dims: (chain, draw). Flatten to (chain*draw,)
        values = np.asarray(da.values)
        if values.ndim != 2:
            # Only support scalar parameters stored as (chain, draw)
            continue
        cols.append(values.reshape(-1))
    if not cols:
        return np.empty((0, 0), dtype=float)
    return np.column_stack(cols).astype(float, copy=False)


def _cov_eigen_diagnostics(
    draws_2d: np.ndarray,
    var_names: list[str] | None = None,
    *,
    near_singular_tol_ratio: float = 1e-10,
    top_eigenvector_loadings: int = 10,
) -> dict:
    """
    Compute covariance eigen-structure diagnostics from posterior/prior draws.

    The input must be a 2D array of draws with shape ``(n_draws, n_parameters)``.
    The function computes the covariance matrix and reports the smallest and largest
    (nonnegative-clipped) eigenvalues, a condition number, and an effective rank
    (Shannon entropy of normalized eigenvalues).

    If the covariance is detected as ill-conditioned (large anisotropy), the eigenvector associated with the
    *largest* eigenvalue (largest posterior-variance direction) is also reported. This direction is the covariance
    analogue of the near-null Hessian direction used in maximum-likelihood identification checks.

    :param draws_2d: Draws arranged as a 2D array of shape ``(n_draws, n_parameters)``.
    :param var_names: Optional parameter names used to label the eigenvector entries.
    :param near_singular_tol_ratio: Threshold on ``min_eigenvalue / max_eigenvalue`` below which
        the covariance is considered near singular (used to flag large condition number: cond >= 1/tol).
    :param top_eigenvector_loadings: Maximum number of largest-magnitude coefficients to report in
        ``max_eigenvector_top``.
    :return: A dictionary of diagnostics. Always includes basic scalar metrics when possible.
        When ill-conditioned, also includes ``max_eigenvector`` and ``max_eigenvector_top``.
    """
    if draws_2d.size == 0:
        return {}

    n_draws, n_params = draws_2d.shape
    if n_draws < 2 or n_params < 1:
        return {"n_parameters": int(n_params), "n_draws": int(n_draws)}

    # Center and covariance
    x = draws_2d - np.mean(draws_2d, axis=0, keepdims=True)
    cov = (x.T @ x) / float(n_draws - 1)
    cov = 0.5 * (cov + cov.T)

    # Need eigenvectors -> eigh (symmetric)
    eigvals, eigvecs = np.linalg.eigh(cov)

    eigvals = np.asarray(eigvals, dtype=float)
    eig_clipped = np.clip(eigvals, 0.0, None)

    min_eig = float(np.min(eig_clipped))
    max_eig = float(np.max(eig_clipped))

    cond_ratio = float(max_eig / min_eig) if min_eig > 0.0 else float("inf")

    # Effective rank (Shannon)
    total = float(np.sum(eig_clipped))
    if total > 0.0:
        p = eig_clipped / total
        p = p[p > 0.0]
        h = float(-np.sum(p * np.log(p)))
        eff_rank = float(np.exp(h))
    else:
        eff_rank = 0.0

    out: dict = {
        "n_parameters": int(n_params),
        "n_draws": int(n_draws),
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "condition_number": cond_ratio,
        "effective_rank": eff_rank,
    }

    if max_eig > 0.0:
        ratio = float(min_eig / max_eig)
        out["min_eigenvalue_ratio"] = ratio
        out["condition_number"] = cond_ratio

        # Ill-conditioned => report the eigenvector for largest eigenvalue (weak identification direction)
        if cond_ratio >= (1.0 / float(near_singular_tol_ratio)):
            v = np.asarray(eigvecs[:, -1], dtype=float)  # column for largest eigenvalue

            # Stabilize sign so output is reproducible/readable
            idx = int(np.argmax(np.abs(v)))
            if v[idx] < 0:
                v = -v

            names = (
                var_names
                if (var_names is not None and len(var_names) == n_params)
                else [f"param_{i}" for i in range(n_params)]
            )

            mapping = {names[i]: float(v[i]) for i in range(n_params)}

            order = np.argsort(np.abs(v))[::-1]
            k = int(min(max(top_eigenvector_loadings, 1), n_params))
            top = [(names[int(i)], float(v[int(i)])) for i in order[:k]]

            out["max_eigenvector"] = mapping
            out["max_eigenvector_top"] = top

    return out


def format_real_number(value: float) -> str:
    """
    Format a real number for inclusion in an HTML table.

    :param value: Number to format.
    :return: A short string representation (currently using ``.3g`` formatting).
    """
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
    """
    Generate ArviZ diagnostic figures and return an HTML snippet embedding them.

    The figures (trace, rank, energy, autocorrelation) are saved next to the HTML file
    in a sibling folder named ``<html_basename>_figs``.

    :param estimation_results: Bayesian estimation results holding the ``InferenceData``.
    :param html_filename: Target HTML filename (used to locate the figures directory).
    :param var_names: Optional list of variables to include in the diagnostics.
    :param figure_size: Size level controlling figure sizes and embedded image width.
    :return: An HTML snippet containing ``<img>`` tags (or an empty string if disabled).
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


def get_html_general_statistics(estimation_results: BayesianResults) -> str:
    """Get the general statistics coded in HTML

    :return: HTML code
    """

    html = '<table border="0">\n'
    for description, value in estimation_results.generate_general_information().items():
        html += generate_one_row(description=description, value=f'{value}')
    html += '</table>\n'
    return html


# --- Identification diagnostics section ---
def get_html_identification_diagnostics(estimation_results: BayesianResults) -> str:
    """
    Generate an HTML section reporting identification diagnostics.

    The goal is to help detect (i) non-identification / weak identification
    and (ii) cases where parameters appear to be primarily determined by the
    prior rather than by the likelihood.

    The section is based on the posterior (and, if present in the
    ``InferenceData``, the prior) draws.

    Interpretation guide (heuristics):

    - **Posterior covariance eigenvalues / condition number**: a very small
      minimum eigenvalue or a very large condition number suggests directions
      in parameter space that are nearly flat (non-identified or weakly
      identified). This typically manifests as strong posterior correlations,
      slow mixing, divergent transitions, and sensitivity to priors.

    - **Effective rank**: an effective rank substantially smaller than the
      number of parameters indicates that the posterior variability is
      concentrated in a lower-dimensional subspace, which is consistent with
      linear (or nearly linear) dependencies among parameters.

    - **Prior vs posterior dispersion** (only if prior draws were saved):
      if a parameter's posterior standard deviation is close to its prior
      standard deviation, the data may be providing little information about
      that parameter ("identified by the prior"). Conversely, a much smaller
      posterior standard deviation indicates that the likelihood is informative.

    These diagnostics are *not* formal proofs of non-identification; they are
    practical signals to investigate the model specification (normalizations,
    redundant parameters, collinearity, and coding).

    :param estimation_results: Bayesian estimation results.
    :return: HTML code for the identification diagnostics section, or an empty string if unavailable.
    """

    # If the BayesianResults object does not expose diagnostics, fail gracefully.
    if not hasattr(estimation_results, "identification_diagnostics"):
        return ""

    try:
        identification_threshold = Parameters().get_value('identification_threshold')
        diag = estimation_results.identification_diagnostics(
            identification_threshold=identification_threshold
        )
    except Exception as e:  # pragma: no cover
        logger.warning(f"Identification diagnostics failed: {e}")
        return ""

    if not isinstance(diag, dict) or not diag:
        return ""

    # Ensure covariance/eigen diagnostics are present for HTML reporting.
    # Some implementations may only return per-parameter prior/posterior dispersion.
    idata = estimation_results.idata

    # Use the model's reported parameter names as defaults
    default_names = list(estimation_results.parameters.keys())

    posterior_block = diag.get("posterior")
    if not isinstance(posterior_block, dict):
        posterior_block = {}
        diag["posterior"] = posterior_block

    if not posterior_block:
        try:
            draws = _flatten_draws_to_matrix(idata.posterior, default_names)
            diag["posterior"] = _cov_eigen_diagnostics(draws, var_names=default_names)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Posterior covariance diagnostics failed: {e}")

    prior_block = diag.get("prior")
    if not isinstance(prior_block, dict):
        prior_block = {}
        diag["prior"] = prior_block

    # Only compute prior diagnostics if a prior group is present.
    if hasattr(idata, "prior") and (not prior_block):
        try:
            draws = _flatten_draws_to_matrix(idata.prior, default_names)
            diag["prior"] = _cov_eigen_diagnostics(draws, var_names=default_names)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Prior covariance diagnostics failed: {e}")

    html = '<h1>Identification diagnostics</h1>\n'

    # Explanatory text
    html += (
        '<p class="biostyle">'
        'This section reports quick numerical checks for <em>non-identification</em> or <em>weak identification</em>. '
        'Intuitively, identification problems mean that some combinations of parameters can change without changing '
        'the likelihood much, so the posterior is very wide (or nearly flat) in some directions. '
        'These checks use the posterior draws (and the prior draws, if available).'
        '</p>\n'
        '<h2>How to read the numbers</h2>\n'
        '<ul class="biostyle">'
        '<li><strong>Posterior covariance diagnostics</strong> (eigenvalues, condition number, effective rank): '
        'these describe the <em>shape</em> of the posterior cloud in parameter space.'
        '<ul class="biostyle">'
        '<li><strong>max_eigenvalue</strong>: the <em>largest posterior-variance</em> direction (widest direction of the posterior). When identification is weak, the posterior can become extremely wide along some linear combination of parameters; this often shows up as a very large <code>max_eigenvalue</code> together with a large condition number. If reported, the <code>max_eigenvector_top</code> loadings indicate which parameters contribute most to that weakly identified linear combination.</li>'
        '<li><strong>condition_number = max_eigenvalue / min_eigenvalue</strong>: anisotropy of the posterior covariance. Larger values indicate stronger near-dependencies among parameters. Rough rule of thumb: around <strong>10^3</strong> deserves attention; <strong>10^5</strong> or more is a strong red flag.</li>'
        '<li><strong>effective_rank</strong>: an “effective dimension” of posterior variability (between 0 and <code>n_parameters</code>). If it is much smaller than <code>n_parameters</code>, the posterior variability concentrates in a lower-dimensional subspace, consistent with (near) linear dependencies among parameters.</li>'
        '</ul></li>'
        '<li><strong>Prior covariance diagnostics</strong>: same metrics, but for the prior. '
        'If the prior has normal scale and full rank but the posterior becomes ill-conditioned, '
        'the issue is typically in the likelihood/model specification (not in the prior).</li>'
        '<li><strong>Identified by the prior</strong> (requires prior draws): '
        'compare prior vs posterior dispersion. For each parameter, '
        '<code>std_ratio_post_over_prior ≈ 1</code> means the data did not shrink uncertainty much (likelihood weakly informative for that parameter). '
        'A ratio <strong>well below 1</strong> (say 0.1 or 0.01) means the likelihood is informative for that parameter.</li>'
        '</ul>\n'
        '</p>\n'
    )

    # Flags (if any)
    flags = diag.get("flags")
    if flags:
        html += (
            '<p class="biostyle"><strong>Flags:</strong> '
            + ", ".join([str(f) for f in flags])
            + "</p>\n"
        )

    # Helper to render a small dict as a 2-col table
    def _render_kv_table(title: str, dct: dict) -> str:
        if not dct:
            return ""

        def _fmt(v) -> str:
            # list of (name, coeff) for compact eigenvector display
            if isinstance(v, list) and v and isinstance(v[0], tuple) and len(v[0]) == 2:
                parts: list[str] = []
                for name, coef in v:
                    coef_str = (
                        format_real_number(float(coef))
                        if isinstance(coef, (float, np.floating))
                        else str(coef)
                    )
                    parts.append(f"{coef_str}·{name}")
                return ", ".join(parts)

            # dict of {name: coeff}: show only top contributors
            if isinstance(v, dict):
                items = list(v.items())
                items.sort(key=lambda t: abs(float(t[1])), reverse=True)
                parts: list[str] = []
                for name, coef in items[:10]:
                    parts.append(f"{format_real_number(float(coef))}·{name}")
                return ", ".join(parts)

            if isinstance(v, float):
                if np.isinf(v):
                    return "inf"
                if np.isnan(v):
                    return "nan"
                return format_real_number(v)

            return str(v)

        out = '<h2>' + title + '</h2>\n'
        out += '<table border="0">\n'
        for k, v in dct.items():
            out += generate_one_row(description=str(k), value=_fmt(v))
        out += '</table>\n'
        return out

    html += _render_kv_table(
        "Posterior covariance diagnostics", diag.get("posterior", {})
    )
    html += _render_kv_table("Prior covariance diagnostics", diag.get("prior", {}))

    # Per-parameter table (if available)
    per_param = diag.get("per_parameter")
    if per_param is not None:
        html += '<h2>Per-parameter prior/posterior dispersion</h2>\n'
        html += (
            '<p class="biostyle">'
            'The table below compares posterior and prior standard deviations when prior draws are available. '
            'A ratio close to 1 suggests the prior dominates; a ratio well below 1 suggests the data are informative. '
            '</p>\n'
        )
        # If it is a pandas DataFrame (or any object providing to_html), use it.
        if hasattr(per_param, "to_html"):
            html += per_param.to_html(
                index=True,
                border=0,
                classes=["table", "table-striped", "table-sm"],
                justify="left",
            )
            html += "\n"
        else:
            # Fallback: try to interpret as an iterable of mappings
            try:
                rows = list(per_param)
            except TypeError:
                rows = []
            if rows:
                # Determine columns
                if isinstance(rows[0], dict):
                    cols = list(rows[0].keys())
                else:
                    cols = []
                html += '<table border="1">\n'
                html += '<tr class=biostyle>'
                for c in cols:
                    html += f'<th>{c}</th>'
                html += '</tr>\n'
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    html += '<tr class=biostyle>'
                    for c in cols:
                        html += f'<td>{r.get(c, "")}</td>'
                    html += '</tr>\n'
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
                parameter_name=name,
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
        for column, doc in EstimatedBeta.documentation.items():
            html += '<tr class=biostyle>'
            html += f'<td>{column}:</td>'
            html += f'<td>{doc}</td>'
            html += '</tr>\n'
        html += '</table>\n'
    return html


def generate_html_simulated_data(estimation_results: BayesianResults) -> str:
    simulated_data = estimation_results.report_stored_variables()
    return simulated_data.to_html(
        index=False,
        border=0,
        classes=["table", "table-striped", "table-sm"],
        justify="left",
    )


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
            figure_size=figure_size,
        )
        if diagnostics:
            print(diagnostics, file=file)

        footer = get_html_footer()
        print(footer, file=file)
    logger.info(f'File {filename} has been generated.')
