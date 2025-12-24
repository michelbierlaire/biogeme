"""Store results in Pandas databases

Michel Bierlaire
Thu Oct 30 2025, 10:02:39
"""

import pandas as pd

from .bayesian_results import BayesianResults


def _build_parameters_dataframe(
    estimation_results: BayesianResults,
    *,
    estimated_parameters: bool = True,
    renaming_parameters: dict[str, str] | None = None,
    sort_by_name: bool = False,
) -> pd.DataFrame:
    """
    Return a pandas DataFrame with the same content as the HTML table of parameters.

    Columns:
        Id, Name, Value (mean), Value (median), Value (mode), std err., z-value, p-value, HDI low, HDI high, R hat, ESS (bulk), ESS (tail)
    """
    if renaming_parameters is not None:
        name_values = list(renaming_parameters.values())
        if len(name_values) != len(set(name_values)):
            # Optional: log a warning if you have a logger; here we stay silent as requested
            pass

    param_keys = (
        estimation_results.parameter_estimates()
        if estimated_parameters
        else estimation_results.other_variables()
    )

    rows: list[dict] = []
    for param_idx, original_name in enumerate(param_keys):
        est = estimation_results.parameters.get(original_name)
        if est is None:
            continue
        display_name = (
            renaming_parameters.get(original_name, original_name)
            if renaming_parameters is not None
            else original_name
        )
        rows.append(
            {
                "Id": param_idx,
                "Name": display_name,
                "Value (mean)": est.mean,
                "Value (median)": est.median,
                "Value (mode)": est.mode,
                "std err.": est.std_err,
                "z-value": est.z_value,
                "p-value": est.p_value,
                "HDI low": est.hdi_low,
                "HDI high": est.hdi_high,
                "R hat": est.rhat,
                "ESS (bulk)": est.effective_sample_size_bulk,
                "ESS (tail)": est.effective_sample_size_tail,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "Id",
            "Name",
            "Value (mean)",
            "Value (median)",
            "Value (mode)",
            "std err.",
            "z-value",
            "p-value",
            "HDI low",
            "HDI high",
            "R hat",
            "ESS (bulk)",
            "ESS (tail)",
        ],
    )

    if sort_by_name and not df.empty:
        df = df.sort_values(by=["Name", "Id"], kind="mergesort").reset_index(drop=True)

    df.set_index('Id', inplace=True)
    df.index.name = None
    return df


def get_pandas_estimated_parameters(
    estimation_results: BayesianResults,
    renaming_parameters: dict[str, str] | None = None,
    sort_by_name: bool = False,
) -> pd.DataFrame:
    """DataFrame of *estimated parameters* (same content as the HTML table)."""
    return _build_parameters_dataframe(
        estimation_results=estimation_results,
        estimated_parameters=True,
        renaming_parameters=renaming_parameters,
        sort_by_name=sort_by_name,
    )


def get_pandas_other_variables(
    estimation_results: BayesianResults,
    renaming_parameters: dict[str, str] | None = None,
    sort_by_name: bool = False,
) -> pd.DataFrame:
    """DataFrame of *other simulated variables* (same content as the HTML table)."""
    return _build_parameters_dataframe(
        estimation_results=estimation_results,
        estimated_parameters=False,
        renaming_parameters=renaming_parameters,
        sort_by_name=sort_by_name,
    )
