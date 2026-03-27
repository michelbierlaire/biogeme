from __future__ import annotations

"""Lightweight Bayesian results summaries serializable to YAML.

This module defines summary objects that store the information needed to inspect
Bayesian estimation results without keeping the full posterior draws in memory
or on disk. It is intended as a compact alternative to very large NetCDF files.

The summary objects preserve:

- general metadata about the estimation;
- posterior summaries for scalar parameters;
- metadata about non-scalar posterior variables;
- optional predictive and model-comparison criteria.

They do not preserve posterior draws. Therefore, methods requiring the full
posterior sample, such as sensitivity analysis from draws or ArviZ-based
recomputation of diagnostics, are intentionally unavailable.

Michel Bierlaire
Tue Mar 17 2026, 11:43:18
"""

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from enum import Enum

import pandas as pd
import yaml
from tabulate import tabulate

from biogeme.exceptions import BiogemeError


@dataclass
class EstimatedBetaSummary:
    """Posterior summary for one scalar parameter.

    :param name: Name of the parameter.
    :param mean: Posterior mean.
    :param median: Posterior median.
    :param mode: Posterior mode.
    :param std_err: Posterior standard deviation.
    :param z_value: Mean divided by posterior standard deviation.
    :param p_value: Two-sided posterior tail probability relative to zero.
    :param hdi_low: Lower bound of the highest-density interval.
    :param hdi_high: Upper bound of the highest-density interval.
    :param rhat: Rank-normalized R-hat diagnostic.
    :param effective_sample_size_bulk: Bulk effective sample size.
    :param effective_sample_size_tail: Tail effective sample size.
    """

    name: str
    mean: float
    median: float
    mode: float
    std_err: float
    z_value: float | None
    p_value: float | None
    hdi_low: float | None
    hdi_high: float | None
    rhat: float
    effective_sample_size_bulk: float
    effective_sample_size_tail: float

    documentation: ClassVar[dict[str, str]] = {
        "Name": "Identifier of the model parameter being estimated.",
        "Value": "Posterior mean (expected value) of the parameter.",
        "Median": "Posterior median (50% quantile) of the parameter.",
        "Mode": "Posterior mode (most frequent value) of the parameter.",
        "Std err.": (
            "Posterior standard deviation, measuring uncertainty around the mean."
        ),
        "z-value": (
            "Standardized estimate (mean divided by std. dev.), indicating "
            "signal-to-noise ratio."
        ),
        "p-value": (
            "Two-sided Bayesian tail probability that the parameter differs in "
            "sign from zero."
        ),
        "HDI low / HDI high": (
            "Lower and upper bounds of the Highest Density Interval containing "
            "the most probable parameter values."
        ),
        "R-hat (Gelman–Rubin)": (
            "Convergence diagnostic; values very close to 1 (typically ≤ 1.01) "
            "indicate well-mixed chains."
        ),
        "ESS (bulk)": (
            "Effective sample size for the central part of the posterior; values "
            "above ~400 are generally considered sufficient."
        ),
        "ESS (tail)": (
            "Effective sample size for the posterior tails; values above ~100 "
            "ensure reliable estimates of extreme quantiles."
        ),
    }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the summary to a plain dictionary.

        :return: Dictionary representation.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EstimatedBetaSummary:
        """Build an instance from a plain dictionary.

        :param data: Dictionary representation.
        :return: Reconstructed summary object.
        """
        return cls(**data)


@dataclass
class BayesianResultsSummary:
    """Compact summary of Bayesian estimation results.

    This object is designed to be saved as YAML and reloaded later without
    requiring the original posterior draws. It supports summary-level inspection
    of the estimation results, but not computations that depend on posterior
    samples.

    :param model_name: Name of the estimated model.
    :param data_name: Name of the dataset.
    :param chains: Number of MCMC chains.
    :param draws: Number of draws per chain.
    :param hdi_prob: Credible mass used for highest-density intervals.
    :param calculate_likelihood: Whether likelihood-based quantities were
        computed.
    :param calculate_waic: Whether WAIC was computed.
    :param calculate_loo: Whether LOO was computed.
    :param beta_names: Names of the model parameters of primary interest.
    :param parameters: Mapping from scalar posterior variable name to summary.
    :param array_metadata: Metadata for posterior variables with extra
        dimensions beyond chain and draw.
    :param posterior_predictive_loglike: Posterior-predictive log density.
    :param expected_log_likelihood: Posterior expectation of the total
        log-likelihood.
    :param best_draw_log_likelihood: Best total log-likelihood across posterior
        draws.
    :param waic: WAIC criterion.
    :param waic_se: Standard error of WAIC.
    :param p_waic: Effective number of parameters for WAIC.
    :param loo: LOO criterion.
    :param loo_se: Standard error of LOO.
    :param p_loo: Effective number of parameters for LOO.
    :param sampler: Name of the sampler.
    :param target_accept: Target acceptance rate.
    :param run_time: Total run time as a string or numeric value.
    :param number_of_observations: Sample size.
    :param user_notes: Optional user-defined notes to include in reports.
    :param stored_variables_report: Optional report of variables stored in the
        underlying inference data.
    :param model_name: Name of the estimated model.
    :param data_name: Name of the dataset.
    :param chains: Number of MCMC chains.
    :param draws: Number of draws per chain.
    :param hdi_prob: Credible mass used for highest-density intervals.
    :param calculate_likelihood: Whether likelihood-based quantities were
        computed.
    :param calculate_waic: Whether WAIC was computed.
    :param calculate_loo: Whether LOO was computed.
    :param beta_names: Names of the model parameters of primary interest.
    :param parameters: Mapping from scalar posterior variable name to summary.
    :param array_metadata: Metadata for posterior variables with extra
        dimensions beyond chain and draw.
    :param posterior_predictive_loglike: Posterior-predictive log density.
    :param expected_log_likelihood: Posterior expectation of the total
        log-likelihood.
    :param best_draw_log_likelihood: Best total log-likelihood across posterior
        draws.
    :param waic: WAIC criterion.
    :param waic_se: Standard error of WAIC.
    :param p_waic: Effective number of parameters for WAIC.
    :param loo: LOO criterion.
    :param loo_se: Standard error of LOO.
    :param p_loo: Effective number of parameters for LOO.
    :param sampler: Name of the sampler.
    :param target_accept: Target acceptance rate.
    :param run_time: Total run time as a string or numeric value.
    :param number_of_observations: Sample size.
    :param stored_variables_report: Optional report of variables stored in the
        underlying inference data.
    :param identification_diagnostics_summary: Optional precomputed summary of
        identification diagnostics derived from the posterior draws.
    :param diagnostic_figure_references: Optional mapping from diagnostic
        figure names to paths or filenames of pre-rendered figures.
    """

    model_name: str
    data_name: str
    chains: int
    draws: int
    hdi_prob: float
    calculate_likelihood: bool
    calculate_waic: bool
    calculate_loo: bool
    beta_names: list[str]
    parameters: dict[str, EstimatedBetaSummary]
    array_metadata: dict[str, dict]
    posterior_predictive_loglike: float | None = None
    expected_log_likelihood: float | None = None
    best_draw_log_likelihood: float | None = None
    waic: float | None = None
    waic_se: float | None = None
    p_waic: float | None = None
    loo: float | None = None
    loo_se: float | None = None
    p_loo: float | None = None
    sampler: str | None = None
    target_accept: float | None = None
    run_time: str | float | None = None
    number_of_observations: int | None = None
    user_notes: list[str] | None = None
    stored_variables_report: list[dict[str, Any]] | None = None
    identification_diagnostics_summary: dict[str, Any] | None = None
    diagnostic_figure_references: dict[str, str] | None = None
    has_posterior_draws: bool = field(default=False, init=False)

    @property
    def posterior_draws(self) -> int:
        """Total number of posterior draws.

        :return: Number of chains times number of draws per chain.
        """
        return self.chains * self.draws

    @staticmethod
    def _make_yaml_safe(value: Any) -> Any:
        """Recursively convert objects to YAML-safe plain Python values.

        This converts NumPy/Pandas scalar-like objects through ``item()``,
        converts mappings recursively, converts sequences to lists, and keeps
        plain Python scalars unchanged. NaN and infinite floats are converted to
        strings so that YAML output is stable and explicit.

        :param value: Object to sanitize.
        :return: YAML-safe object.
        """
        if value is None:
            return value

        if isinstance(value, Enum):
            return BayesianResultsSummary._make_yaml_safe(value.value)

        if isinstance(value, (datetime, date, time, timedelta)):
            return str(value)

        if isinstance(value, pd.Timestamp | pd.Timedelta):
            return str(value)

        if isinstance(value, (str, bool)):
            return value

        if isinstance(value, int):
            return int(value)

        if isinstance(value, float):
            numeric_value = float(value)
            if math.isnan(numeric_value):
                return 'nan'
            if math.isinf(numeric_value):
                return 'inf' if numeric_value > 0 else '-inf'
            return numeric_value

        if isinstance(value, pd.DataFrame):
            return BayesianResultsSummary._make_yaml_safe(
                value.to_dict(orient='records')
            )

        if isinstance(value, pd.Series):
            return BayesianResultsSummary._make_yaml_safe(value.to_dict())

        if hasattr(value, 'item') and callable(value.item):
            try:
                converted = value.item()
            except (ValueError, TypeError):
                pass
            else:
                if converted is not value:
                    return BayesianResultsSummary._make_yaml_safe(converted)

        if isinstance(value, Mapping):
            return {
                str(key): BayesianResultsSummary._make_yaml_safe(val)
                for key, val in value.items()
            }

        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [BayesianResultsSummary._make_yaml_safe(v) for v in value]

        if type(value).__module__ == 'datetime':
            return str(value)

        return value

    def to_dict(self) -> dict[str, Any]:
        """Serialize the summary to a plain dictionary suitable for YAML.

        :return: Dictionary representation.
        """
        payload = {
            "model_name": self.model_name,
            "data_name": self.data_name,
            "chains": self.chains,
            "draws": self.draws,
            "hdi_prob": self.hdi_prob,
            "calculate_likelihood": self.calculate_likelihood,
            "calculate_waic": self.calculate_waic,
            "calculate_loo": self.calculate_loo,
            "beta_names": list(self.beta_names),
            "parameters": {
                name: beta.to_dict() for name, beta in self.parameters.items()
            },
            "array_metadata": self.array_metadata,
            "posterior_predictive_loglike": self.posterior_predictive_loglike,
            "expected_log_likelihood": self.expected_log_likelihood,
            "best_draw_log_likelihood": self.best_draw_log_likelihood,
            "waic": self.waic,
            "waic_se": self.waic_se,
            "p_waic": self.p_waic,
            "loo": self.loo,
            "loo_se": self.loo_se,
            "p_loo": self.p_loo,
            "sampler": self.sampler,
            "target_accept": self.target_accept,
            "run_time": self.run_time,
            "number_of_observations": self.number_of_observations,
            "user_notes": self.user_notes,
            "stored_variables_report": self.stored_variables_report,
            "identification_diagnostics_summary": (
                self.identification_diagnostics_summary
            ),
            "diagnostic_figure_references": self.diagnostic_figure_references,
        }
        return self._make_yaml_safe(payload)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BayesianResultsSummary:
        """Rebuild a summary object from a plain dictionary.

        :param data: Dictionary representation.
        :return: Reconstructed summary object.
        """
        parameters = {
            name: EstimatedBetaSummary.from_dict(beta_data)
            for name, beta_data in data["parameters"].items()
        }
        return cls(
            model_name=data["model_name"],
            data_name=data["data_name"],
            chains=data["chains"],
            draws=data["draws"],
            hdi_prob=data["hdi_prob"],
            calculate_likelihood=data["calculate_likelihood"],
            calculate_waic=data["calculate_waic"],
            calculate_loo=data["calculate_loo"],
            beta_names=list(data.get("beta_names", [])),
            parameters=parameters,
            array_metadata=dict(data.get("array_metadata", {})),
            posterior_predictive_loglike=data.get("posterior_predictive_loglike"),
            expected_log_likelihood=data.get("expected_log_likelihood"),
            best_draw_log_likelihood=data.get("best_draw_log_likelihood"),
            waic=data.get("waic"),
            waic_se=data.get("waic_se"),
            p_waic=data.get("p_waic"),
            loo=data.get("loo"),
            loo_se=data.get("loo_se"),
            p_loo=data.get("p_loo"),
            sampler=data.get("sampler"),
            target_accept=data.get("target_accept"),
            run_time=data.get("run_time"),
            number_of_observations=data.get("number_of_observations"),
            user_notes=data.get("user_notes"),
            stored_variables_report=data.get("stored_variables_report"),
            identification_diagnostics_summary=data.get(
                "identification_diagnostics_summary"
            ),
            diagnostic_figure_references=data.get("diagnostic_figure_references"),
        )

    def dump_yaml(self, path: str) -> None:
        """Write the summary to a YAML file.

        :param path: Output YAML filename.
        """
        yaml_text = yaml.safe_dump(
            self.to_dict(),
            sort_keys=False,
            allow_unicode=True,
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(yaml_text)

    @classmethod
    def from_yaml_file(cls, filename: str) -> BayesianResultsSummary:
        """Load a summary from a YAML file.

        :param filename: Path to the YAML file.
        :return: Reconstructed summary object.
        """
        with open(filename, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def parameter_estimates(self) -> dict[str, EstimatedBetaSummary]:
        """Return only the explicitly declared model parameters.

        :return: Mapping from parameter name to summary.
        """
        names = set(self.parameters.keys()) & set(self.beta_names)
        return {k: v for k, v in self.parameters.items() if k in names}

    def other_variables(self) -> dict[str, EstimatedBetaSummary]:
        """Return scalar posterior summaries that are not listed as main parameters.

        :return: Mapping from variable name to summary.
        """
        names = set(self.parameters.keys()) - set(self.beta_names)
        return {k: v for k, v in self.parameters.items() if k in names}

    def list_array_variables(self) -> dict[str, dict]:
        """Return metadata for posterior variables with extra dimensions.

        :return: Metadata keyed by variable name.
        """
        return dict(self.array_metadata)

    def report_stored_variables(self) -> pd.DataFrame:
        """Return the stored-variable report as a DataFrame.

        :return: DataFrame describing stored variables.
        """
        if not self.stored_variables_report:
            return pd.DataFrame(columns=["group", "variable", "dims", "shape"])
        return pd.DataFrame(self.stored_variables_report)

    def get_identification_diagnostics_summary(self) -> dict[str, Any] | None:
        """Return the stored identification diagnostics summary.

        :return: Precomputed identification diagnostics summary, if available.
        """
        return self.identification_diagnostics_summary

    def get_diagnostic_figure_references(self) -> dict[str, str]:
        """Return the stored diagnostic figure references.

        :return: Mapping from diagnostic figure names to stored figure paths.
        """
        return (
            {}
            if self.diagnostic_figure_references is None
            else dict(self.diagnostic_figure_references)
        )

    def get_user_notes(self) -> list[str]:
        """Return the stored user notes.

        :return: User notes, or an empty list if none are available.
        """
        return [] if self.user_notes is None else list(self.user_notes)

    def generate_general_information(self) -> dict[str, Any]:
        """Generate a summary dictionary for display.

        :return: Dictionary of general result information.
        """
        results: dict[str, Any] = {
            "Sample size": self.number_of_observations,
            "Sampler": self.sampler,
            "Number of chains": self.chains,
            "Number of draws per chain": self.draws,
            "Total number of draws": self.posterior_draws,
            "Acceptance rate target": self.target_accept,
            "Run time": self.run_time,
        }
        if self.calculate_likelihood:
            results |= {
                "Posterior predictive log-likelihood (sum of log mean p)": (
                    None
                    if self.posterior_predictive_loglike is None
                    else f"{self.posterior_predictive_loglike:.2f}"
                ),
                "Expected log-likelihood E[log L(Y|θ)]": (
                    None
                    if self.expected_log_likelihood is None
                    else f"{self.expected_log_likelihood:.2f}"
                ),
                "Best-draw log-likelihood (posterior upper bound)": (
                    None
                    if self.best_draw_log_likelihood is None
                    else f"{self.best_draw_log_likelihood:.2f}"
                ),
            }
        if self.calculate_waic:
            results |= {
                "WAIC (Widely Applicable Information Criterion)": (
                    None if self.waic is None else f"{self.waic:.2f}"
                ),
                "WAIC Standard Error": (
                    None if self.waic_se is None else f"{self.waic_se:.2f}"
                ),
                "Effective number of parameters (p_WAIC)": (
                    None if self.p_waic is None else f"{self.p_waic:.2f}"
                ),
            }
        if self.calculate_loo:
            results |= {
                "LOO (Leave-One-Out Cross-Validation)": (
                    None if self.loo is None else f"{self.loo:.2f}"
                ),
                "LOO Standard Error": (
                    None if self.loo_se is None else f"{self.loo_se:.2f}"
                ),
                "Effective number of parameters (p_LOO)": (
                    None if self.p_loo is None else f"{self.p_loo:.2f}"
                ),
            }
        return results

    def short_summary(self) -> str:
        """Return a plain-text summary table.

        :return: Text summary.
        """
        return tabulate(self.generate_general_information().items(), tablefmt="plain")

    def get_beta_values(self, my_betas: list[str] | None = None) -> dict[str, float]:
        """Retrieve posterior means for selected parameters.

        :param my_betas: Names of the requested parameters. If None, all model
            parameters are returned.
        :return: Mapping from parameter name to posterior mean.
        :raises BiogemeError: If an unknown parameter name is requested.
        """
        the_betas = self.parameter_estimates()
        if my_betas is not None:
            unknown = [b for b in my_betas if b not in the_betas]
            if unknown:
                raise BiogemeError(f"Unknown parameter(s): {', '.join(unknown)}")
            selected = {name: the_betas[name] for name in my_betas}
        else:
            selected = the_betas
        return {name: beta.mean for name, beta in selected.items()}

    def arviz_summary(self) -> pd.DataFrame:
        """Unavailable for YAML summaries.

        :raises BiogemeError: Always, because posterior draws are not available.
        """
        raise BiogemeError(
            "ArviZ summary is unavailable from a YAML BayesianResultsSummary. "
            "Load the NetCDF results instead."
        )

    def summarize_array_variable(self, *args: Any, **kwargs: Any) -> Any:
        """Unavailable for YAML summaries.

        :raises BiogemeError: Always, because posterior draws are not available.
        """
        raise BiogemeError(
            "Array-variable summarization is unavailable from a YAML "
            "BayesianResultsSummary. Load the NetCDF results instead."
        )

    def posterior_mean_by_observation(self, var_name: str) -> pd.DataFrame:
        """Unavailable for YAML summaries.

        :param var_name: Name of the requested variable.
        :raises BiogemeError: Always, because posterior draws are not available.
        """
        raise BiogemeError(
            f'Posterior means by observation for "{var_name}" are unavailable from '
            "a YAML BayesianResultsSummary. Load the NetCDF results instead."
        )

    def get_betas_for_sensitivity_analysis(
        self,
        my_betas: list[str] | None = None,
        size: int = 100,
    ) -> list[dict[str, float]]:
        """Unavailable for YAML summaries.

        :param my_betas: Names of the requested parameters.
        :param size: Number of requested draws.
        :raises BiogemeError: Always, because posterior draws are not available.
        """
        raise BiogemeError(
            "Sensitivity-analysis draws are unavailable from a YAML "
            "BayesianResultsSummary. Load the NetCDF results instead."
        )

    def identification_diagnostics(self, *args: Any, **kwargs: Any) -> Any:
        """Unavailable for YAML summaries.

        :raises BiogemeError: Always, because posterior draws are not available.
        """
        raise BiogemeError(
            "Identification diagnostics are unavailable from a YAML "
            "BayesianResultsSummary. Load the NetCDF results instead."
        )
