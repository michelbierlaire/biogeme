"""
Derived Bayesian results (posterior summaries) built from RawBayesianResults.

- Posterior mean  -> 'estimate' (analogous to MLE estimate)
- Posterior std   -> 'std_err'  (analogous to MLE standard error)
- z = mean / std  -> 'z_value'  (rough MLE-like t-stat analogue)
- p(two-sided)    -> min(2*P(theta>0), 2*P(theta<0)) from posterior draws
- HDI             -> credible interval (e.g., 94% by default)

Michel Bierlaire
Mon Nov 03 2025, 08:55:59
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gaussian_kde
from tabulate import tabulate

from biogeme.exceptions import BiogemeError
from biogeme.tools import timeit
from .raw_bayesian_results import RawBayesianResults

logger = logging.getLogger(__name__)

CHOICE_LABEL = "_choice"


class PosteriorSummary(str, Enum):
    """Type of posterior point estimate to extract."""

    MEAN = "mean"
    MODE = "mode"


@dataclass
class EstimatedBeta:
    name: str
    mean: float
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
        'Name': 'Identifier of the model parameter being estimated.',
        'Value': 'Posterior mean (expected value) of the parameter.',
        'Mode': 'Posterior mode (most frequent value) of the parameter',
        'Std err.': 'Posterior standard deviation, measuring uncertainty around the mean.',
        'z-value': 'Standardized estimate (mean divided by std. dev.), indicating signal-to-noise ratio.',
        'p-value': 'Two-sided Bayesian tail probability that the parameter differs in sign from zero.',
        'HDI low / HDI high': 'Lower and upper bounds of the Highest Density Interval containing the most probable parameter values.',
        'R-hat (Gelman–Rubin)': 'Convergence diagnostic; values very close to 1 (typically ≤ 1.01) indicate well-mixed chains.',
        'ESS (bulk)': 'Effective sample size for the central part of the posterior; values above ~400 are generally considered sufficient.',
        'ESS (tail)': 'Effective sample size for the posterior tails; values above ~100 ensure reliable estimates of extreme quantiles.',
    }


def _posterior_mode_kde(samples: np.ndarray) -> float:
    """Approximate posterior mode via Gaussian KDE."""
    s = samples[np.isfinite(samples)]
    if s.size == 0:
        return np.nan
    kde = gaussian_kde(s)
    xs = np.linspace(s.min(), s.max(), 512)
    dens = kde(xs)
    return float(xs[np.argmax(dens)])


@dataclass
class BayesianResults:
    """Posterior summaries for parameters, derived from RawBayesianResults.

    parameters: dict mapping parameter name -> EstimatedBeta
    """

    data_name: str
    chains: int
    draws: int
    hdi_prob: float
    parameters: dict[str, EstimatedBeta]
    array_metadata: dict[str, dict]

    def __init__(
        self,
        raw: RawBayesianResults,
        *,
        calculate_likelihood: bool,
        calculate_waic: bool,
        calculate_loo: bool,
        hdi_prob: float = 0.94,
        strict: bool = False,
    ) -> None:
        """
        Build BayesianResults from RawBayesianResults by loading the posterior and
        calculating per-parameter summaries.

        Parameters
        ----------
        raw : RawBayesianResults
            Minimal record with posterior_netcdf_path and parameter names.
        hdi_prob : float
            Credible mass for Highest Density Interval (e.g., 0.94 or 0.95).
        strict : bool
            If True, raise if any listed parameter is not found in the posterior.
        """
        self.raw_bayesian_results = raw
        self.calculate_likelihood = calculate_likelihood
        self.calculate_waic = calculate_waic
        self.calculate_loo = calculate_loo

        self._idata = raw.idata
        self._log_likelihood = None

        self._waic_res = None
        self._loo_res = None
        # ArviZ returns xarray Datasets keyed by variable name (no 'variable' dim)
        rhat_ds = az.rhat(self._idata, method="rank")
        ess_bulk_ds = az.ess(self._idata, method="bulk")
        ess_tail_ds = az.ess(self._idata, method="tail")

        def _scalar_from_ds(ds, var_name: str) -> float:
            # Returns float value for a scalar variable in an xarray Dataset
            if var_name not in ds:
                raise KeyError(
                    f"Variable '{var_name}' not found in diagnostic Dataset."
                )
            return float(np.asarray(ds[var_name]).squeeze())

        params: dict[str, EstimatedBeta] = {}
        arrays: dict[str, dict] = {}
        for name in self._idata.posterior.data_vars:
            da = self._idata.posterior[name]
            extra_dims = [d for d in da.dims if d not in ("chain", "draw")]
            if extra_dims:
                # Record metadata for later, do not summarize by default to avoid huge outputs
                arrays[name] = {
                    "dims": tuple(extra_dims),
                    "shape": tuple(
                        int(da.sizes[d]) for d in da.dims if d not in ("chain", "draw")
                    ),
                    "sizes": {d: int(da.sizes[d]) for d in extra_dims},
                    "dtype": str(da.dtype),
                }
                if strict:
                    raise ValueError(
                        f"Posterior variable '{name}' has extra dims {extra_dims}. "
                        f"Use 'summarize_array_variable' to extract slices/reductions."
                    )
                continue

            samples = np.asarray(da).reshape(-1)
            mean = float(np.nanmean(samples))
            mode = _posterior_mode_kde(samples)
            std = float(np.nanstd(samples, ddof=1)) if samples.size > 1 else np.nan
            z_value = mean / std if (std is not None and std > 0) else np.nan
            p_value = self._two_sided_p_from_posterior(samples)

            hdi_low = hdi_high = None
            try:
                hdi = az.hdi(samples, hdi_prob=hdi_prob)
                hdi_low, hdi_high = float(hdi[0]), float(hdi[1])
            except (ValueError, TypeError):
                pass

            params[name] = EstimatedBeta(
                name=name,
                mean=mean,
                mode=mode,
                std_err=std,
                z_value=z_value,
                p_value=p_value,
                hdi_low=hdi_low,
                hdi_high=hdi_high,
                rhat=_scalar_from_ds(rhat_ds, name),
                effective_sample_size_bulk=_scalar_from_ds(ess_bulk_ds, name),
                effective_sample_size_tail=_scalar_from_ds(ess_tail_ds, name),
            )

        if not params:
            raise ValueError("No scalar variables found in the posterior to summarize.")
        self.model_name = raw.model_name
        self.data_name = raw.data_name
        self.chains = int(getattr(raw, "chains", 0) or 0)
        self.draws = int(getattr(raw, "draws", 0) or 0)
        self.hdi_prob = hdi_prob
        self.parameters = params
        self.array_metadata = arrays
        self._posterior_predictive_loglike = None
        self._expected_log_likelihood = None
        self._best_draw_log_likelihood = None
        self._waic = None
        self._waic_se = None
        self._p_waic = None
        self._loo = None
        self._loo_se = None
        self._p_loo = None

    @classmethod
    def from_netcdf(
        cls,
        filename: str,
        *,
        calculate_likelihood: bool = True,
        calculate_waic: bool = True,
        calculate_loo: bool = True,
        hdi_prob: float = 0.94,
        strict: bool = False,
    ) -> BayesianResults:
        """Alternate constructor: build results directly from a NetCDF file.

        This uses RawBayesianResults.load(path) under the hood and then computes
        posterior summaries with the given options.
        """
        raw = RawBayesianResults.load(filename)
        return cls(
            raw,
            calculate_likelihood=calculate_likelihood,
            calculate_waic=calculate_waic,
            calculate_loo=calculate_loo,
            hdi_prob=hdi_prob,
            strict=strict,
        )

    @property
    def posterior_draws(self) -> int:
        return self.chains * self.draws

    def dump(self, path: str) -> None:
        """Write the underlying posterior + metadata to a single NetCDF file.

        Delegates to RawBayesianResults.save(path).
        """
        self.raw_bayesian_results.save(path)

    @staticmethod
    def _two_sided_p_from_posterior(samples: np.ndarray) -> float:
        """Bayesian two-sided p-value relative to 0: 2 * min(P(theta>0), P(theta<0))."""
        # Drop NaNs just in case
        s = samples[np.isfinite(samples)]
        if s.size == 0:
            return np.nan
        p_pos = np.mean(s > 0.0)
        p_neg = 1.0 - p_pos
        return 2.0 * min(p_pos, p_neg)

    @property
    @timeit(label="log_likelihood")
    def log_likelihood(self):
        if not self.calculate_likelihood:
            return None
        if self._log_likelihood is None:
            self._log_likelihood = self._idata.posterior[
                self.raw_bayesian_results.log_like_name
            ]
            try:
                self._idata.add_groups(
                    {"log_likelihood": xr.Dataset({CHOICE_LABEL: self._log_likelihood})}
                )
            except ValueError:
                ...  # If the group is already there, nothing has to be done.
        return self._log_likelihood

    @property
    @timeit(label="waic_res")
    def waic_res(self):
        if not self.calculate_waic:
            return None
        if self._waic_res is None:
            self._waic_res = az.waic(self.idata, var_name=CHOICE_LABEL)
        return self._waic_res

    @property
    @timeit(label="loo_res")
    def loo_res(self):
        if not self.calculate_loo:
            return None
        if self._loo_res is None:
            self._loo_res = az.loo(self.idata, var_name=CHOICE_LABEL)
        return self._loo_res

    def __getattr__(self, name: str) -> Any:
        # Check raw_estimation_results without triggering __getattr__ recursively
        if (
            'raw_bayesian_results' not in self.__dict__
            or self.__dict__['raw_bayesian_results'] is None
        ):
            raise BiogemeError(f'Impossible to obtain {name}. No result available.')
        return getattr(self.raw_bayesian_results, name)

    @property
    def idata(self) -> az.InferenceData:
        return self._idata

    @timeit(label='arviz_summary')
    def arviz_summary(self) -> pd.DataFrame:
        return az.summary(self.idata)

    @property
    @timeit(label='posterior_predictive_loglike')
    def posterior_predictive_loglike(self) -> float | None:
        """Posterior-predictive log density: sum_n log(mean_{chain,draw} p(y_n|theta)).

        This computes a *posterior-predictive* criterion (log pointwise predictive
        density using arithmetic averaging over theta). It is **not** the ML log-likelihood.
        Works for arrays with shape (chain, draw, obs) or (chain, draw) where the latter
        is treated as a total log-likelihood per draw.
        """
        if self.log_likelihood is None:
            return None
        if self._posterior_predictive_loglike is not None:
            return self._posterior_predictive_loglike
        ll = self.log_likelihood
        a = np.asarray(ll)
        if a.ndim not in (2, 3):
            raise ValueError(
                f"Expected log_likelihood with 2 or 3 dims ((chain, draw) or (chain, draw, obs)); got shape {a.shape}"
            )
        S = a.shape[0] * a.shape[1]
        if a.ndim == 2:
            # total log-likelihood per draw: compute log(mean(exp(total_ll)))
            a_max = np.max(a)
            sumexp = np.sum(np.exp(a - a_max))
            sumexp = np.clip(sumexp, 1e-300, np.inf)
            logmeanexp = np.log(sumexp) - np.log(S) + a_max
            return float(logmeanexp)
        # a.ndim == 3: (chain, draw, obs) -> sum over obs of log-mean-exp across draws
        a_max = np.max(a, axis=(0, 1), keepdims=True)  # (1,1,obs)
        sumexp = np.sum(np.exp(a - a_max), axis=(0, 1))  # (obs,)
        sumexp = np.clip(sumexp, 1e-300, np.inf)
        logmeanexp = np.log(sumexp) - np.log(S) + a_max.squeeze((0, 1))
        self._posterior_predictive_loglike = float(np.sum(logmeanexp))
        return self._posterior_predictive_loglike

    @property
    @timeit(label='expected_log_likelihood')
    def expected_log_likelihood(self) -> float:
        """E_theta[ log L(Y|theta) ] across posterior draws.

        Computes the mean of the *total* log-likelihood over (chain, draw).
        For (chain, draw, obs) arrays, totals are computed by summing over obs first.
        """
        if self.log_likelihood is None:
            return None
        if self._expected_log_likelihood is not None:
            return self._expected_log_likelihood

        ll = self.log_likelihood
        a = np.asarray(ll)
        if a.ndim == 3:
            a = a.sum(axis=2)  # (chain, draw)
        if a.ndim != 2:
            raise ValueError(f"Expected 2D or 3D log_likelihood; got shape {a.shape}")
        self._expected_log_likelihood = float(np.mean(a))
        return self._expected_log_likelihood

    @property
    @timeit(label='best_draw_log_likelihood')
    def best_draw_log_likelihood(self) -> float:
        if self.log_likelihood is None:
            return None
        if self._best_draw_log_likelihood is not None:
            return self._best_draw_log_likelihood
        """Max over draws of the total log-likelihood (upper-bound proxy for ML)."""
        ll = self.log_likelihood
        a = np.asarray(ll)
        if a.ndim == 3:
            a = a.sum(axis=2)
        if a.ndim != 2:
            raise ValueError(f"Expected 2D or 3D log_likelihood; got shape {a.shape}")
        self._best_draw_log_likelihood = float(np.max(a))
        return self._best_draw_log_likelihood

    @property
    @timeit(label='waic')
    def waic(self):
        if self.waic_res is None:
            return None
        if self._waic is None:
            self._waic = float(
                getattr(
                    self._waic,
                    "waic",
                    getattr(self.waic_res, "elpd_waic", float("nan")),
                )
            )
        return self._waic

    @property
    @timeit(label='waic_se')
    def waic_se(self):
        if self.waic_res is None:
            return None
        if self._waic_se is None:
            self._waic_se = float(
                getattr(
                    self._waic, "waic_se", getattr(self.waic_res, "se", float("nan"))
                )
            )
        return self._waic_se

    @property
    @timeit(label='p_waic')
    def p_waic(self):
        if self.waic_res is None:
            return None
        if self._p_waic is None:
            self._p_waic = float(getattr(self.waic_res, "p_waic", float("nan")))
        return self._p_waic

    @property
    @timeit(label='loo')
    def loo(self):
        if self.loo_res is None:
            return None
        if self._loo is None:
            self._loo = float(
                getattr(
                    self._loo, "loo", getattr(self.loo_res, "elpd_loo", float("nan"))
                )
            )
        return self._loo

    @property
    @timeit(label='loo_se')
    def loo_se(self):
        if self._loo_res is None:
            return None
        if self._loo_se is None:
            self._loo_se = float(
                getattr(self._loo, "loo_se", getattr(self.loo_res, "se", float("nan")))
            )
        return self._loo_se

    @property
    @timeit(label='p_loo')
    def p_loo(self):
        if self.loo_res is None:
            return self.loo_res
        if self._p_loo is None:
            self._p_loo = float(getattr(self.loo_res, "p_loo", float("nan")))
        return self._p_loo

    def parameter_estimates(self) -> dict[str, EstimatedBeta]:
        """Return only the parameters explicitly listed in `raw_bayesian_results.beta_names`.

        Missing names are ignored silently (they may have been skipped if multidimensional
        or missing in the posterior). The returned dict maps name -> EstimatedBeta.
        """
        names = set(self.parameters.keys()) & set(self.raw_bayesian_results.beta_names)
        return {k: v for k, v in self.parameters.items() if k in names}

    def other_variables(self) -> dict[str, EstimatedBeta]:
        """Return posterior scalar variables that are *not* listed as parameters.

        Useful to expose derived/deterministic quantities stored in the posterior
        (e.g., total log-likelihood) without mixing them with parameter estimates.
        """
        names = set(self.parameters.keys()) - set(self.raw_bayesian_results.beta_names)
        return {k: v for k, v in self.parameters.items() if k in names}

    def list_array_variables(self) -> dict[str, dict]:
        """Return metadata for posterior variables that have extra dims beyond (chain, draw).

        Each entry contains: dims (tuple), shape (tuple), sizes (dict), dtype (str).
        """
        return dict(self.array_metadata)

    def generate_general_information(self):
        results = {
            'Sample size': self.number_of_observations,
            'Sampler': self.sampler,
            'Number of chains': self.chains,
            'Number of draws per chain': self.draws,
            'Total number of draws': self.chains * self.draws,
            'Acceptance rate target': self.target_accept,
            'Run time': self.run_time,
        }
        if self.calculate_likelihood:
            results |= {
                'Posterior predictive log-likelihood (sum of log mean p)': f'{self.posterior_predictive_loglike:.2f}',
                'Expected log-likelihood E[log L(Y|θ)]': f'{self.expected_log_likelihood:.2f}',
                'Best-draw log-likelihood (posterior upper bound)': f'{self.best_draw_log_likelihood:.2f}',
            }
        if self.calculate_waic:
            results |= {
                'WAIC (Widely Applicable Information Criterion)': f'{self.waic:.2f}',
                'WAIC Standard Error': f'{self.waic_se:.2f}',
                'Effective number of parameters (p_WAIC)': f'{self.p_waic:.2f}',
            }
        if self.calculate_loo:
            results |= {
                'LOO (Leave-One-Out Cross-Validation)': f'{self.loo:.2f}',
                'LOO Standard Error': f'{self.loo_se:.2f}',
                'Effective number of parameters (p_LOO)': f'{self.p_loo:.2f}',
            }
        return results

    def short_summary(self):
        return tabulate(self.generate_general_information().items(), tablefmt="plain")

    def summarize_array_variable(
        self,
        name: str,
        *,
        dim: str,
        indices: list[int] | None = None,
        hdi_prob: float | None = None,
    ) -> dict[int, EstimatedBeta]:
        """Summarize a multi-dimensional posterior variable for specific indices along a given dimension.

        Parameters
        ----------
        name : str
            The posterior variable to summarize (must be in `array_metadata`).
        dim : str
            The extra dimension along which to pick indices (e.g., 'observations').
        indices : list[int] | None
            Which indices to summarize. If None, summarize all indices (may be large!).
        hdi_prob : float | None
            If provided, overrides the instance `hdi_prob` for this call.

        Returns
        -------
        dict[int, EstimatedBeta]
            Mapping from index to an EstimatedBeta summary based on samples across chains/draws.
        """
        if name not in self.array_metadata:
            raise KeyError(
                f"Variable '{name}' is not a recorded multi-dimensional posterior variable."
            )
        meta = self.array_metadata[name]
        if dim not in meta["sizes"]:
            raise KeyError(
                f"Dimension '{dim}' not found in variable '{name}' (dims: {meta['dims']})."
            )

        # Use already loaded idata
        idata = self._idata
        da = idata.posterior[name]

        size = meta["sizes"][dim]
        idx_list = list(range(size)) if indices is None else list(indices)
        out: dict[int, EstimatedBeta] = {}
        hp = self.hdi_prob if hdi_prob is None else hdi_prob

        # Precompute diagnostics datasets
        rhat_ds = az.rhat(idata, method="rank")
        ess_bulk_ds = az.ess(idata, method="bulk")
        ess_tail_ds = az.ess(idata, method="tail")

        for i in idx_list:
            # select along the requested dim
            sub = da.sel({dim: i})
            # ensure scalar after selecting; if still has more dims, skip
            extra_dims = [d for d in sub.dims if d not in ("chain", "draw")]
            if extra_dims:
                continue
            samples = np.asarray(sub).reshape(-1)
            mean = float(np.nanmean(samples))
            std = float(np.nanstd(samples, ddof=1)) if samples.size > 1 else np.nan
            mode = _posterior_mode_kde(samples)
            z_value = mean / std if (std is not None and std > 0) else np.nan
            p_value = self._two_sided_p_from_posterior(samples)
            hdi_low = hdi_high = None
            try:
                hdi = az.hdi(samples, hdi_prob=hp)
                hdi_low, hdi_high = float(hdi[0]), float(hdi[1])
            except (ValueError, TypeError):
                pass

            # rhat/ess for that slice live under a composite var name in the diagnostics dataset,
            # ArviZ uses coordinate-based variables; so compute directly when selection is scalar.
            try:
                rhat_val = float(np.asarray(az.rhat(sub, method="rank")).squeeze())
                essb_val = float(np.asarray(az.ess(sub, method="bulk")).squeeze())
                esst_val = float(np.asarray(az.ess(sub, method="tail")).squeeze())
            except (ValueError, TypeError):
                rhat_val = np.nan
                essb_val = np.nan
                esst_val = np.nan

            out[i] = EstimatedBeta(
                name=f"{name}[{dim}={i}]",
                mean=mean,
                mode=mode,
                std_err=std,
                z_value=z_value,
                p_value=p_value,
                hdi_low=hdi_low,
                hdi_high=hdi_high,
                rhat=rhat_val,
                effective_sample_size_bulk=essb_val,
                effective_sample_size_tail=esst_val,
            )
        return out

    def posterior_mean_by_observation(self, var_name: str) -> pd.DataFrame:
        """
        Return a DataFrame giving the posterior mean for each observation of the requested variable.

        The variable must have shape (chain, draw, obs_dim), i.e., exactly one dimension
        besides 'chain' and 'draw'. The returned DataFrame has one row per observation,
        indexed by the observation coordinate if available.

        :param var_name: Name of the posterior variable to summarize.
        :return: pd.DataFrame with index = observation and column = posterior mean of var_name.
        :raises BiogemeError: if the variable is not present, not an array, or not indexed by a single observation dimension.
        """
        if var_name not in self.idata.posterior:
            raise BiogemeError(f'Variable "{var_name}" not found in posterior.')
        da = self.idata.posterior[var_name]
        extra_dims = [d for d in da.dims if d not in ("chain", "draw")]
        if len(extra_dims) == 0:
            raise BiogemeError(
                f'Variable "{var_name}" has no observation dimension; dims are {da.dims!r}.'
            )
        if len(extra_dims) > 1:
            raise BiogemeError(
                f'Variable "{var_name}" has multiple non-(chain,draw) dims {extra_dims}; '
                f'use summarize_array_variable instead.'
            )
        obs_dim = extra_dims[0]
        mean_da = da.mean(dim=("chain", "draw"))
        obs_coord = mean_da.coords.get(obs_dim, None)
        if obs_coord is not None:
            index = pd.Index(np.asarray(obs_coord), name=obs_dim)
        else:
            index = pd.RangeIndex(mean_da.shape[0], name=obs_dim)
        df = pd.DataFrame({var_name: np.asarray(mean_da)}, index=index)
        return df

    def get_beta_values(
        self,
        my_betas: list[str] | None = None,
        *,
        summary: PosteriorSummary = PosteriorSummary.MEAN,
    ) -> dict[str, float]:
        """Retrieve posterior point estimates for a set of parameters.

        :param my_betas: names of requested parameters. If None, all parameters
            are returned.
        :param summary: PosteriorSummary enum specifying whether to return
            the posterior mean or the posterior mode. Default: MEAN.
        """
        the_betas = self.parameter_estimates()

        # Validate requested beta names
        if my_betas is not None:
            unknown = [b for b in my_betas if b not in the_betas]
            if unknown:
                raise BiogemeError(f"Unknown parameter(s): {', '.join(unknown)}")
            selected = {name: the_betas[name] for name in my_betas}
        else:
            selected = the_betas

        # Extract selected summary
        if summary is PosteriorSummary.MEAN:
            extractor = lambda b: b.mean
        elif summary is PosteriorSummary.MODE:
            extractor = lambda b: b.mode
        else:
            raise BiogemeError(f"Invalid posterior summary: {summary!r}")

        return {name: extractor(beta) for name, beta in selected.items()}

    def get_betas_for_sensitivity_analysis(
        self,
        my_betas: list[str] | None = None,
        size: int = 100,
    ) -> list[dict[str, float]]:
        """Generate draws from the distribution of the estimates, for
        sensitivity analysis.

        :param my_betas: names of the parameters for which draws are requested.
        :param size: number of draws. If use_bootstrap is True, the value is
            ignored and a warning is issued. Default: 100.
        :return: list of dict. Each dict has a many entries as parameters.
                The list has as many entries as draws.

        """
        if getattr(self, 'raw_bayesian_results', None) is None:
            raise BiogemeError('No result available')
        if my_betas is None:
            my_betas = self.raw_bayesian_results.beta_names

        # From an ArviZ InferenceData, build a list of dicts mapping each name in `my_betas`
        # to its scalar value for one posterior draw. One dict per returned draw.
        # Variables must be scalar per draw (or broadcastable size-1).
        post = self.idata.posterior
        arrays: dict[str, np.ndarray] = {}
        total_S: int | None = None

        for name in my_betas:
            if name not in post:
                raise KeyError(f'Variable "{name}" not found in posterior.')

            da = post[
                name
            ]  # xarray.DataArray with dims ('chain','draw', ...optional extra)
            vals = da.values  # numpy array
            S = int(da.sizes['chain'] * da.sizes['draw'])

            # Flatten to one scalar per (chain, draw); allow extra dims of size 1
            if vals.size == S:
                vec = vals.reshape(S)
            else:
                other = vals.size // S
                if other != 1:
                    raise ValueError(
                        f"Variable '{name}' is not scalar per draw; has extra size {other}."
                    )
                vec = vals.reshape(S, other)[:, 0]

            arrays[name] = vec.astype(float)
            if total_S is None:
                total_S = S
            elif total_S != S:
                raise ValueError("All variables must have the same number of draws.")

        # Select draw indices with thinning and optional shuffle
        idx = np.arange(0, total_S)
        np.random.shuffle(idx)
        idx = idx[: min(size, idx.size)]
        if size > idx.size:
            error_msg = f'{size} draws are requested for simulation. Only {idx.size} are available from the posterior.'
            logger.warning(error_msg)

        # Assemble one dict per selected draw
        return [{name: float(arrays[name][i]) for name in my_betas} for i in idx]
