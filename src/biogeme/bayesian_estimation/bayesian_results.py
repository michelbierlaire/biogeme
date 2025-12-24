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
from biogeme.exceptions import BiogemeError
from biogeme.tools import timeit
from scipy.stats import gaussian_kde
from tabulate import tabulate

from .raw_bayesian_results import RawBayesianResults

logger = logging.getLogger(__name__)

CHOICE_LABEL = "_choice"


class PosteriorSummary(str, Enum):
    """Type of posterior point estimate to extract."""

    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"


@dataclass
class EstimatedBeta:
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
        'Name': 'Identifier of the model parameter being estimated.',
        'Value': 'Posterior mean (expected value) of the parameter.',
        'Median': 'Posterior median (50% quantile) of the parameter.',
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
        """Build BayesianResults from RawBayesianResults by loading the posterior and computing per-parameter summaries.

        :param raw: Minimal record with posterior InferenceData (`idata`), model/data names, and parameter names.
        :param calculate_likelihood: If True, expose/add the ArviZ `log_likelihood` group and enable predictive criteria.
        :param calculate_waic: If True, compute WAIC (requires `calculate_likelihood=True`).
        :param calculate_loo: If True, compute LOO (requires `calculate_likelihood=True`).
        :param hdi_prob: Credible mass for the Highest Density Interval (e.g., 0.94 or 0.95).
        :param strict: If True, raise when posterior variables have extra dimensions beyond (chain, draw).
        :raises ValueError: If WAIC/LOO are requested without likelihood, or if no scalar posterior variables are found.
        """
        if calculate_waic or calculate_loo:
            if not calculate_likelihood:
                raise ValueError("WAIC/LOO require calculate_likelihood=True.")
        self.raw_bayesian_results = raw
        self.calculate_likelihood = calculate_likelihood
        self.calculate_waic = calculate_waic
        self.calculate_loo = calculate_loo

        self._idata = raw.idata

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
            median = float(np.nanmedian(samples))
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
                median=median,
                mode=mode,
                std_err=std,
                z_value=z_value,
                p_value=p_value,
                hdi_low=hdi_low,
                hdi_high=hdi_high,
                rhat=np.nan,
                effective_sample_size_bulk=np.nan,
                effective_sample_size_tail=np.nan,
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
        self._log_likelihood = None
        self._waic_res = None
        self._loo_res = None
        self._rhat_ds = None
        self._ess_bulk_ds = None
        self._ess_tail_ds = None
        self._diagnostics_computed = False

    def ensure_diagnostics(self) -> None:
        """Compute R-hat and ESS lazily. Cached after first attempt."""
        if getattr(self, "_diagnostics_computed", False):
            return

        import time

        t0 = time.time()

        try:
            self._rhat_ds = az.rhat(self._idata, method="rank")
            self._ess_bulk_ds = az.ess(self._idata, method="bulk")
            self._ess_tail_ds = az.ess(self._idata, method="tail")
            self._diagnostics_error = None
        except (ValueError, TypeError, RuntimeError) as e:
            logger.warning(
                "Diagnostics computation failed (R-hat/ESS). "
                "Diagnostics will not be retried. Error: %s",
                e,
            )
            self._diagnostics_error = e
            self._rhat_ds = None
            self._ess_bulk_ds = None
            self._ess_tail_ds = None
        finally:
            self._diagnostics_computed = True
            elapsed = time.time() - t0
            if elapsed > 2.0:
                logger.info(
                    "Diagnostics computation took %.1f seconds (cached).", elapsed
                )

        # If diagnostics could not be computed, leave NaNs in the EstimatedBetas
        if (
            self._rhat_ds is None
            or self._ess_bulk_ds is None
            or self._ess_tail_ds is None
        ):
            return

        def _scalar_from_ds(ds: xr.Dataset, var_name: str) -> float:
            if var_name not in ds:
                return float("nan")
            return float(np.asarray(ds[var_name]).squeeze())

        for name, beta in self.parameters.items():
            beta.rhat = _scalar_from_ds(self._rhat_ds, name)
            beta.effective_sample_size_bulk = _scalar_from_ds(self._ess_bulk_ds, name)
            beta.effective_sample_size_tail = _scalar_from_ds(self._ess_tail_ds, name)

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

        This uses :meth:`RawBayesianResults.load` under the hood and then computes posterior summaries.

        :param filename: Path to the NetCDF file.
        :param calculate_likelihood: If True, expose/add the ArviZ `log_likelihood` group and enable predictive criteria.
        :param calculate_waic: If True, compute WAIC (requires `calculate_likelihood=True`).
        :param calculate_loo: If True, compute LOO (requires `calculate_likelihood=True`).
        :param hdi_prob: Credible mass for the Highest Density Interval.
        :param strict: If True, raise when posterior variables have extra dimensions beyond (chain, draw).
        :return: A :class:`BayesianResults` instance built from the file.
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

        Delegates to :meth:`RawBayesianResults.save`.

        :param path: Output path for the NetCDF file.
        """
        self.raw_bayesian_results.save(path)

    @staticmethod
    def _two_sided_p_from_posterior(samples: np.ndarray) -> float:
        """Compute a Bayesian two-sided tail probability relative to 0.

        This is defined as ``2 * min(P(theta > 0), P(theta < 0))`` estimated from the posterior draws.

        :param samples: 1D array of posterior draws.
        :return: Two-sided tail probability (NaN if no finite draws are available).
        """
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
        # Ensure log_likelihood group exists
        if self.log_likelihood is None:
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
        self.ensure_diagnostics()
        return az.summary(self.idata)

    @property
    @timeit(label='posterior_predictive_loglike')
    def posterior_predictive_loglike(self) -> float | None:
        """Posterior-predictive log density.

        Computes ``sum_n log(mean_{chain,draw} p(y_n|theta))`` using the log-likelihood draws.
        This is a posterior-predictive criterion (log pointwise predictive density via arithmetic
        averaging over ``theta``); it is **not** the maximum-likelihood log-likelihood.

        :return: Posterior-predictive log density, or None if likelihood was not computed.
        :raises ValueError: If the stored log-likelihood has an unexpected shape.
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
            self._posterior_predictive_loglike = float(logmeanexp)
            return self._posterior_predictive_loglike

        # a.ndim == 3: (chain, draw, obs) -> sum over obs of log-mean-exp across draws
        a_max = np.max(a, axis=(0, 1), keepdims=True)  # (1,1,obs)
        sumexp = np.sum(np.exp(a - a_max), axis=(0, 1))  # (obs,)
        sumexp = np.clip(sumexp, 1e-300, np.inf)
        logmeanexp = np.log(sumexp) - np.log(S) + a_max.squeeze((0, 1))
        self._posterior_predictive_loglike = float(np.sum(logmeanexp))
        return self._posterior_predictive_loglike

    @property
    @timeit(label='expected_log_likelihood')
    def expected_log_likelihood(self) -> float | None:
        """Posterior expectation of the total log-likelihood.

        Computes ``E_theta[ log L(Y|theta) ]`` across posterior draws. For pointwise arrays
        of shape (chain, draw, obs), totals are formed by summing over observations first.

        :return: Expected total log-likelihood, or None if likelihood was not computed.
        :raises ValueError: If the stored log-likelihood has an unexpected shape.
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
    def best_draw_log_likelihood(self) -> float | None:
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
            res = self.waic_res
            self._waic = float(
                getattr(res, "waic", getattr(res, "elpd_waic", float("nan")))
            )
        return self._waic

    @property
    @timeit(label='waic_se')
    def waic_se(self):
        if self.waic_res is None:
            return None
        if self._waic_se is None:
            res = self.waic_res
            self._waic_se = float(
                getattr(res, "waic_se", getattr(res, "se", float("nan")))
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
    def loo(self) -> float | None:
        if self.loo_res is None:
            return None
        if self._loo is None:
            self._loo = float(
                getattr(
                    self.loo_res, "loo", getattr(self.loo_res, "elpd_loo", float("nan"))
                )
            )
        return self._loo

    @property
    @timeit(label='loo_se')
    def loo_se(self):
        if self.loo_res is None:
            return None
        if self._loo_se is None:
            self._loo_se = float(
                getattr(
                    self.loo_res, "loo_se", getattr(self.loo_res, "se", float("nan"))
                )
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
        self.ensure_diagnostics()
        names = set(self.parameters.keys()) & set(self.raw_bayesian_results.beta_names)
        return {k: v for k, v in self.parameters.items() if k in names}

    def other_variables(self) -> dict[str, EstimatedBeta]:
        """Return posterior scalar variables that are *not* listed as parameters.

        Useful to expose derived/deterministic quantities stored in the posterior
        (e.g., total log-likelihood) without mixing them with parameter estimates.
        """
        self.ensure_diagnostics()
        names = set(self.parameters.keys()) - set(self.raw_bayesian_results.beta_names)
        return {k: v for k, v in self.parameters.items() if k in names}

    def list_array_variables(self) -> dict[str, dict]:
        """Return metadata for posterior variables that have extra dims beyond (chain, draw).

        Each entry contains: dims (tuple), shape (tuple), sizes (dict), dtype (str).
        """
        return dict(self.array_metadata)

    def report_stored_variables(self) -> pd.DataFrame:
        """Report all variables stored in the underlying NetCDF/InferenceData.

        This is a convenience method to inspect what PyMC/ArviZ stored in the
        results file. It lists each variable together with its group, dimensions,
        and shape. The dimensions typically include ``chain`` and ``draw`` for
        posterior quantities.

        :return: A DataFrame with columns ``group``, ``variable``, ``dims``, and ``shape``.
        :raises BiogemeError: If the inference data is missing or malformed.
        """
        if getattr(self, "_idata", None) is None:
            raise BiogemeError("No inference data is available.")

        rows: list[dict[str, Any]] = []

        # Iterate over ArviZ groups present in the InferenceData
        for group in getattr(self._idata, "groups", lambda: [])():
            ds = getattr(self._idata, group, None)
            if ds is None:
                continue
            if not isinstance(ds, (xr.Dataset, xr.DataArray)):
                continue

            # ArviZ groups are typically xarray.Dataset
            if isinstance(ds, xr.DataArray):
                data_vars = {ds.name or "<unnamed>": ds}
            else:
                data_vars = dict(ds.data_vars)

            for var_name, da in data_vars.items():
                dims = tuple(str(d) for d in da.dims)
                shape = tuple(int(s) for s in da.shape)
                rows.append(
                    {
                        "group": str(group),
                        "variable": str(var_name),
                        "dims": dims,
                        "shape": shape,
                    }
                )

        if not rows:
            return pd.DataFrame(columns=["group", "variable", "dims", "shape"])

        df = pd.DataFrame(rows)
        # Stable, readable ordering
        df = df.sort_values(["group", "variable"], kind="stable").reset_index(drop=True)
        return df

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
        """Summarize a multi-dimensional posterior variable for selected indices along one extra dimension.

        :param name: Name of the posterior variable to summarize (must be present in `array_metadata`).
        :param dim: Name of the extra dimension along which indices are selected (e.g., an observation dimension).
        :param indices: Indices to summarize. If None, summarize all indices (may be large).
        :param hdi_prob: If provided, overrides the instance `hdi_prob` for this call.
        :return: Mapping ``index -> EstimatedBeta`` computed from samples across chains/draws.
        :raises KeyError: If the variable or dimension is unknown.
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

        for i in idx_list:
            # select along the requested dim
            sub = da.sel({dim: i})
            # ensure scalar after selecting; if still has more dims, skip
            extra_dims = [d for d in sub.dims if d not in ("chain", "draw")]
            if extra_dims:
                continue
            samples = np.asarray(sub).reshape(-1)
            mean = float(np.nanmean(samples))
            median = float(np.nanmedian(samples))
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
                median=median,
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
            the posterior mean, median, or mode. Default: MEAN.
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
        elif summary is PosteriorSummary.MEDIAN:
            extractor = lambda b: b.median
        elif summary is PosteriorSummary.MODE:
            extractor = lambda b: b.mode
        else:
            raise BiogemeError(
                f"Invalid posterior summary: {summary!r}. Valid options are PosteriorSummary.MEAN, PosteriorSummary.MEDIAN, PosteriorSummary.MODE."
            )

        return {name: extractor(beta) for name, beta in selected.items()}

    def get_betas_for_sensitivity_analysis(
        self,
        my_betas: list[str] | None = None,
        size: int = 100,
    ) -> list[dict[str, float]]:
        """Generate draws from the distribution of the estimates, for
        sensitivity analysis.

        :param my_betas: names of the parameters for which draws are requested.
        :param size: number of draws.  Default: 100.
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

    @staticmethod
    def _samples_matrix(
        idata: az.InferenceData,
        *,
        group: str,
        var_names: list[str],
    ) -> np.ndarray:
        """Extract draws as a 2D matrix ``(n_draws, n_vars)`` from an ArviZ InferenceData group.

        Each requested variable must be scalar per draw (extra dimensions of total size 1 are allowed).

        :param idata: InferenceData object holding the requested group.
        :param group: Name of the group to extract from (typically ``"posterior"`` or ``"prior"``).
        :param var_names: Variable names to extract.
        :return: Array of shape ``(S, P)`` where ``S = chains * draws`` and ``P = len(var_names)``.
        :raises KeyError: If the requested group or any variable is missing.
        :raises ValueError: If a variable is not scalar per draw.
        """
        if not hasattr(idata, group):
            raise KeyError(f"InferenceData has no group '{group}'.")
        ds = getattr(idata, group)

        cols: list[np.ndarray] = []
        S_expected: int | None = None

        for name in var_names:
            if name not in ds:
                raise KeyError(f"Variable '{name}' not found in {group}.")
            da = ds[name]
            # Require (chain, draw) only (allow extra dims of size 1)
            vals = np.asarray(da.values)
            if 'chain' not in da.dims or 'draw' not in da.dims:
                raise ValueError(
                    f"Variable '{name}' in {group} must have dims including ('chain','draw'); got {da.dims!r}."
                )
            S = int(da.sizes['chain'] * da.sizes['draw'])
            if S_expected is None:
                S_expected = S
            elif S_expected != S:
                raise ValueError("All variables must have the same number of draws.")

            if vals.size == S:
                vec = vals.reshape(S)
            else:
                other = vals.size // S
                if other != 1:
                    raise ValueError(
                        f"Variable '{name}' in {group} is not scalar per draw; extra size is {other}."
                    )
                vec = vals.reshape(S, other)[:, 0]

            cols.append(vec.astype(float))

        X = np.column_stack(cols)
        return X

    @staticmethod
    def _cov_eigen_diagnostics(cov: np.ndarray) -> dict[str, float]:
        """Compute eigen-structure diagnostics for a covariance matrix.

        The returned values are scalar summaries. Use :meth:`_near_null_direction` to obtain an
        interpretable eigenvector (labelled by variable names) when a near-zero variance direction
        is detected.

        :param cov: Covariance matrix (P x P).
        :return: Dictionary with effective rank, eigenvalue extrema and a simple condition number.
        """
        # Symmetrize defensively
        C = 0.5 * (cov + cov.T)
        try:
            eigvals = np.linalg.eigvalsh(C)
        except np.linalg.LinAlgError:
            eigvals = np.linalg.eigvals(C).real
        eigvals = np.asarray(eigvals, dtype=float)
        eigvals_sorted = np.sort(eigvals)
        # Numerical thresholds: treat tiny/negative as ~0
        eps = float(np.finfo(float).eps)
        scale = float(np.max(np.abs(eigvals_sorted))) if eigvals_sorted.size else 0.0
        tol = max(1e-12, 1e3 * eps * max(1.0, scale))
        positive = eigvals_sorted[eigvals_sorted > tol]
        effective_rank = float(positive.size)
        min_pos = float(np.min(positive)) if positive.size else 0.0
        max_pos = float(np.max(positive)) if positive.size else 0.0
        cond = float(max_pos / min_pos) if (min_pos > 0.0) else float('inf')
        return {
            'effective_rank': effective_rank,
            'min_eigenvalue': (
                float(eigvals_sorted[0]) if eigvals_sorted.size else float('nan')
            ),
            'max_eigenvalue': (
                float(eigvals_sorted[-1]) if eigvals_sorted.size else float('nan')
            ),
            'min_positive_eigenvalue': min_pos,
            'condition_number': cond,
        }

    @staticmethod
    def _near_null_direction(
        cov: np.ndarray,
        *,
        var_names: list[str],
        tol_ratio: float,
        max_terms: int = 8,
    ) -> dict[str, Any] | None:
        """Return an interpretable weak-identification direction from a covariance matrix.

        In maximum likelihood, identification problems are typically detected from the Hessian/
        information matrix: a *small* eigenvalue indicates a nearly flat (unidentified) direction.

        Here we work with the **posterior covariance** instead. A weakly identified direction then
        corresponds to a **large posterior variance** direction, i.e. the eigenvector associated
        with the **largest** covariance eigenvalue.

        We report such a direction when the covariance is highly anisotropic, using the ML-like
        criterion translated to covariance:

        - ML trigger:      min(H) <= tol_ratio * max(H)
        - Covariance analog: max(Cov) >= (1 / tol_ratio) * min_positive(Cov)

        The returned vector is normalized to unit Euclidean norm. For readability, the result
        includes the largest absolute loadings (with signs) labelled by parameter names.

        :param cov: Covariance matrix (P x P).
        :param var_names: Names of the P variables (same ordering as `cov`).
        :param tol_ratio: Trigger threshold (same interpretation as ML). Smaller values are
            stricter. A typical value is e.g. 1e-5.
        :param max_terms: Maximum number of largest-magnitude coefficients to report.
        :return: None if no weak-identification direction is detected; otherwise a dict with keys
            ``eigenvalue`` (largest variance), ``ratio_to_min_positive`` (anisotropy),
            ``vector`` (full mapping) and ``top_loadings``.
        :raises ValueError: If `var_names` length does not match the covariance dimension.
        """
        C = 0.5 * (cov + cov.T)
        P = int(C.shape[0])
        if P == 0:
            return None
        if len(var_names) != P:
            raise ValueError(
                f"var_names length ({len(var_names)}) does not match covariance size ({P})."
            )

        # Eigen-decomposition (ascending eigenvalues); eigenvectors are columns.
        try:
            eigvals, eigvecs = np.linalg.eigh(C)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eig(C)
            eigvals = np.asarray(eigvals, dtype=float).real
            eigvecs = np.asarray(eigvecs, dtype=float).real
            order = np.argsort(eigvals)
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

        eigvals = np.asarray(eigvals, dtype=float)
        if eigvals.size == 0:
            return None

        # Determine smallest *positive* eigenvalue to avoid dividing by ~0 due to numerical noise.
        eps = float(np.finfo(float).eps)
        vmax = float(np.max(np.abs(eigvals)))
        # scale-aware tolerance for considering an eigenvalue "positive"
        tol = max(1e-12, 1e3 * eps * max(1.0, vmax))
        positive = eigvals[eigvals > tol]
        if positive.size == 0:
            return None

        vmin_pos = float(np.min(positive))
        vmax_pos = float(np.max(positive))
        if not np.isfinite(vmin_pos) or not np.isfinite(vmax_pos) or vmax_pos <= 0.0:
            return None

        # ML-like trigger translated to covariance: huge max variance relative to min positive variance
        ratio = float(vmax_pos / vmin_pos) if vmin_pos > 0.0 else float("inf")
        if ratio < float(1.0 / max(tol_ratio, 1e-300)):
            return None

        # Take eigenvector associated with the largest eigenvalue (largest variance direction).
        idx_max = int(np.argmax(eigvals))
        v = np.asarray(eigvecs[:, idx_max], dtype=float).reshape(-1)
        nrm = float(np.linalg.norm(v))
        if nrm > 0.0:
            v = v / nrm

        full = {name: float(v[i]) for i, name in enumerate(var_names)}
        order = np.argsort(np.abs(v))[::-1]
        top = [(var_names[i], float(v[i])) for i in order[:max_terms]]

        return {
            "eigenvalue": float(eigvals[idx_max]),
            "ratio_to_min_positive": ratio,
            "vector": full,
            "top_loadings": top,
        }

    def identification_diagnostics(
        self,
        *,
        identification_threshold: float,
        prior_idata: az.InferenceData | None = None,
        var_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute heuristic diagnostics for potential identification issues.

        Designed for the workflow where a posterior :class:`arviz.InferenceData` is available and
        an optional `prior_idata` is produced via
        ``pm.sample_prior_predictive(..., return_inferencedata=True)``.

        If `prior_idata` is provided, it is merged into the stored InferenceData using
        ``idata.extend(prior_idata)`` so the resulting NetCDF can contain both posterior and prior groups.

        The diagnostics are heuristics (not proofs):

        - Eigen-structure of the posterior covariance (near-zero eigenvalues / large condition number)
          can indicate weak or non-identification.
        - Comparing posterior vs prior marginal scales highlights parameters that may be largely
          "identified by the prior" (posterior std close to prior std).

        :param prior_idata: Optional prior InferenceData to merge before computing diagnostics.
        :param var_names: Variables to analyze. If None, uses `raw_bayesian_results.beta_names`
            filtered to scalar variables present in the posterior.
        :return: Dictionary with keys ``has_prior``, ``posterior_cov``, ``prior_cov``,
            ``per_parameter`` (DataFrame), ``flags`` (list of strings), and (if detected)
            ``posterior_near_null_direction`` / ``prior_near_null_direction``.
        """
        idata = self._idata
        if prior_idata is not None:
            try:
                idata.extend(prior_idata)
            except Exception as e:  # pragma: no cover
                logger.warning("Could not extend InferenceData with prior group: %s", e)

        # Select variables
        if var_names is None:
            candidates = list(getattr(self.raw_bayesian_results, 'beta_names', []))
            # keep only scalar posterior vars that we summarized
            var_names = [n for n in candidates if n in self.parameters]
            if not var_names:
                # fallback: all scalar posterior vars
                var_names = list(self.parameters.keys())

        flags: list[str] = []
        prior_null_direction = None

        # Posterior matrix / covariance
        X_post = self._samples_matrix(idata, group='posterior', var_names=var_names)
        cov_post = np.cov(X_post, rowvar=False, ddof=1)
        post_diag = self._cov_eigen_diagnostics(cov_post)

        # If the posterior covariance shows extreme anisotropy, report the largest-variance
        # direction as a named linear combination to help diagnose overspecification.
        # (This is the covariance analogue of a near-zero Hessian eigenvalue in ML.)
        post_null_direction = self._near_null_direction(
            cov_post,
            var_names=var_names,
            tol_ratio=identification_threshold,
            max_terms=8,
        )
        if post_null_direction is not None:
            top = post_null_direction["top_loadings"]
            human = " + ".join(
                [f"{coef:+.3g}·{nm}" for nm, coef in top if np.isfinite(coef)]
            )
            flags.append(
                "Weak-identification direction detected from the posterior covariance (largest posterior variance direction). "
                "This suggests a linear combination of parameters that remains weakly constrained. "
                f"Top loadings: {human}"
            )

        # Per-parameter scale diagnostics (posterior)
        post_std = np.sqrt(np.diag(cov_post))
        post_std = np.asarray(post_std, dtype=float)

        has_prior = (
            hasattr(idata, 'prior') and getattr(idata, 'prior', None) is not None
        )
        prior_diag: dict[str, float] | None = None
        prior_std: np.ndarray | None = None

        if has_prior:
            try:
                X_prior = self._samples_matrix(
                    idata, group='prior', var_names=var_names
                )
                cov_prior = np.cov(X_prior, rowvar=False, ddof=1)
                prior_diag = self._cov_eigen_diagnostics(cov_prior)
                prior_std = np.sqrt(np.diag(cov_prior)).astype(float)
                prior_null_direction = self._near_null_direction(
                    cov_prior,
                    var_names=var_names,
                    tol_ratio=identification_threshold,
                    max_terms=8,
                )
            except Exception as e:
                has_prior = False
                prior_diag = None
                prior_std = None
                prior_null_direction = None
                logger.warning("Prior group present but could not be analyzed: %s", e)

        # Build per-parameter table
        df = pd.DataFrame({'name': var_names, 'posterior_std': post_std})
        if prior_std is not None:
            df['prior_std'] = prior_std
            with np.errstate(divide='ignore', invalid='ignore'):
                df['std_ratio_post_over_prior'] = df['posterior_std'] / df['prior_std']
        else:
            df['prior_std'] = np.nan
            df['std_ratio_post_over_prior'] = np.nan

        # Heuristic flags
        if (
            np.isfinite(post_diag.get('condition_number', np.nan))
            and post_diag['condition_number'] > 1e10
        ):
            flags.append(
                "Posterior covariance is extremely ill-conditioned (condition number > 1e10); this can indicate weak/non-identification."
            )
        if post_diag.get('min_positive_eigenvalue', 0.0) <= 0.0:
            flags.append(
                "Posterior covariance appears rank-deficient (no strictly positive eigenvalues above tolerance); this strongly suggests non-identification or severe collinearity."
            )

        if prior_std is not None:
            # Identify parameters whose marginal scale barely changed from prior
            close = df['std_ratio_post_over_prior']
            # ratio near 1 means the likelihood barely informed the parameter
            near_one = close[np.isfinite(close) & (close > 0.8)]
            if len(near_one) > 0:
                worst = df.loc[near_one.index, 'name'].tolist()[:10]
                flags.append(
                    "Some parameters have posterior std close to prior std (ratio > 0.8), suggesting they may be largely identified by the prior: "
                    + ", ".join(worst)
                    + ("" if len(worst) < 10 else ", ...")
                )

        return {
            "has_prior": bool(prior_std is not None),
            "posterior_cov": post_diag,
            "prior_cov": prior_diag,
            "per_parameter": df.reset_index(drop=True),
            "flags": flags,
            "posterior_near_null_direction": post_null_direction,
            "prior_near_null_direction": (
                prior_null_direction if prior_std is not None else None
            ),
        }
