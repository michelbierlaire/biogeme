# test_bayesian_results.py

from __future__ import annotations

from datetime import timedelta
from typing import Dict, List

import arviz as az
import numpy as np
import pandas as pd
import pytest
from biogeme.bayesian_estimation import RawBayesianResults, bayesian_results as br_mod
from biogeme.bayesian_estimation.bayesian_results import (
    BayesianResults,
    CHOICE_LABEL,
    EstimatedBeta,
    PosteriorSummary,
    _posterior_mode_kde,
)
from biogeme.exceptions import BiogemeError


class DummyRawBayesianResults:
    """Minimal stand-in for RawBayesianResults, suitable for testing BayesianResults."""

    def __init__(
        self, idata: az.InferenceData, beta_names: List[str], log_like_name: str
    ):
        self.idata = idata
        self.model_name = "test_model"
        self.data_name = "test_data"
        self.chains = int(idata.posterior.dims.get("chain", 0))
        self.draws = int(idata.posterior.dims.get("draw", 0))
        self.log_like_name = log_like_name
        self.beta_names = list(beta_names)
        self.sampler = "NUTS"
        self.target_accept = 0.9
        self.run_time = timedelta(seconds=12.3)

        # Infer number of observations from the log-likelihood variable if present
        ll = idata.posterior.get(log_like_name, None)
        if ll is not None and ll.ndim == 3:
            self.number_of_observations = int(ll.sizes[list(ll.dims)[-1]])
        else:
            self.number_of_observations = 0

        # Flags used in other parts of Biogeme; harmless here
        self.log_like_name = log_like_name

        # To verify dump()
        self._saved_paths: List[str] = []

    def save(self, path: str) -> None:
        """Record where save() was called, for testing dump()."""
        self._saved_paths.append(path)


@pytest.fixture
def simple_idata() -> az.InferenceData:
    """Small synthetic InferenceData with scalar and array variables and log-likelihood."""
    rng = np.random.default_rng(12345)
    chains, draws, obs, arr_dim = 2, 5, 7, 4

    # Posterior scalars (per draw)
    beta1 = rng.normal(loc=0.5, scale=0.1, size=(chains, draws))
    beta2 = rng.normal(loc=-1.0, scale=0.2, size=(chains, draws))

    # Array variable (acts like observation-level parameter)
    array_var = rng.normal(size=(chains, draws, arr_dim))

    # Log-likelihood per observation
    log_like = rng.normal(loc=-1.0, scale=0.5, size=(chains, draws, obs))

    idata = az.from_dict(
        posterior={
            "beta1": beta1,
            "beta2": beta2,
            "array_var": array_var,
            "log_like": log_like,
        }
    )
    return idata


@pytest.fixture
def dummy_raw(simple_idata: az.InferenceData) -> RawBayesianResults:
    """RawBayesianResults object with beta1 and beta2 as parameters."""
    # infer number_of_observations from the log_like variable
    log_like = simple_idata.posterior["log_like"]
    n_obs = int(log_like.sizes[list(log_like.dims)[-1]])

    return RawBayesianResults(
        idata=simple_idata,
        model_name="test_model",
        log_like_name="log_like",
        number_of_observations=n_obs,
        user_notes="",
        data_name="test_data",
        beta_names=["beta1", "beta2"],
        sampler="NUTS",
        target_accept=0.9,
        run_time=timedelta(seconds=12.3),
    )


@pytest.fixture
def bayes_results(dummy_raw: DummyRawBayesianResults) -> BayesianResults:
    """A fully configured BayesianResults with all criteria enabled."""
    return BayesianResults(
        raw=dummy_raw,
        calculate_likelihood=True,
        calculate_waic=True,
        calculate_loo=True,
        hdi_prob=0.94,
        strict=False,
    )


def test_posterior_mode_kde_basic() -> None:
    """_posterior_mode_kde should approximate the mode of a simple normal sample."""
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=1.23, scale=0.5, size=5_000)
    mode_est = _posterior_mode_kde(samples)
    assert np.isfinite(mode_est)
    # Mode of a Gaussian is its mean
    assert pytest.approx(1.23, abs=0.1) == mode_est


def test_posterior_mode_kde_all_nan_returns_nan() -> None:
    samples = np.array([np.nan, np.nan])
    mode_est = _posterior_mode_kde(samples)
    assert np.isnan(mode_est)


def test_constructor_builds_parameter_summaries(bayes_results: BayesianResults) -> None:
    """BayesianResults should collect scalar posterior vars into EstimatedBeta summaries."""
    params: Dict[str, EstimatedBeta] = bayes_results.parameters
    assert set(params.keys()) >= {"beta1", "beta2"}
    for name, beta in params.items():
        assert beta.name == name
        assert np.isfinite(beta.mean)
        assert np.isfinite(beta.mode) or np.isnan(beta.mode)
        assert beta.std_err >= 0.0 or np.isnan(beta.std_err)
        assert beta.rhat > 0 or np.isnan(beta.rhat)
        assert beta.effective_sample_size_bulk > 0 or np.isnan(
            beta.effective_sample_size_bulk
        )
        assert beta.effective_sample_size_tail > 0 or np.isnan(
            beta.effective_sample_size_tail
        )


def test_array_metadata_collected_for_extra_dims(
    bayes_results: BayesianResults,
) -> None:
    """Variables with extra dims should be recorded in array_metadata."""
    meta = bayes_results.list_array_variables()
    assert "array_var" in meta
    info = meta["array_var"]
    assert "dims" in info and "shape" in info and "sizes" in info and "dtype" in info
    # array_var has exactly one extra dimension beyond chain/draw
    assert "array_var_dim_0" in info["sizes"]
    assert info["sizes"]["array_var_dim_0"] > 0


def test_parameter_and_other_variables_split(bayes_results: BayesianResults) -> None:
    """parameter_estimates vs other_variables split according to beta_names."""
    param_est = bayes_results.parameter_estimates()
    other = bayes_results.other_variables()

    # beta1, beta2 are parameters
    assert set(param_est.keys()) >= {"beta1", "beta2"}
    # array_var is multi-dim; not in parameters
    assert "array_var" not in param_est
    # other_variables excludes parameters
    assert "beta1" not in other
    assert "beta2" not in other


def test_two_sided_p_from_posterior() -> None:
    """Check basic behavior of the Bayesian two-sided p-value helper."""
    s = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    p = BayesianResults._two_sided_p_from_posterior(s)
    # For this symmetric sample, tails are 2/5 => two-sided p = 0.8
    assert pytest.approx(0.8, rel=1e-12, abs=1e-12) == p


def test_log_likelihood_property_adds_group(bayes_results: BayesianResults) -> None:
    """Accessing log_likelihood should also create the log_likelihood group with CHOICE_LABEL."""
    ll = bayes_results.log_likelihood
    assert ll is not None

    # After property access, group must exist
    assert "log_likelihood" in bayes_results.idata.groups()
    ds = bayes_results.idata.log_likelihood
    assert CHOICE_LABEL in ds.data_vars


def test_posterior_predictive_and_expected_loglike(
    bayes_results: BayesianResults,
) -> None:
    """posterior_predictive_loglike and expected_log_likelihood should be finite."""
    ppl = bayes_results.posterior_predictive_loglike
    ell = bayes_results.expected_log_likelihood
    assert np.isfinite(ppl)
    assert np.isfinite(ell)


def test_best_draw_log_likelihood(bayes_results: BayesianResults) -> None:
    best = bayes_results.best_draw_log_likelihood
    assert np.isfinite(best)


def test_waic_and_loo_properties(bayes_results: BayesianResults) -> None:
    """WAIC/LOO and their SE/p should all be finite when enabled."""

    # Ensure log_likelihood group exists in idata for ArviZ
    _ = bayes_results.log_likelihood

    assert np.isfinite(bayes_results.waic)
    assert np.isfinite(bayes_results.waic_se)
    assert np.isfinite(bayes_results.p_waic)

    assert np.isfinite(bayes_results.loo)
    assert np.isfinite(bayes_results.loo_se)
    assert np.isfinite(bayes_results.p_loo)


def test_disable_likelihood_waic_loo(simple_idata: az.InferenceData) -> None:
    """If calculate_* flags are False, the corresponding properties must be None."""
    raw = DummyRawBayesianResults(simple_idata, ["beta1"], log_like_name="log_like")
    br = BayesianResults(
        raw=raw,
        calculate_likelihood=False,
        calculate_waic=False,
        calculate_loo=False,
        hdi_prob=0.94,
        strict=False,
    )
    assert br.log_likelihood is None
    assert br.waic is None
    assert br.loo is None


def test_posterior_draws_property(
    bayes_results: BayesianResults, dummy_raw: DummyRawBayesianResults
) -> None:
    """posterior_draws = chains * draws."""
    assert bayes_results.posterior_draws == dummy_raw.chains * dummy_raw.draws


def test_from_netcdf_uses_raw_loader(
    monkeypatch, simple_idata: az.InferenceData
) -> None:
    """from_netcdf should call RawBayesianResults.load and then construct BayesianResults."""

    dummy = DummyRawBayesianResults(
        simple_idata, ["beta1", "beta2"], log_like_name="log_like"
    )

    def fake_load(cls, filename: str) -> DummyRawBayesianResults:  # type: ignore[override]
        assert filename == "dummy_file.nc"
        return dummy

    # Monkeypatch RawBayesianResults.load in the module under test
    monkeypatch.setattr(br_mod.RawBayesianResults, "load", classmethod(fake_load))

    br = BayesianResults.from_netcdf("dummy_file.nc")
    assert isinstance(br, BayesianResults)
    assert br.data_name == "test_data"
    assert set(br.parameters.keys()) >= {"beta1", "beta2"}


def test_arviz_summary_returns_dataframe(bayes_results: BayesianResults) -> None:
    df = bayes_results.arviz_summary()
    assert isinstance(df, pd.DataFrame)
    assert "mean" in df.columns


def test_generate_general_information_and_short_summary(
    bayes_results: BayesianResults,
) -> None:
    info = bayes_results.generate_general_information()
    # Check some keys
    assert "Sample size" in info
    assert "Number of chains" in info
    assert "Posterior predictive log-likelihood (sum of log mean p)" in info

    summary_str = bayes_results.short_summary()
    assert isinstance(summary_str, str)
    # tabulate produces lines with 'Sample size' in them
    assert "Sample size" in summary_str


def test_summarize_array_variable_basic(bayes_results: BayesianResults) -> None:
    meta = bayes_results.list_array_variables()
    assert "array_var" in meta
    dim_name = list(meta["array_var"]["sizes"].keys())[0]  # e.g. 'array_var_dim_0'

    # Summarize indices 0 and 2
    summaries = bayes_results.summarize_array_variable(
        "array_var", dim=dim_name, indices=[0, 2]
    )
    assert set(summaries.keys()) == {0, 2}
    for idx, eb in summaries.items():
        assert eb.name.startswith("array_var[")
        assert np.isfinite(eb.mean)
        assert np.isfinite(eb.mode) or np.isnan(eb.mode)


def test_summarize_array_variable_errors(bayes_results: BayesianResults) -> None:
    meta = bayes_results.list_array_variables()
    dim_name = list(meta["array_var"]["sizes"].keys())[0]

    with pytest.raises(KeyError):
        bayes_results.summarize_array_variable("unknown_var", dim=dim_name)

    with pytest.raises(KeyError):
        bayes_results.summarize_array_variable("array_var", dim="unknown_dim")


def test_posterior_mean_by_observation(simple_idata: az.InferenceData) -> None:
    """posterior_mean_by_observation should compute per-observation means for array_var."""

    raw = DummyRawBayesianResults(
        simple_idata, ["beta1", "beta2"], log_like_name="log_like"
    )
    br = BayesianResults(
        raw=raw,
        calculate_likelihood=True,
        calculate_waic=False,
        calculate_loo=False,
        hdi_prob=0.94,
        strict=False,
    )

    meta = br.list_array_variables()
    dim_name = list(meta["array_var"]["sizes"].keys())[0]

    df = br.posterior_mean_by_observation("array_var")
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == meta["array_var"]["sizes"][dim_name]
    assert "array_var" in df.columns

    # Compare with manual mean over chain, draw
    da = br.idata.posterior["array_var"]
    manual_mean = np.asarray(da.mean(dim=("chain", "draw")))
    assert np.allclose(df["array_var"].values, manual_mean)


def test_posterior_mean_by_observation_errors(bayes_results: BayesianResults) -> None:
    with pytest.raises(BiogemeError):
        bayes_results.posterior_mean_by_observation("unknown_var")


def test_get_beta_values_mean_and_mode(bayes_results: BayesianResults) -> None:
    # Default: all parameters, mean
    betas_mean = bayes_results.get_beta_values()
    assert set(betas_mean.keys()) >= {"beta1", "beta2"}

    # Specific subset, mode
    betas_mode = bayes_results.get_beta_values(
        my_betas=["beta1"], summary=PosteriorSummary.MODE
    )
    assert list(betas_mode.keys()) == ["beta1"]
    # mode and mean need not be equal but should be finite
    assert np.isfinite(list(betas_mode.values())[0])


def test_get_beta_values_unknown_parameter_raises(
    bayes_results: BayesianResults,
) -> None:
    with pytest.raises(BiogemeError):
        bayes_results.get_beta_values(my_betas=["unknown_param"])


def test_get_betas_for_sensitivity_analysis(bayes_results: BayesianResults) -> None:
    draws = bayes_results.get_betas_for_sensitivity_analysis(size=10)
    assert isinstance(draws, list)
    assert 1 <= len(draws) <= 10  # may be less if posterior has fewer draws
    for d in draws:
        assert "beta1" in d and "beta2" in d
        assert isinstance(d["beta1"], float)


def test_get_betas_for_sensitivity_analysis_unknown_variable_ignored(
    bayes_results: BayesianResults,
) -> None:
    # Temporarily modify beta_names to include a non-existent variable
    bayes_results.raw_bayesian_results.beta_names.append("does_not_exist")

    draws = bayes_results.get_betas_for_sensitivity_analysis(size=5)
    assert isinstance(draws, list)
    assert len(draws) > 0

    for d in draws:
        # Known parameters still present
        assert "beta1" in d and "beta2" in d
        # Unknown parameter is simply ignored
        assert "does_not_exist" not in d


def test___getattr___forwards_to_raw(dummy_raw: DummyRawBayesianResults) -> None:
    br = BayesianResults(
        raw=dummy_raw,
        calculate_likelihood=False,
        calculate_waic=False,
        calculate_loo=False,
        hdi_prob=0.94,
        strict=False,
    )
    # number_of_observations is defined on DummyRawBayesianResults
    assert br.number_of_observations == dummy_raw.number_of_observations


def test_posterior_summary_enum_values() -> None:
    assert PosteriorSummary.MEAN.name == "MEAN"
    assert PosteriorSummary.MODE.name == "MODE"
