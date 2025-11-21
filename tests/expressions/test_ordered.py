# tests/test_ordered_models.py
import math

import numpy as np
import pandas as pd
import pymc as pm
import pytensor as pt
import pytest
from scipy import special as npspecial

from biogeme.expressions import (
    Beta,
    OrderedLogLogit,
    OrderedLogProbit,
    OrderedLogit,
    OrderedProbit,
    Variable,
)


def logistic_cdf(z: float) -> float:
    # numerically stable sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def normal_cdf(z: float) -> float:
    # standard normal CDF via numpy.special.erf (aligns with PyTensor backend)
    return 0.5 * (1.0 + npspecial.erf(z / np.sqrt(2.0)))


def _expected_p_row_logit(row, beta_vals, taus):
    eta = beta_vals["income"] * row["Income"] + beta_vals["age"] * row["Age"]
    y = row["Satisfaction"]
    if y == 99:
        return 1.0
    probs = ordered_prob_from_cdf(eta, taus, logistic_cdf)
    k = int(y) - 1
    return max(min(probs[k], 1 - 1e-12), 1e-12)


def _expected_p_row_probit(row, beta_vals, taus):
    eta = beta_vals["income"] * row["Income"] + beta_vals["age"] * row["Age"]
    y = row["Satisfaction"]
    if y == 99:
        return 1.0
    probs = ordered_prob_from_cdf(eta, taus, normal_cdf)
    k = int(y) - 1
    return max(min(probs[k], 1 - 1e-12), 1e-12)


def ordered_prob_from_cdf(eta: float, taus: list[float], cdf) -> list[float]:
    """
    Given eta, increasing cutpoints taus (length K-1), and a CDF function,
    return probabilities for K ordered categories: P(y=1),...,P(y=K).
    """
    K = len(taus) + 1
    if K == 1:
        return [1.0]
    p0 = cdf(taus[0] - eta)
    mids = [cdf(taus[k] - eta) - cdf(taus[k - 1] - eta) for k in range(1, K - 1)]
    pK = 1.0 - cdf(taus[-1] - eta)
    return [p0, *mids, pK]


def softplus_math(x: float) -> float:
    # numerically stable softplus for scalars
    if x > 20:
        return x  # exp(x) dominates
    if x < -20:
        return math.exp(x)  # log1p(exp(x)) ~ exp(x)
    return math.log1p(math.exp(x))


def enforce_softplus_taus(taus: list[float]) -> list[float]:
    """
    Apply the same softplus-increment reparameterization used by the model when
    enforce_order=True. Input is a list of raw, ordered (K-1) thresholds.
    """
    if not taus:
        return []
    out = [taus[0]]
    for i in range(1, len(taus)):
        d = taus[i] - taus[i - 1]
        out.append(out[-1] + softplus_math(d))
    return out


@pytest.fixture
def tiny_df():
    # Two valid responses and one neutral/skip label (99)
    return pd.DataFrame(
        {
            "Income": [2.0, 1.0, 3.0],
            "Age": [0.5, -1.0, 1.5],
            "Satisfaction": [1.0, 3.0, 99.0],  # 99 is neutral
        }
    )


@pytest.fixture
def expressions():
    # Fixed betas for determinism in tests (status=1 => fixed)
    beta_income = Beta("beta_income", 0.8, None, None, status=1)
    beta_age = Beta("beta_age", -0.5, None, None, status=1)

    # Four cutpoints for K=5 categories; strictly increasing
    tau1 = Beta("tau1", -1.0, None, None, status=1)
    tau2 = Beta("tau2", 0.0, None, None, status=1)
    tau3 = Beta("tau3", 1.0, None, None, status=1)
    tau4 = Beta("tau4", 2.0, None, None, status=1)

    income = Variable("Income")
    income.specific_id = 0
    age = Variable("Age")
    age.specific_id = 1
    y = Variable("Satisfaction")
    y.specific_id = 2

    eta = beta_income * income + beta_age * age
    cutpoints = [tau1, tau2, tau3, tau4]

    categories = [1, 2, 3, 4, 5]
    neutral_labels = [99]  # row 3 will be neutral

    return dict(
        eta=eta,
        cutpoints=cutpoints,
        y=y,
        categories=categories,
        neutral_labels=neutral_labels,
        taus=[-1.0, 0.0, 1.0, 2.0],
        beta_vals=dict(income=0.8, age=-0.5),
    )


def _expected_ll_row_logit(row, beta_vals, taus):
    eta = beta_vals["income"] * row["Income"] + beta_vals["age"] * row["Age"]
    y = row["Satisfaction"]
    if y == 99:
        return 0.0
    probs = ordered_prob_from_cdf(eta, taus, logistic_cdf)
    k = int(y) - 1  # categories are [1..5]
    p = max(min(probs[k], 1 - 1e-12), 1e-12)  # clamp like code
    return math.log(p)


def _expected_ll_row_probit(row, beta_vals, taus):
    eta = beta_vals["income"] * row["Income"] + beta_vals["age"] * row["Age"]
    y = row["Satisfaction"]
    if y == 99:
        return 0.0
    probs = ordered_prob_from_cdf(eta, taus, normal_cdf)
    k = int(y) - 1
    p = max(min(probs[k], 1 - 1e-12), 1e-12)
    return math.log(p)


def _eval_pt_vector_from_builder(tensor_expr):
    # Compile a no-input function: all data nodes are captured inside the builder
    f = pt.function([], tensor_expr)
    return np.asarray(f(), dtype=np.float64)


def _eval_jax_row(jax_fn, row):
    # Parameters, draws, and random variables are unused because betas are fixed.
    params = np.array([], dtype=float)
    # JAX layer indexes variables by their numeric `specific_id` -> pass a dense vector
    one_row = row.to_numpy(dtype=float)
    draws = np.array([], dtype=float)
    rvars = np.array([], dtype=float)
    return float(jax_fn(params, one_row, draws, rvars))


@pytest.mark.parametrize(
    "model_cls, exp_row_fn",
    [
        (OrderedLogit, _expected_p_row_logit),
        (OrderedProbit, _expected_p_row_probit),
    ],
)
@pytest.mark.parametrize("enforce", [False, True])
def test_ordered_models_prob_pt_and_jax_agree(
    tiny_df, expressions, model_cls, exp_row_fn, enforce
):
    # Build the model
    model = model_cls(
        eta=expressions["eta"],
        cutpoints=expressions["cutpoints"],
        y=expressions["y"],
        categories=expressions["categories"],
        neutral_labels=expressions["neutral_labels"],
        enforce_order=enforce,
        eps=1e-12,
    )

    # ---------- PyTensor path ----------
    with pm.Model():
        pt_builder = model.recursive_construct_pymc_model_builder()
        p_tensor = pt_builder(tiny_df)  # shape (N,)
        p_pt = _eval_pt_vector_from_builder(p_tensor)  # numpy array, shape (N,)

    # ---------- JAX path ----------
    jax_builder = model.recursive_construct_jax_function(numerically_safe=True)
    p_jax = np.array(
        [_eval_jax_row(jax_builder, tiny_df.iloc[i]) for i in range(len(tiny_df))]
    )

    # ---------- Expected values by direct math ----------
    taus = expressions["taus"]
    if enforce:
        taus = enforce_softplus_taus(taus)
    expected = np.array(
        [
            exp_row_fn(tiny_df.iloc[i], expressions["beta_vals"], taus)
            for i in range(len(tiny_df))
        ]
    )

    # Sanity: neutral label returns 1.0 for probability
    assert expected[-1] == pytest.approx(1.0, abs=1e-14)

    # Compare PyTensor vs expected
    np.testing.assert_allclose(p_pt, expected, rtol=1e-9, atol=2e-7)

    # Compare JAX vs expected
    np.testing.assert_allclose(p_jax, expected, rtol=1e-9, atol=2e-7)

    # Check consistency between backends
    np.testing.assert_allclose(p_pt, p_jax, rtol=1e-9, atol=2e-7)


@pytest.mark.parametrize(
    "model_cls, exp_row_fn",
    [
        (OrderedLogLogit, _expected_ll_row_logit),
        (OrderedLogProbit, _expected_ll_row_probit),
    ],
)
@pytest.mark.parametrize("enforce", [False, True])
def test_ordered_log_models_ll_pt_and_jax_agree(
    tiny_df, expressions, model_cls, exp_row_fn, enforce
):
    model = model_cls(
        eta=expressions["eta"],
        cutpoints=expressions["cutpoints"],
        y=expressions["y"],
        categories=expressions["categories"],
        neutral_labels=expressions["neutral_labels"],
        enforce_order=enforce,
        eps=1e-12,
    )
    with pm.Model():
        pt_builder = model.recursive_construct_pymc_model_builder()
        ll_pt = _eval_pt_vector_from_builder(pt_builder(tiny_df))
    jax_builder = model.recursive_construct_jax_function(numerically_safe=True)
    ll_jax = np.array(
        [_eval_jax_row(jax_builder, tiny_df.iloc[i]) for i in range(len(tiny_df))]
    )

    taus = expressions["taus"]
    if enforce:
        taus = enforce_softplus_taus(taus)
    expected = np.array(
        [
            exp_row_fn(tiny_df.iloc[i], expressions["beta_vals"], taus)
            for i in range(len(tiny_df))
        ]
    )

    assert expected[-1] == pytest.approx(0.0, abs=1e-14)
    np.testing.assert_allclose(ll_pt, expected, rtol=1e-9, atol=2e-7)
    np.testing.assert_allclose(ll_jax, expected, rtol=1e-9, atol=2e-7)
    np.testing.assert_allclose(ll_pt, ll_jax, rtol=1e-9, atol=2e-7)


@pytest.mark.parametrize("model_cls", [OrderedLogit, OrderedProbit])
@pytest.mark.parametrize("enforce", [False, True])
def test_backend_consistency_over_enforcement_prob(
    tiny_df, expressions, model_cls, enforce
):
    model = model_cls(
        eta=expressions["eta"],
        cutpoints=expressions["cutpoints"],
        y=expressions["y"],
        categories=expressions["categories"],
        neutral_labels=expressions["neutral_labels"],
        enforce_order=enforce,
        eps=1e-12,
    )
    with pm.Model():
        pt_builder = model.recursive_construct_pymc_model_builder()
        p_pt = np.asarray(pt.function([], pt_builder(tiny_df))())
    jax_builder = model.recursive_construct_jax_function(numerically_safe=True)
    p_jax = np.array(
        [_eval_jax_row(jax_builder, tiny_df.iloc[i]) for i in range(len(tiny_df))]
    )
    np.testing.assert_allclose(p_pt, p_jax, rtol=1e-9, atol=2e-7)


@pytest.mark.parametrize("model_cls", [OrderedLogLogit, OrderedLogProbit])
@pytest.mark.parametrize("enforce", [False, True])
def test_backend_consistency_over_enforcement_log(
    tiny_df, expressions, model_cls, enforce
):
    model = model_cls(
        eta=expressions["eta"],
        cutpoints=expressions["cutpoints"],
        y=expressions["y"],
        categories=expressions["categories"],
        neutral_labels=expressions["neutral_labels"],
        enforce_order=enforce,
        eps=1e-12,
    )
    with pm.Model():
        pt_builder = model.recursive_construct_pymc_model_builder()
        ll_pt = np.asarray(pt.function([], pt_builder(tiny_df))())
    jax_builder = model.recursive_construct_jax_function(numerically_safe=True)
    ll_jax = np.array(
        [_eval_jax_row(jax_builder, tiny_df.iloc[i]) for i in range(len(tiny_df))]
    )
    np.testing.assert_allclose(ll_pt, ll_jax, rtol=1e-9, atol=2e-7)
