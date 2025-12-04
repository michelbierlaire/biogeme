import math

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except Exception:  # pragma: no cover - defensive
    JAX_AVAILABLE = False

from biogeme.expressions.boxcox import BoxCox
from biogeme.expressions.numeric_expressions import Numeric


# ----------------------------------------------------------------------
# Helper: reference Box–Cox implementation (scalar)
# BC(x, λ) = (x^λ - 1) / λ for λ != 0;  log(x) for λ == 0.
# For x == 0 we follow the convention BC(0, λ) = 0.
# ----------------------------------------------------------------------
def _boxcox_reference(x: float, lam: float) -> float:
    if x == 0.0:
        return 0.0
    if lam == 0.0:
        return math.log(x)
    return (x**lam - 1.0) / lam


# ----------------------------------------------------------------------
# NumPy / get_value tests
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "x, lam",
    [
        (0.0, 1.2),  # convention BC(0, λ) = 0
        (0.0, 0.0),  # also 0 by convention
        (2.0, 0.5),  # generic case
        (3.5, -0.3),  # negative lambda
        (1.7, 1e-8),  # |λ log x| very small → Taylor branch
        (4.2, 0.0),  # λ → 0 limit → log(x)
    ],
)
def test_boxcox_get_value_matches_reference(x, lam):
    """BoxCox.get_value should match the standard Box–Cox definition."""
    expr = BoxCox(Numeric(x), Numeric(lam))

    val = expr.get_value()
    ref = _boxcox_reference(x, lam)

    # Use reasonably tight tolerances; the implementation is numerically
    # stabilized, so it may differ slightly from the naive formula
    np.testing.assert_allclose(val, ref, rtol=1e-9, atol=1e-9)


# ----------------------------------------------------------------------
# PyTensor / PyMC builder tests
# ----------------------------------------------------------------------
def _eval_pytensor_vector(tensor: pt.Variable) -> np.ndarray:
    """Utility to evaluate a TensorVariable with no symbolic inputs."""
    f = pytensor.function([], tensor)
    return np.asarray(f())


def test_boxcox_pymc_builder_constant_numeric():
    """
    The PyMC builder with Numeric children should return a vector of
    Box–Cox values, one per row of the dataframe.
    """
    x_val = 2.5
    lam_val = 0.7
    expr = BoxCox(Numeric(x_val), Numeric(lam_val))
    builder = expr.recursive_construct_pymc_model_builder()

    # Dataframe length defines the output length; contents are irrelevant
    df = pd.DataFrame({"dummy": [0, 1, 2, 3]})

    tensor = builder(df)  # expect shape (N,)
    out = _eval_pytensor_vector(tensor)

    assert out.shape == (len(df),)
    ref = _boxcox_reference(x_val, lam_val)
    np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-12)


def test_boxcox_pymc_builder_handles_zero_without_nan():
    """
    For x == 0 the PyMC builder should return exactly 0 and not NaN/inf.
    """
    expr = BoxCox(Numeric(0.0), Numeric(1.23))
    builder = expr.recursive_construct_pymc_model_builder()
    df = pd.DataFrame({"dummy": [0, 1, 2]})

    tensor = builder(df)
    out = _eval_pytensor_vector(tensor)

    assert out.shape == (len(df),)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, 0.0, atol=0.0)


def test_boxcox_pymc_builder_taylor_region_close_to_limit():
    """
    When |λ log x| is tiny, the implementation uses a Taylor expansion.
    The result should be close to the theoretical λ→0 limit log(x).
    """
    x_val = 1.5
    lam_val = 1e-9  # tiny → forces Taylor branch
    expr = BoxCox(Numeric(x_val), Numeric(lam_val))
    builder = expr.recursive_construct_pymc_model_builder()
    df = pd.DataFrame({"dummy": [0, 1, 2, 3, 4]})

    tensor = builder(df)
    out = _eval_pytensor_vector(tensor)

    expected = math.log(x_val)
    np.testing.assert_allclose(out, expected, rtol=1e-8, atol=1e-10)


# ----------------------------------------------------------------------
# JAX builder tests
# ----------------------------------------------------------------------
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
@pytest.mark.parametrize("numerically_safe", [False, True])
def test_boxcox_jax_builder_matches_reference(numerically_safe):
    """
    JAX builder should agree with the reference Box–Cox definition for
    a variety of (x, λ) pairs.
    """
    jax.config.update("jax_enable_x64", True)

    # a few pairs including zero and small |λ log x|
    test_values = [
        (0.0, 1.0),
        (2.0, 0.5),
        (3.5, -0.2),
        (1.7, 1e-8),
        (4.2, 0.0),
    ]

    for x_val, lam_val in test_values:
        expr = BoxCox(Numeric(x_val), Numeric(lam_val))
        jax_fn = expr.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )

        # Arguments are unused by Numeric; we can use simple 1D arrays
        params = jnp.zeros(1)
        row = jnp.zeros(1)
        draws = jnp.zeros(1)
        rvars = jnp.zeros(1)

        out = jax_fn(params, row, draws, rvars)
        out_np = np.asarray(out)

        ref = _boxcox_reference(x_val, lam_val)
        np.testing.assert_allclose(out_np, ref, rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
def test_boxcox_jax_builder_zero_handling_exact():
    """
    For x == 0 the JAX builder should return exactly 0 (within float precision),
    for both numerically_safe branches.
    """
    jax.config.update("jax_enable_x64", True)
    expr = BoxCox(Numeric(0.0), Numeric(0.3))

    params = jnp.zeros(1)
    row = jnp.zeros(1)
    draws = jnp.zeros(1)
    rvars = jnp.zeros(1)

    for numerically_safe in (False, True):
        jax_fn = expr.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )
        out = np.asarray(jax_fn(params, row, draws, rvars))
        np.testing.assert_allclose(out, 0.0, atol=0.0)
