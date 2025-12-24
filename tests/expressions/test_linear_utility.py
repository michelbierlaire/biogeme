# tests/expressions/test_linear_utility.py

import logging

import pandas as pd
import pytensor.tensor as pt
import pytest

from biogeme.exceptions import BiogemeError
from biogeme.expressions.beta_parameters import Beta
from biogeme.expressions.linear_utility import LinearTermTuple, LinearUtility
from biogeme.expressions.variable import Variable

logger = logging.getLogger(__name__)


def _make_beta(name: str, value: float = 0.0) -> Beta:
    """Helper to create a simple Beta parameter."""
    # Standard Biogeme signature: Beta(name, value, lowerbound, upperbound, status)
    return Beta(name, value, None, None, 0)


def _make_var(name: str) -> Variable:
    """Helper to create a simple Variable."""
    return Variable(name)


# ---------------------------------------------------------------------------
# __init__ and basic structure
# ---------------------------------------------------------------------------


def test_linear_utility_init_valid_terms() -> None:
    """Valid (Beta, Variable) pairs are stored correctly."""
    b1 = _make_beta("b_time")
    b2 = _make_beta("b_cost")
    v1 = _make_var("time")
    v2 = _make_var("cost")

    terms = [LinearTermTuple(beta=b1, x=v1), LinearTermTuple(beta=b2, x=v2)]
    lu = LinearUtility(list_of_terms=terms)

    # list_of_terms preserved
    assert lu.list_of_terms == terms

    # betas and variables extracted correctly, in order
    assert lu.betas == [b1, b2]
    assert lu.variables == [v1, v2]

    # Children should contain all betas and variables.
    # Compare by name to avoid triggering Biogeme's overloaded == operator,
    # which does not support boolean evaluation.
    children_names = {child.name for child in lu.children}
    expected_names = [b.name for b in lu.betas] + [v.name for v in lu.variables]
    for name in expected_names:
        assert name in children_names


def test_linear_utility_init_invalid_term_raises() -> None:
    """Non-(Beta, Variable) terms must raise a BiogemeError."""
    b1 = _make_beta("b_time")
    v1 = _make_var("time")

    # First element not a Beta
    with pytest.raises(BiogemeError):
        LinearUtility(list_of_terms=[LinearTermTuple(beta=v1, x=v1)])  # type: ignore[arg-type]

    # Second element not a Variable
    with pytest.raises(BiogemeError):
        LinearUtility(list_of_terms=[LinearTermTuple(beta=b1, x=b1)])  # type: ignore[arg-type]


def test_linear_utility_empty_terms() -> None:
    """An empty list of terms is technically allowed and produces empty structures."""
    with pytest.raises(BiogemeError):
        _ = LinearUtility(list_of_terms=[])


# ---------------------------------------------------------------------------
# deep_flat_copy
# ---------------------------------------------------------------------------


def test_deep_flat_copy_creates_independent_objects() -> None:
    """
    deep_flat_copy must:
    - return a LinearUtility instance,
    - preserve the structure (number of terms, names),
    - create new Beta/Variable instances (deep copy).
    """
    b1 = _make_beta("b_time", value=1.0)
    b2 = _make_beta("b_cost", value=-2.0)
    v1 = _make_var("time")
    v2 = _make_var("cost")

    lu = LinearUtility(
        list_of_terms=[LinearTermTuple(beta=b1, x=v1), LinearTermTuple(beta=b2, x=v2)]
    )

    lu_copy = lu.deep_flat_copy()

    # Type and basic content
    assert isinstance(lu_copy, LinearUtility)
    assert len(lu_copy.list_of_terms) == len(lu.list_of_terms)

    # Same names, different object identities (deep copy)
    orig_betas = lu.betas
    orig_vars = lu.variables
    copy_betas = lu_copy.betas
    copy_vars = lu_copy.variables

    assert [b.name for b in orig_betas] == [b.name for b in copy_betas]
    assert [v.name for v in orig_vars] == [v.name for v in copy_vars]

    for ob, cb in zip(orig_betas, copy_betas):
        assert ob is not cb

    for ov, cv in zip(orig_vars, copy_vars):
        assert ov is not cv


# ---------------------------------------------------------------------------
# __str__ and __repr__
# ---------------------------------------------------------------------------


def test_str_representation() -> None:
    """__str__ should produce a readable linear-form expression string."""
    b1 = _make_beta("b_time")
    b2 = _make_beta("b_cost")
    v1 = _make_var("time")
    v2 = _make_var("cost")

    lu = LinearUtility(
        list_of_terms=[LinearTermTuple(beta=b1, x=v1), LinearTermTuple(beta=b2, x=v2)]
    )

    s = str(lu)
    # Order and formatting: "b_time * time + b_cost * cost"
    assert "Beta('b_time', 0.0, None, None, 0) * time" in s
    assert "Beta('b_cost', 0.0, None, None, 0) * cost" in s
    assert "+" in s


def test_repr_roundtrip_like() -> None:
    """__repr__ should contain the class name and the list_of_terms representation."""
    b = _make_beta("b_time")
    v = _make_var("time")

    lu = LinearUtility(list_of_terms=[LinearTermTuple(beta=b, x=v)])

    r = repr(lu)
    assert "LinearUtility" in r
    assert "b_time" in r
    assert "time" in r


# ---------------------------------------------------------------------------
# JAX builder
# ---------------------------------------------------------------------------


def test_recursive_construct_jax_function_returns_callable() -> None:
    """
    The JAX builder must return a callable that accepts
    (parameters, one_row, draws, random_variables).

    We do not assert numerical values here, only that the function object
    is created and can be called without raising for simple inputs.
    """

    b1 = _make_beta("b_time")
    v1 = _make_var("time")

    lu = LinearUtility(list_of_terms=[LinearTermTuple(beta=b1, x=v1)])

    jax_fn = lu.recursive_construct_jax_function(numerically_safe=False)
    assert callable(jax_fn)


# ---------------------------------------------------------------------------
# PyMC builder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_obs", [1, 5])
def test_recursive_construct_pymc_model_builder_shape(n_obs: int) -> None:
    """
    The PyMC model builder should return a per-observation tensor
    whose length equals the number of rows in the input DataFrame.
    """
    import pymc as pm

    # Two variables, each with their own Beta
    b_time = _make_beta("b_time")
    b_cost = _make_beta("b_cost")
    v_time = _make_var("time")
    v_cost = _make_var("cost")

    lu = LinearUtility(
        list_of_terms=[
            LinearTermTuple(beta=b_time, x=v_time),
            LinearTermTuple(beta=b_cost, x=v_cost),
        ]
    )

    builder = lu.recursive_construct_pymc_model_builder()

    # Create a simple DataFrame with matching column names
    df = pd.DataFrame(
        {
            "time": [1.0 + i for i in range(n_obs)],
            "cost": [0.5 + 0.1 * i for i in range(n_obs)],
        }
    )

    with pm.Model():
        tensor = builder(dataframe=df)

    # We expect a PyTensor variable with one dimension (observations)
    assert isinstance(tensor, pt.TensorVariable)
    assert tensor.ndim == 1
    assert tensor.shape[0].eval() == n_obs  # type: ignore[call-arg]


def test_recursive_construct_pymc_single_term_returns_vector() -> None:
    """
    Even with a single term, the PyMC builder must return a vector
    over observations (not a scalar).
    """
    import pymc as pm

    b = _make_beta("b_time")
    v = _make_var("time")

    lu = LinearUtility(list_of_terms=[LinearTermTuple(beta=b, x=v)])
    builder = lu.recursive_construct_pymc_model_builder()

    df = pd.DataFrame({"time": [1.0, 2.0, 3.0]})

    with pm.Model():
        tensor = builder(dataframe=df)

    assert isinstance(tensor, pt.TensorVariable)
    assert tensor.ndim == 1
    assert tensor.shape[0].eval() == 3  # type: ignore[call-arg]


def test_pymc_builder_raises_on_mismatched_lengths(monkeypatch) -> None:
    """
    Defensive check: if, for some reason, the number of beta and variable
    nodes disagree at build time, the builder should raise BiogemeError.

    We trigger this by temporarily altering the internal lists after construction.
    """
    import pymc as pm

    b = _make_beta("b_time")
    v = _make_var("time")

    lu = LinearUtility(list_of_terms=[LinearTermTuple(beta=b, x=v)])

    # Force an inconsistency between betas and variables to hit the error branch
    lu.betas = [b, _make_beta("b_dummy")]  # type: ignore[assignment]

    builder = lu.recursive_construct_pymc_model_builder()
    df = pd.DataFrame({"time": [1.0, 2.0]})

    with pm.Model():
        with pytest.raises(BiogemeError, match="LinearUtility mismatch"):
            _ = builder(dataframe=df)
