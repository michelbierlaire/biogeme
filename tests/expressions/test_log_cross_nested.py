"""Tests for LogCrossNested."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from biogeme.exceptions import BiogemeError
from biogeme.expressions.log_cross_nested import LogCrossNested
from biogeme.nests import NestsForCrossNestedLogit, OneNestForCrossNestedLogit


def _evaluate_jax(expression: LogCrossNested) -> float:
    jax_function = expression.recursive_construct_jax_function(numerically_safe=True)
    value = jax_function(
        jnp.array([]),
        jnp.array([]),
        jnp.array([]),
        jnp.array([]),
    )
    return float(value)


# Helper to convert Numeric or float to float
def _as_float(value) -> float:
    """Convert a numeric literal or a Biogeme Numeric expression to float."""
    if hasattr(value, 'get_value'):
        return float(value.get_value())
    return float(value)


def _build_cross_nested_structure(mu_existing=1.4, mu_public=1.7):
    existing = OneNestForCrossNestedLogit(
        nest_param=mu_existing,
        dict_of_alpha={
            1: 0.7,
            2: 0.0,
            3: 1.0,
        },
        name='existing',
    )
    public = OneNestForCrossNestedLogit(
        nest_param=mu_public,
        dict_of_alpha={
            1: 0.3,
            2: 1.0,
            3: 0.0,
        },
        name='public',
    )
    return NestsForCrossNestedLogit(
        choice_set=[1, 2, 3],
        tuple_of_nests=(existing, public),
    )


def _manual_log_cross_nested(
    utilities: dict[int, float],
    availability: dict[int, float] | None,
    choice: int,
    nests: NestsForCrossNestedLogit,
    mu: float | None = None,
) -> float:
    if availability is None:
        availability = {i: 1.0 for i in utilities}

    if availability[choice] == 0.0:
        return -math.inf

    kernels = {i: -math.inf for i in utilities}

    for nest in nests:
        mu_m = _as_float(nest.nest_param)
        alpha_exponent = mu_m if mu is None else mu_m / mu

        biosum = 0.0
        for i, alpha in nest.dict_of_alpha.items():
            biosum += (
                availability[i]
                * _as_float(alpha) ** alpha_exponent
                * math.exp(mu_m * utilities[i])
            )

        if biosum <= 0.0:
            continue

        log_biosum = math.log(biosum)

        for i, alpha in nest.dict_of_alpha.items():
            alpha = _as_float(alpha)
            if alpha == 0.0:
                continue

            if mu is None:
                term = (
                    mu_m * math.log(alpha)
                    + mu_m * utilities[i]
                    + ((1.0 - mu_m) / mu_m) * log_biosum
                )
            else:
                term = (
                    alpha_exponent * math.log(alpha)
                    + mu_m * utilities[i]
                    + ((mu / mu_m) - 1.0) * log_biosum
                )

            kernels[i] = np.logaddexp(kernels[i], term)

    if mu is not None:
        kernels = {i: math.log(mu) + kernel for i, kernel in kernels.items()}

    denominator = sum(availability[i] * math.exp(kernels[i]) for i in utilities)

    return kernels[choice] - math.log(denominator)


def test_log_cross_nested_is_complex():
    expression = LogCrossNested(
        util={1: 0.0, 2: 1.0, 3: 2.0},
        av=None,
        nests=_build_cross_nested_structure(),
        choice=1,
    )

    assert expression.is_complex()


def test_get_value_matches_manual_standard_cnl():
    utilities = {1: -0.2, 2: 0.4, 3: 1.1}
    availability = {1: 1.0, 2: 1.0, 3: 1.0}
    nests = _build_cross_nested_structure()

    expression = LogCrossNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=1,
    )

    expected = _manual_log_cross_nested(
        utilities=utilities,
        availability=availability,
        choice=1,
        nests=nests,
    )

    assert expression.get_value() == pytest.approx(expected)


def test_jax_matches_manual_standard_cnl():
    utilities = {1: -0.2, 2: 0.4, 3: 1.1}
    availability = {1: 1.0, 2: 1.0, 3: 1.0}
    nests = _build_cross_nested_structure()

    expression = LogCrossNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=3,
    )

    expected = _manual_log_cross_nested(
        utilities=utilities,
        availability=availability,
        choice=3,
        nests=nests,
    )

    assert _evaluate_jax(expression) == pytest.approx(expected)


@pytest.mark.parametrize('choice', [1, 2, 3])
def test_jax_and_get_value_match_for_all_choices(choice):
    utilities = {1: -0.6, 2: 0.2, 3: 1.4}
    availability = {1: 1.0, 2: 1.0, 3: 1.0}
    nests = _build_cross_nested_structure(mu_existing=1.3, mu_public=1.9)

    expression = LogCrossNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=choice,
    )

    assert _evaluate_jax(expression) == pytest.approx(expression.get_value())


def test_get_value_matches_manual_explicit_mu_cnl():
    utilities = {1: -0.5, 2: 0.3, 3: 1.2}
    availability = {1: 1.0, 2: 1.0, 3: 1.0}
    nests = _build_cross_nested_structure(mu_existing=1.4, mu_public=1.8)
    mu = 1.3

    expression = LogCrossNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=1,
        mu=mu,
    )

    expected = _manual_log_cross_nested(
        utilities=utilities,
        availability=availability,
        choice=1,
        nests=nests,
        mu=mu,
    )

    assert expression.get_value() == pytest.approx(expected)


def test_jax_matches_manual_explicit_mu_cnl():
    utilities = {1: -0.5, 2: 0.3, 3: 1.2}
    availability = {1: 1.0, 2: 1.0, 3: 1.0}
    nests = _build_cross_nested_structure(mu_existing=1.4, mu_public=1.8)
    mu = 1.3

    expression = LogCrossNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=2,
        mu=mu,
    )

    expected = _manual_log_cross_nested(
        utilities=utilities,
        availability=availability,
        choice=2,
        nests=nests,
        mu=mu,
    )

    assert _evaluate_jax(expression) == pytest.approx(expected)


def test_without_availability_all_alternatives_are_available():
    utilities = {1: -0.6, 2: 0.2, 3: 1.4}
    nests = _build_cross_nested_structure()

    expression = LogCrossNested(
        util=utilities,
        av=None,
        nests=nests,
        choice=2,
    )

    expected = _manual_log_cross_nested(
        utilities=utilities,
        availability=None,
        choice=2,
        nests=nests,
    )

    assert expression.get_value() == pytest.approx(expected)
    assert _evaluate_jax(expression) == pytest.approx(expected)


def test_availability_dictionary_order_does_not_matter_get_value():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability_reference = {1: 1.0, 2: 1.0, 3: 0.0}
    availability_reordered = {3: 0.0, 1: 1.0, 2: 1.0}
    nests = _build_cross_nested_structure()

    reference = LogCrossNested(
        util=utilities,
        av=availability_reference,
        nests=nests,
        choice=1,
    )
    reordered = LogCrossNested(
        util=utilities,
        av=availability_reordered,
        nests=nests,
        choice=1,
    )

    assert reordered.get_value() == pytest.approx(reference.get_value())


def test_availability_dictionary_order_does_not_matter_jax():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability_reference = {1: 1.0, 2: 1.0, 3: 0.0}
    availability_reordered = {3: 0.0, 1: 1.0, 2: 1.0}
    nests = _build_cross_nested_structure()

    reference = LogCrossNested(
        util=utilities,
        av=availability_reference,
        nests=nests,
        choice=1,
    )
    reordered = LogCrossNested(
        util=utilities,
        av=availability_reordered,
        nests=nests,
        choice=1,
    )

    assert _evaluate_jax(reordered) == pytest.approx(_evaluate_jax(reference))


def test_zero_alpha_does_not_contribute_to_biosum_or_kernel():
    utilities = {1: 0.1, 2: 2.0, 3: 0.3}
    availability = {1: 1.0, 2: 1.0, 3: 1.0}
    nests = _build_cross_nested_structure()

    expression = LogCrossNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=3,
    )

    expected = _manual_log_cross_nested(
        utilities=utilities,
        availability=availability,
        choice=3,
        nests=nests,
    )

    assert expression.get_value() == pytest.approx(expected)
    assert _evaluate_jax(expression) == pytest.approx(expected)


def test_unavailable_alternative_is_excluded_from_biosums():
    utilities = {1: 0.1, 2: 0.5, 3: 3.0}
    availability = {1: 1.0, 2: 1.0, 3: 0.0}
    nests = _build_cross_nested_structure()

    expression = LogCrossNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=1,
    )

    expected = _manual_log_cross_nested(
        utilities=utilities,
        availability=availability,
        choice=1,
        nests=nests,
    )

    assert expression.get_value() == pytest.approx(expected)
    assert _evaluate_jax(expression) == pytest.approx(expected)


def test_get_value_detects_unavailable_chosen_alternative():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability = {1: 1.0, 2: 0.0, 3: 1.0}
    nests = _build_cross_nested_structure()

    expression = LogCrossNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=2,
    )

    assert np.isneginf(expression.get_value())


def test_jax_detects_unavailable_chosen_alternative():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability = {1: 1.0, 2: 0.0, 3: 1.0}
    nests = _build_cross_nested_structure()

    expression = LogCrossNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=2,
    )

    value = _evaluate_jax(expression)

    assert np.isneginf(value) or value < -1.0e20


def test_missing_availability_key_raises_biogeme_error():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability = {1: 1.0, 2: 1.0}
    nests = _build_cross_nested_structure()

    with pytest.raises(BiogemeError, match='availability dictionary'):
        LogCrossNested(
            util=utilities,
            av=availability,
            nests=nests,
            choice=2,
        )


def test_unknown_availability_key_raises_biogeme_error():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    nests = _build_cross_nested_structure()

    with pytest.raises(BiogemeError, match='availability dictionary'):
        LogCrossNested(
            util=utilities,
            av=availability,
            nests=nests,
            choice=2,
        )


def test_missing_and_unknown_availability_keys_raise_biogeme_error():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability = {1: 1.0, 2: 1.0, 4: 1.0}
    nests = _build_cross_nested_structure()

    with pytest.raises(BiogemeError, match='availability dictionary'):
        LogCrossNested(
            util=utilities,
            av=availability,
            nests=nests,
            choice=2,
        )


def test_unknown_choice_raises_biogeme_error_get_value():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    nests = _build_cross_nested_structure()

    expression = LogCrossNested(
        util=utilities,
        av=None,
        nests=nests,
        choice=4,
    )

    with pytest.raises(BiogemeError, match='does not appear in the utilities'):
        expression.get_value()


def test_availability_key_mismatch_is_detected_before_evaluation():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability = {1: 1.0, 2: 1.0, 4: 1.0}
    nests = _build_cross_nested_structure()

    with pytest.raises(BiogemeError, match='availability dictionary'):
        LogCrossNested(
            util=utilities,
            av=availability,
            nests=nests,
            choice=1,
        )


def test_deep_flat_copy_preserves_standard_value():
    utilities = {1: -0.3, 2: 0.2, 3: 1.0}
    availability = {1: 1.0, 2: 1.0, 3: 0.0}
    nests = _build_cross_nested_structure()

    expression = LogCrossNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=1,
    )
    copied = expression.deep_flat_copy()

    assert copied.get_value() == pytest.approx(expression.get_value())
    assert _evaluate_jax(copied) == pytest.approx(_evaluate_jax(expression))


def test_deep_flat_copy_preserves_explicit_mu_value():
    utilities = {1: -0.3, 2: 0.2, 3: 1.0}
    availability = {1: 1.0, 2: 1.0, 3: 1.0}
    nests = _build_cross_nested_structure()

    expression = LogCrossNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=1,
        mu=1.25,
    )
    copied = expression.deep_flat_copy()

    assert copied.get_value() == pytest.approx(expression.get_value())
    assert _evaluate_jax(copied) == pytest.approx(_evaluate_jax(expression))


def test_string_representation_contains_choice_and_utilities():
    expression = LogCrossNested(
        util={1: 0.0, 2: 1.0, 3: 2.0},
        av=None,
        nests=_build_cross_nested_structure(),
        choice=2,
    )

    representation = str(expression)

    assert 'LogCrossNested' in representation
    assert 'choice=' in representation
    assert '2' in representation
    assert '1:' in representation
    assert '2:' in representation
    assert '3:' in representation
