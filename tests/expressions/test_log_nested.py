"""Tests for LogNested."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from biogeme.exceptions import BiogemeError
from biogeme.expressions.log_nested import LogNested
from biogeme.nests import NestsForNestedLogit, OneNestForNestedLogit


def _evaluate_jax_lognested(expression: LogNested) -> float:
    jax_function = expression.recursive_construct_jax_function(numerically_safe=True)
    value = jax_function(
        jnp.array([]),
        jnp.array([]),
        jnp.array([]),
        jnp.array([]),
    )
    return float(value)


def _build_nested_structure(nest_parameter=1.5):
    existing = OneNestForNestedLogit(
        nest_param=nest_parameter,
        list_of_alternatives=[1, 3],
        name='existing',
    )
    return NestsForNestedLogit(
        choice_set=[1, 2, 3],
        tuple_of_nests=(existing,),
    )


def _manual_nested_log_probability(
    utilities: dict[int, float],
    availability: dict[int, float] | None,
    choice: int,
    nest_parameter: float,
    mu: float | None = None,
) -> float:
    if availability is None:
        availability = {i: 1.0 for i in utilities}

    if availability[choice] == 0.0:
        return -math.inf

    nested_alternatives = [1, 3]
    alone_alternatives = [2]

    biosum = sum(
        availability[i] * math.exp(nest_parameter * utilities[i])
        for i in nested_alternatives
    )

    kernels = {}

    for i in nested_alternatives:
        if mu is None:
            kernels[i] = nest_parameter * utilities[i] + (
                1.0 / nest_parameter - 1.0
            ) * math.log(biosum)
        else:
            kernels[i] = (
                math.log(mu)
                + nest_parameter * utilities[i]
                + (mu / nest_parameter - 1.0) * math.log(biosum)
            )

    for i in alone_alternatives:
        if mu is None:
            kernels[i] = utilities[i]
        else:
            kernels[i] = math.log(mu) + mu * utilities[i]

    denominator = sum(availability[i] * math.exp(kernels[i]) for i in utilities)

    return kernels[choice] - math.log(denominator)


def _stable_logsumexp(values: list[float]) -> float:
    """Evaluate log(sum(exp(values))) without overflowing."""
    if not values:
        return -math.inf
    maximum = max(values)
    if not math.isfinite(maximum):
        return maximum
    return maximum + math.log(
        sum(math.exp(value - maximum) for value in values)
    )


def _stable_manual_nested_log_probability(
    utilities: dict[int, float],
    availability: dict[int, float],
    choice: int,
    nest_parameter: float,
    mu: float | None = None,
) -> float:
    """Independent log-domain reference for the test nested structure."""
    nested_alternatives = [1, 3]
    alone_alternatives = [2]

    if availability[choice] == 0.0:
        return -math.inf

    log_biosum = _stable_logsumexp(
        [
            math.log(availability[alternative])
            + nest_parameter * utilities[alternative]
            for alternative in nested_alternatives
            if availability[alternative] > 0.0
        ]
    )

    kernels: dict[int, float] = {}
    if math.isfinite(log_biosum):
        for alternative in nested_alternatives:
            if availability[alternative] > 0.0:
                if mu is None:
                    kernels[alternative] = nest_parameter * utilities[alternative] + (
                        1.0 / nest_parameter - 1.0
                    ) * log_biosum
                else:
                    kernels[alternative] = (
                        math.log(mu)
                        + nest_parameter * utilities[alternative]
                        + (mu / nest_parameter - 1.0) * log_biosum
                    )
    for alternative in alone_alternatives:
        kernels[alternative] = (
            utilities[alternative]
            if mu is None
            else math.log(mu) + mu * utilities[alternative]
        )

    denominator_terms = [
        math.log(availability[alternative]) + kernels[alternative]
        for alternative in utilities
        if availability[alternative] > 0.0 and alternative in kernels
    ]
    return kernels[choice] - _stable_logsumexp(denominator_terms)


def test_lognested_is_complex():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    nests = _build_nested_structure()

    expression = LogNested(
        util=utilities,
        av=None,
        nests=nests,
        choice=1,
    )

    assert expression.is_complex()


def test_get_value_matches_manual_standard_nested_logit():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability = {1: 1.0, 2: 1.0, 3: 1.0}
    nest_parameter = 1.5
    nests = _build_nested_structure(nest_parameter=nest_parameter)

    expression = LogNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=1,
    )

    expected = _manual_nested_log_probability(
        utilities=utilities,
        availability=availability,
        choice=1,
        nest_parameter=nest_parameter,
    )

    assert expression.get_value() == pytest.approx(expected)


def test_jax_matches_manual_standard_nested_logit():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability = {1: 1.0, 2: 1.0, 3: 1.0}
    nest_parameter = 1.5
    nests = _build_nested_structure(nest_parameter=nest_parameter)

    expression = LogNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=1,
    )

    expected = _manual_nested_log_probability(
        utilities=utilities,
        availability=availability,
        choice=1,
        nest_parameter=nest_parameter,
    )

    assert _evaluate_jax_lognested(expression) == pytest.approx(expected)


@pytest.mark.parametrize('mu', [None, 1.2])
def test_numerically_safe_jax_matches_log_domain_reference_for_extreme_utilities(
    mu,
):
    """The safe JAX path must agree with an independent log-domain reference."""
    utilities = {1: 1000.0, 2: -1000.0, 3: -1000.0}
    availability = {1: 1.0, 2: 1.0, 3: 1.0}
    nest_parameter = 1.5
    nests = _build_nested_structure(nest_parameter=nest_parameter)

    expression = LogNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=1,
        mu=mu,
    )

    expected = _stable_manual_nested_log_probability(
        utilities=utilities,
        availability=availability,
        choice=1,
        nest_parameter=nest_parameter,
        mu=mu,
    )
    result = _evaluate_jax_lognested(expression)

    assert np.isfinite(expected)
    assert np.isfinite(result)
    assert result == pytest.approx(expected)


def test_numerically_safe_jax_ignores_completely_unavailable_nested_nest():
    """An unavailable nest must not create 0 * (-inf) or contaminate the denominator."""
    utilities = {1: 0.0, 2: 0.5, 3: 2.0}
    availability = {1: 0.0, 2: 1.0, 3: 0.0}
    nest_parameter = 1.5
    nests = _build_nested_structure(nest_parameter=nest_parameter)

    expression = LogNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=2,
    )

    expected = _stable_manual_nested_log_probability(
        utilities=utilities,
        availability=availability,
        choice=2,
        nest_parameter=nest_parameter,
    )
    result = _evaluate_jax_lognested(expression)

    assert np.isfinite(expected)
    assert np.isfinite(result)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize('choice', [1, 2, 3])
def test_jax_and_get_value_match_for_all_choices(choice):
    utilities = {1: -0.4, 2: 0.7, 3: 1.3}
    availability = {1: 1.0, 2: 1.0, 3: 1.0}
    nests = _build_nested_structure(nest_parameter=1.7)

    expression = LogNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=choice,
    )

    assert _evaluate_jax_lognested(expression) == pytest.approx(expression.get_value())


def test_get_value_matches_manual_explicit_mu_nested_logit():
    utilities = {1: -0.2, 2: 0.5, 3: 1.4}
    availability = {1: 1.0, 2: 1.0, 3: 1.0}
    nest_parameter = 1.6
    mu = 1.2
    nests = _build_nested_structure(nest_parameter=nest_parameter)

    expression = LogNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=3,
        mu=mu,
    )

    expected = _manual_nested_log_probability(
        utilities=utilities,
        availability=availability,
        choice=3,
        nest_parameter=nest_parameter,
        mu=mu,
    )

    assert expression.get_value() == pytest.approx(expected)


def test_jax_matches_manual_explicit_mu_nested_logit():
    utilities = {1: -0.2, 2: 0.5, 3: 1.4}
    availability = {1: 1.0, 2: 1.0, 3: 1.0}
    nest_parameter = 1.6
    mu = 1.2
    nests = _build_nested_structure(nest_parameter=nest_parameter)

    expression = LogNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=3,
        mu=mu,
    )

    expected = _manual_nested_log_probability(
        utilities=utilities,
        availability=availability,
        choice=3,
        nest_parameter=nest_parameter,
        mu=mu,
    )

    assert _evaluate_jax_lognested(expression) == pytest.approx(expected)


def test_without_availability_all_alternatives_are_available():
    utilities = {1: -0.4, 2: 0.7, 3: 1.3}
    nest_parameter = 1.7
    nests = _build_nested_structure(nest_parameter=nest_parameter)

    expression = LogNested(
        util=utilities,
        av=None,
        nests=nests,
        choice=2,
    )

    expected = _manual_nested_log_probability(
        utilities=utilities,
        availability=None,
        choice=2,
        nest_parameter=nest_parameter,
    )

    assert expression.get_value() == pytest.approx(expected)
    assert _evaluate_jax_lognested(expression) == pytest.approx(expected)


def test_availability_dictionary_order_does_not_matter_get_value():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability_reference = {1: 1.0, 2: 1.0, 3: 0.0}
    availability_reordered = {3: 0.0, 1: 1.0, 2: 1.0}
    nests = _build_nested_structure(nest_parameter=1.5)

    expression_reference = LogNested(
        util=utilities,
        av=availability_reference,
        nests=nests,
        choice=1,
    )
    expression_reordered = LogNested(
        util=utilities,
        av=availability_reordered,
        nests=nests,
        choice=1,
    )

    assert expression_reordered.get_value() == pytest.approx(
        expression_reference.get_value()
    )


def test_availability_dictionary_order_does_not_matter_jax():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability_reference = {1: 1.0, 2: 1.0, 3: 0.0}
    availability_reordered = {3: 0.0, 1: 1.0, 2: 1.0}
    nests = _build_nested_structure(nest_parameter=1.5)

    expression_reference = LogNested(
        util=utilities,
        av=availability_reference,
        nests=nests,
        choice=1,
    )
    expression_reordered = LogNested(
        util=utilities,
        av=availability_reordered,
        nests=nests,
        choice=1,
    )

    assert _evaluate_jax_lognested(expression_reordered) == pytest.approx(
        _evaluate_jax_lognested(expression_reference)
    )


def test_jax_uses_availability_keys_not_availability_positions():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability = {3: 0.0, 1: 1.0, 2: 1.0}
    nests = _build_nested_structure(nest_parameter=1.5)

    expression = LogNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=2,
    )

    expected = _manual_nested_log_probability(
        utilities=utilities,
        availability={1: 1.0, 2: 1.0, 3: 0.0},
        choice=2,
        nest_parameter=1.5,
    )

    assert _evaluate_jax_lognested(expression) == pytest.approx(expected)


def test_get_value_detects_unavailable_chosen_alternative():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability = {1: 1.0, 2: 0.0, 3: 1.0}
    nests = _build_nested_structure(nest_parameter=1.5)

    expression = LogNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=2,
    )

    assert np.isneginf(expression.get_value())


def test_jax_detects_unavailable_chosen_alternative():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability = {1: 1.0, 2: 0.0, 3: 1.0}
    nests = _build_nested_structure(nest_parameter=1.5)

    expression = LogNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=2,
    )

    value = _evaluate_jax_lognested(expression)

    assert np.isneginf(value) or value < -1.0e20


def test_unavailable_alternative_inside_nest_is_excluded():
    utilities = {1: 0.0, 2: 0.5, 3: 2.0}
    availability = {1: 1.0, 2: 1.0, 3: 0.0}
    nest_parameter = 1.5
    nests = _build_nested_structure(nest_parameter=nest_parameter)

    expression = LogNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=1,
    )

    expected = _manual_nested_log_probability(
        utilities=utilities,
        availability=availability,
        choice=1,
        nest_parameter=nest_parameter,
    )

    assert expression.get_value() == pytest.approx(expected)
    assert _evaluate_jax_lognested(expression) == pytest.approx(expected)


def test_alone_alternative_standard_kernel_is_ordinary_utility():
    utilities = {1: 0.0, 2: 0.5, 3: 1.0}
    availability = {1: 1.0, 2: 1.0, 3: 1.0}
    nest_parameter = 1.4
    nests = _build_nested_structure(nest_parameter=nest_parameter)

    expression = LogNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=2,
    )

    expected = _manual_nested_log_probability(
        utilities=utilities,
        availability=availability,
        choice=2,
        nest_parameter=nest_parameter,
    )

    assert expression.get_value() == pytest.approx(expected)
    assert _evaluate_jax_lognested(expression) == pytest.approx(expected)


def test_alone_alternative_explicit_mu_kernel_is_mu_scaled_utility():
    utilities = {1: 0.0, 2: 0.5, 3: 1.0}
    availability = {1: 1.0, 2: 1.0, 3: 1.0}
    nest_parameter = 1.4
    mu = 1.3
    nests = _build_nested_structure(nest_parameter=nest_parameter)

    expression = LogNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=2,
        mu=mu,
    )

    expected = _manual_nested_log_probability(
        utilities=utilities,
        availability=availability,
        choice=2,
        nest_parameter=nest_parameter,
        mu=mu,
    )

    assert expression.get_value() == pytest.approx(expected)
    assert _evaluate_jax_lognested(expression) == pytest.approx(expected)


def test_missing_availability_key_raises_biogeme_error():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability = {1: 1.0, 2: 1.0}
    nests = _build_nested_structure()

    with pytest.raises(BiogemeError, match='availability dictionary'):
        LogNested(
            util=utilities,
            av=availability,
            nests=nests,
            choice=2,
        )


def test_unknown_availability_key_raises_biogeme_error():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    nests = _build_nested_structure()

    with pytest.raises(BiogemeError, match='availability dictionary'):
        LogNested(
            util=utilities,
            av=availability,
            nests=nests,
            choice=2,
        )


def test_missing_and_unknown_availability_keys_raise_biogeme_error():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability = {1: 1.0, 2: 1.0, 4: 1.0}
    nests = _build_nested_structure()

    with pytest.raises(BiogemeError, match='availability dictionary'):
        LogNested(
            util=utilities,
            av=availability,
            nests=nests,
            choice=2,
        )


def test_unknown_choice_raises_biogeme_error_get_value():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    nests = _build_nested_structure()

    expression = LogNested(
        util=utilities,
        av=None,
        nests=nests,
        choice=4,
    )

    with pytest.raises(BiogemeError, match='does not appear in the utilities'):
        expression.get_value()


def test_invalid_partition_raises_biogeme_error():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}

    nest_a = OneNestForNestedLogit(
        nest_param=1.2,
        list_of_alternatives=[1, 3],
        name='a',
    )
    nest_b = OneNestForNestedLogit(
        nest_param=1.4,
        list_of_alternatives=[3],
        name='b',
    )
    invalid_nests = NestsForNestedLogit(
        choice_set=[1, 2, 3],
        tuple_of_nests=(nest_a, nest_b),
    )

    with pytest.raises(BiogemeError):
        LogNested(
            util=utilities,
            av=None,
            nests=invalid_nests,
            choice=1,
        )


def test_deep_flat_copy_preserves_value():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability = {1: 1.0, 2: 1.0, 3: 0.0}
    nests = _build_nested_structure(nest_parameter=1.5)

    expression = LogNested(
        util=utilities,
        av=availability,
        nests=nests,
        choice=1,
    )
    copied_expression = expression.deep_flat_copy()

    assert copied_expression.get_value() == pytest.approx(expression.get_value())
    assert _evaluate_jax_lognested(copied_expression) == pytest.approx(
        _evaluate_jax_lognested(expression)
    )


def test_string_representation_contains_choice_and_utilities():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    nests = _build_nested_structure()

    expression = LogNested(
        util=utilities,
        av=None,
        nests=nests,
        choice=2,
    )

    representation = str(expression)

    assert 'LogNested' in representation
    assert 'choice=' in representation
    assert '2' in representation
    assert '1:' in representation
    assert '2:' in representation
    assert '3:' in representation
