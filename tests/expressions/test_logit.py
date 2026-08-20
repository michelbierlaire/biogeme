import math

import jax.numpy as jnp
import pytest

from biogeme.exceptions import BiogemeError
from biogeme.expressions.logit_expressions import LogLogit
from biogeme.floating_point import NEGATIVE_LARGE


def _evaluate_jax_loglogit(expression: LogLogit) -> float:
    jax_function = expression.recursive_construct_jax_function(numerically_safe=True)
    value = jax_function(
        jnp.array([]),
        jnp.array([]),
        jnp.array([]),
        jnp.array([]),
    )
    return float(value)


def test_jax_loglogit_is_independent_of_availability_dictionary_order():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}

    availability_reference_order = {1: 1.0, 2: 1.0, 3: 0.0}
    availability_different_order = {3: 0.0, 1: 1.0, 2: 1.0}

    expression_reference = LogLogit(
        util=utilities,
        av=availability_reference_order,
        choice=2,
    )
    expression_reordered = LogLogit(
        util=utilities,
        av=availability_different_order,
        choice=2,
    )

    log_prob_reference = _evaluate_jax_loglogit(expression_reference)
    log_prob_reordered = _evaluate_jax_loglogit(expression_reordered)

    assert log_prob_reordered == pytest.approx(log_prob_reference)


def test_jax_loglogit_uses_availability_keys_not_availability_positions():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability = {3: 0.0, 1: 1.0, 2: 1.0}

    expression = LogLogit(util=utilities, av=availability, choice=2)

    log_probability = _evaluate_jax_loglogit(expression)

    expected = 1.0 - math.log(math.exp(0.0) + math.exp(1.0))

    assert log_probability == pytest.approx(expected)


def test_jax_loglogit_reordered_availability_detects_unavailable_chosen_alternative():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability = {3: 1.0, 1: 1.0, 2: 0.0}

    expression = LogLogit(util=utilities, av=availability, choice=2)

    log_probability = _evaluate_jax_loglogit(expression)

    assert log_probability == NEGATIVE_LARGE


def test_get_value_returns_finite_sentinel_for_unavailable_choice():
    expression = LogLogit(
        util={1: 0.0, 2: 1.0},
        av={1: 1.0, 2: 0.0},
        choice=2,
    )

    assert expression.get_value() == NEGATIVE_LARGE


def test_get_value_is_independent_of_availability_dictionary_order():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}

    availability_reference_order = {1: 1.0, 2: 1.0, 3: 0.0}
    availability_different_order = {3: 0.0, 1: 1.0, 2: 1.0}

    expression_reference = LogLogit(
        util=utilities,
        av=availability_reference_order,
        choice=2,
    )
    expression_reordered = LogLogit(
        util=utilities,
        av=availability_different_order,
        choice=2,
    )

    assert expression_reordered.get_value() == pytest.approx(
        expression_reference.get_value()
    )


def test_availability_values_are_aligned_with_utility_keys():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability = {3: 0.0, 1: 1.0, 2: 1.0}

    expression = LogLogit(util=utilities, av=availability, choice=2)

    assert list(expression.util_keys) == pytest.approx([1.0, 2.0, 3.0])
    assert [av.get_value() for av in expression.av_values] == [1.0, 1.0, 0.0]


def test_missing_availability_key_raises_biogeme_error():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability = {1: 1.0, 2: 1.0}

    with pytest.raises(BiogemeError, match='availability dictionary'):
        LogLogit(util=utilities, av=availability, choice=2)


def test_unknown_availability_key_raises_biogeme_error():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

    with pytest.raises(BiogemeError, match='availability dictionary'):
        LogLogit(util=utilities, av=availability, choice=2)


def test_missing_and_unknown_availability_keys_raise_biogeme_error():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}
    availability = {1: 1.0, 2: 1.0, 4: 1.0}

    with pytest.raises(BiogemeError, match='availability dictionary'):
        LogLogit(util=utilities, av=availability, choice=2)


def test_loglogit_without_availability_still_works():
    utilities = {1: 0.0, 2: 1.0, 3: 2.0}

    expression = LogLogit(util=utilities, av=None, choice=2)

    log_probability = _evaluate_jax_loglogit(expression)
    expected = 1.0 - math.log(math.exp(0.0) + math.exp(1.0) + math.exp(2.0))

    assert log_probability == pytest.approx(expected)
