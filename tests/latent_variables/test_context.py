from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from biogeme.latent_variables.context import (
    BuildContext,
    EstimationMode,
    PositivityMode,
)
from biogeme.latent_variables.naming import DefaultNamingPolicy


def test_estimation_mode_enum_values() -> None:
    assert EstimationMode.MAXIMUM_LIKELIHOOD.value == 'maximum_likelihood'
    assert EstimationMode.BAYESIAN.value == 'bayesian'


def test_positivity_mode_enum_values() -> None:
    assert PositivityMode.LOG_EXP.value == 'log_exp'
    assert PositivityMode.LOWER_BOUND.value == 'lower_bound'


def test_build_context_direct_construction_with_defaults() -> None:
    context = BuildContext(
        estimation_mode=EstimationMode.MAXIMUM_LIKELIHOOD,
        draw_type='NORMAL_MLHS_ANTI',
        positivity_mode=PositivityMode.LOG_EXP,
        naming=DefaultNamingPolicy(),
    )

    assert context.estimation_mode is EstimationMode.MAXIMUM_LIKELIHOOD
    assert context.draw_type == 'NORMAL_MLHS_ANTI'
    assert context.positivity_mode is PositivityMode.LOG_EXP
    assert isinstance(context.naming, DefaultNamingPolicy)
    assert context.naming == DefaultNamingPolicy()
    assert context.ordinal_eps == 1e-12
    assert context.ordinal_enforce_order is True


def test_build_context_direct_construction_with_custom_optional_values() -> None:
    context = BuildContext(
        estimation_mode=EstimationMode.BAYESIAN,
        draw_type='Normal',
        positivity_mode=PositivityMode.LOWER_BOUND,
        naming=DefaultNamingPolicy(),
        ordinal_eps=1e-8,
        ordinal_enforce_order=False,
    )

    assert context.estimation_mode is EstimationMode.BAYESIAN
    assert context.draw_type == 'Normal'
    assert context.positivity_mode is PositivityMode.LOWER_BOUND
    assert isinstance(context.naming, DefaultNamingPolicy)
    assert context.ordinal_eps == 1e-8
    assert context.ordinal_enforce_order is False


def test_build_context_default_maximum_likelihood() -> None:
    context = BuildContext.default(EstimationMode.MAXIMUM_LIKELIHOOD)

    assert isinstance(context, BuildContext)
    assert context.estimation_mode is EstimationMode.MAXIMUM_LIKELIHOOD
    assert context.draw_type == 'NORMAL_MLHS_ANTI'
    assert context.positivity_mode is PositivityMode.LOG_EXP
    assert isinstance(context.naming, DefaultNamingPolicy)
    assert context.naming == DefaultNamingPolicy()
    assert context.ordinal_eps == 1e-12
    assert context.ordinal_enforce_order is True


def test_build_context_default_bayesian() -> None:
    context = BuildContext.default(EstimationMode.BAYESIAN)

    assert isinstance(context, BuildContext)
    assert context.estimation_mode is EstimationMode.BAYESIAN
    assert context.draw_type == 'Normal'
    assert context.positivity_mode is PositivityMode.LOWER_BOUND
    assert isinstance(context.naming, DefaultNamingPolicy)
    assert context.naming == DefaultNamingPolicy()
    assert context.ordinal_eps == 1e-12
    assert context.ordinal_enforce_order is True


def test_build_context_is_frozen() -> None:
    context = BuildContext.default(EstimationMode.MAXIMUM_LIKELIHOOD)

    with pytest.raises(FrozenInstanceError):
        context.draw_type = 'something_else'  # type: ignore[misc]


def test_build_context_uses_slots() -> None:
    context = BuildContext.default(EstimationMode.BAYESIAN)

    assert hasattr(BuildContext, '__slots__')
    assert '__dict__' not in dir(context)

    with pytest.raises((AttributeError, TypeError, FrozenInstanceError)):
        context.some_new_attribute = 123  # type: ignore[attr-defined]


def test_default_returns_distinct_naming_policy_instances() -> None:
    context_1 = BuildContext.default(EstimationMode.MAXIMUM_LIKELIHOOD)
    context_2 = BuildContext.default(EstimationMode.MAXIMUM_LIKELIHOOD)

    assert isinstance(context_1.naming, DefaultNamingPolicy)
    assert isinstance(context_2.naming, DefaultNamingPolicy)
    assert context_1.naming is not context_2.naming
