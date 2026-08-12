"""Contract tests for explicit replacement of automatically generated parameters.

These tests deliberately describe the public, preprocessing-only API.  The
implementation is expected to be added after the contract has been agreed:
``ParameterOverrides`` collects replacements and ``apply_parameter_overrides``
applies them once to one expression, a mapping of formulas, or a catalog.
"""

from __future__ import annotations

import pytest

from biogeme.catalog import Catalog, Controller
from biogeme.exceptions import BiogemeError, DuplicateError
from biogeme.expressions import (
    Beta,
    Numeric,
    ParameterOverrides,
    Variable,
    apply_parameter_overrides,
    exp,
    list_of_all_betas_in_expression,
)


def beta_names(expression) -> set[str]:
    """Return all beta names occurring in an expression."""

    return {beta.name for beta in list_of_all_betas_in_expression(expression)}


def test_identity_replacement_preserves_the_formula_and_does_not_mutate_it():
    original_beta = Beta('beta', 1.5, -10.0, 10.0, 0)
    original = original_beta + Numeric(2)

    overrides = ParameterOverrides()
    overrides.set('beta', Beta('beta', 1.5, -10.0, 10.0, 0))

    transformed = apply_parameter_overrides(original, overrides)

    assert transformed is not original
    assert transformed.get_value() == original.get_value()
    assert beta_names(transformed) == {'beta'}
    assert original_beta.init_value == 1.5
    assert original_beta.lower_bound == -10.0
    assert original_beta.upper_bound == 10.0


def test_numeric_replacement_removes_all_occurrences():
    beta = Beta('beta', 1.0, None, None, 0)
    original = beta + beta

    overrides = ParameterOverrides()
    overrides.set('beta', Numeric(0))
    transformed = apply_parameter_overrides(original, overrides)

    assert 'beta' in beta_names(original)
    assert 'beta' not in beta_names(transformed)
    assert transformed.get_value() == 0


def test_replacement_can_change_beta_initial_value_bounds_and_status():
    original = Beta('beta', 1.0, None, None, 0) * Variable('income')
    replacement = Beta('beta', -1.0, -10.0, 0.0, 1)

    overrides = ParameterOverrides()
    overrides.set('beta', replacement)
    transformed = apply_parameter_overrides(original, overrides)

    transformed_beta = next(iter(list_of_all_betas_in_expression(transformed)))
    assert transformed_beta.name == 'beta'
    assert transformed_beta.init_value == -1.0
    assert transformed_beta.lower_bound == -10.0
    assert transformed_beta.upper_bound == 0.0
    assert transformed_beta.status == 1
    # The replacement itself is not modified as a side effect of applying it.
    assert replacement.init_value == -1.0
    assert replacement.status == 1


def test_complex_expression_replacement_is_supported():
    original = Beta('beta', 1.0, None, None, 0) * Variable('income') + exp(
        Beta('other', 0.0, None, None, 0)
    )
    replacement = Beta('new_beta', -2.0, -10.0, 0.0, 0) * Variable('time') + exp(
        Beta('new_other', 0.0, None, None, 0)
    )

    overrides = ParameterOverrides()
    overrides.set('beta', replacement)
    transformed = apply_parameter_overrides(original, overrides)

    assert beta_names(transformed) == {'new_beta', 'new_other', 'other'}
    assert 'beta' not in beta_names(transformed)


def test_repeated_beta_objects_with_the_same_name_are_all_replaced():
    original = Beta('x', 1.0, None, None, 0) + Beta('x', 2.0, None, None, 0)

    overrides = ParameterOverrides()
    overrides.set('x', Numeric(3))
    transformed = apply_parameter_overrides(original, overrides)

    assert beta_names(transformed) == set()
    assert transformed.get_value() == 6


def test_all_formulas_in_a_mapping_are_replaced():
    formulas = {
        'loglike': Beta('beta', 1.0, None, None, 0) + Numeric(1),
        'weight': Beta('beta', 1.0, None, None, 0) * Numeric(2),
    }
    overrides = ParameterOverrides()
    overrides.set('beta', Numeric(0))

    transformed = apply_parameter_overrides(formulas, overrides)

    assert set(transformed) == {'loglike', 'weight'}
    assert all(beta_names(formula) == set() for formula in transformed.values())
    assert beta_names(formulas['loglike']) == {'beta'}


def test_catalog_alternatives_are_replaced_in_every_branch():
    catalog = Catalog.from_dict(
        catalog_name='alternatives',
        dict_of_expressions={
            'first': Beta('x', 1.0, None, None, 0) + Numeric(1),
            'second': Numeric(2) * Beta('x', 1.0, None, None, 0),
        },
    )
    overrides = ParameterOverrides()
    overrides.set('x', Numeric(0))

    transformed = apply_parameter_overrides(catalog, overrides)

    assert transformed is not catalog
    assert transformed.selected_name() == catalog.selected_name()
    assert len(transformed.named_expressions) == len(catalog.named_expressions)
    assert all(
        beta_names(named.expression) == set() for named in transformed.named_expressions
    )


def test_shared_catalog_controller_is_preserved_across_formula_mapping():
    controller = Controller('shared', ('first', 'second'))
    first = Catalog.from_dict(
        catalog_name='first_catalog',
        dict_of_expressions={
            'first': Beta('x', 1.0, None, None, 0),
            'second': Numeric(2),
        },
        controlled_by=controller,
    )
    second = Catalog.from_dict(
        catalog_name='second_catalog',
        dict_of_expressions={
            'first': Numeric(3),
            'second': Beta('x', 4.0, None, None, 0),
        },
        controlled_by=controller,
    )
    overrides = ParameterOverrides()
    overrides.set('x', Numeric(0))

    transformed = apply_parameter_overrides(
        {'first': first, 'second': second}, overrides
    )

    assert transformed['first'].controlled_by is transformed['second'].controlled_by
    assert transformed['first'].selected_name() == first.selected_name()
    assert transformed['second'].selected_name() == second.selected_name()


def test_self_named_replacement_is_not_recursively_rewritten():
    original = Beta('gamma', 1.0, None, None, 0) + Numeric(1)
    replacement = Beta('gamma', 2.0, None, None, 0) * Variable('income')

    overrides = ParameterOverrides()
    overrides.set('gamma', replacement)
    transformed = apply_parameter_overrides(original, overrides)

    # A replacement is an opaque expression for this pass.  In particular,
    # this must terminate and contain one inserted ``gamma`` beta.
    assert beta_names(transformed) == {'gamma'}
    assert len(list_of_all_betas_in_expression(transformed)) == 1


def test_overrides_are_simultaneous_and_not_transitive():
    original = Beta('a', 0.0, None, None, 0) + Beta('b', 0.0, None, None, 0)
    overrides = ParameterOverrides()
    overrides.set('a', Beta('b', 2.0, None, None, 0))
    overrides.set('b', Numeric(0))

    transformed = apply_parameter_overrides(original, overrides)

    # The original ``a`` becomes the inserted ``b``; the original ``b`` is
    # replaced by zero.  The inserted expression is not processed again.
    assert beta_names(transformed) == {'b'}
    assert transformed.get_value() == 2


def test_empty_overrides_leave_the_expression_unchanged():
    original = Beta('beta', 1.0, None, None, 0) + Numeric(2)

    transformed = apply_parameter_overrides(original, ParameterOverrides())

    assert transformed.get_value() == original.get_value()
    assert beta_names(transformed) == beta_names(original)


def test_unknown_target_is_reported():
    overrides = ParameterOverrides()
    overrides.set('missing', Numeric(0))

    with pytest.raises(BiogemeError, match='missing'):
        apply_parameter_overrides(Beta('present', 1.0, None, None, 0), overrides)


def test_target_name_must_be_a_nonempty_string():
    overrides = ParameterOverrides()

    with pytest.raises(BiogemeError):
        overrides.set('', Numeric(0))
    with pytest.raises(BiogemeError):
        overrides.set(123, Numeric(0))


def test_replacement_must_be_a_valid_biogeme_expression():
    overrides = ParameterOverrides()

    with pytest.raises(TypeError):
        overrides.set('beta', object())


def test_duplicate_target_names_are_rejected():
    overrides = ParameterOverrides()
    overrides.set('beta', Numeric(0))

    with pytest.raises(DuplicateError):
        overrides.set('beta', Numeric(1))


def test_input_must_be_an_expression_mapping_or_catalog():
    overrides = ParameterOverrides()
    overrides.set('beta', Numeric(0))

    with pytest.raises(BiogemeError):
        apply_parameter_overrides(None, overrides)
