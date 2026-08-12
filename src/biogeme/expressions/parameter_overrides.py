"""Explicit replacement of parameters in Biogeme expressions.

The objects in this module are deliberately independent of :class:`BIOGEME`.
They are intended to be used as a preprocessing step, before formulas are
passed to an estimation or simulation object.
"""

from __future__ import annotations

import copy
from collections.abc import ItemsView, Iterator
from typing import Any

from biogeme.exceptions import BiogemeError, DuplicateError

from .base_expressions import Expression, ExpressionOrNumeric
from .beta_parameters import Beta
from .collectors import list_of_all_betas_in_expression
from .convert import validate_and_convert

FormulaInput = Expression | dict[str, ExpressionOrNumeric]
"""Accepted input to :func:`apply_parameter_overrides`."""


class ParameterOverrides:
    """Collection of explicit replacements keyed by the original Beta name.

    Replacements are stored in insertion order.  They are applied
    simultaneously and exactly once: if a replacement contains a Beta whose
    name is also overridden, that inserted Beta is not rewritten again during
    the same call.
    """

    def __init__(self) -> None:
        self._replacements: dict[str, Expression] = {}

    def set(self, name: str, replacement: ExpressionOrNumeric) -> None:
        """Register a replacement for a Beta name.

        :param name: Name of the original Beta to replace.
        :param replacement: Any valid Biogeme expression, including a numeric
            value which is converted to :class:`Numeric`.
        :raises BiogemeError: if ``name`` is not a non-empty string.
        :raises DuplicateError: if the name was already registered.
        :raises TypeError: if ``replacement`` is not a valid expression.
        """

        if not isinstance(name, str) or not name:
            raise BiogemeError(
                'An override name must be a non-empty string, '
                f'not {name!r} of type {type(name).__name__}.'
            )
        if name in self._replacements:
            raise DuplicateError(f'An override for Beta [{name}] already exists.')

        self._replacements[name] = validate_and_convert(replacement)

    def items(self) -> ItemsView[str, Expression]:
        """Return the registered overrides in insertion order."""

        return self._replacements.items()

    def __contains__(self, name: object) -> bool:
        return name in self._replacements

    def __getitem__(self, name: str) -> Expression:
        return self._replacements[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._replacements)

    def __len__(self) -> int:
        return len(self._replacements)


def _normalise_formulas(formulas: FormulaInput) -> FormulaInput:
    """Validate formula inputs without changing their public container shape."""

    if isinstance(formulas, Expression):
        return formulas

    if isinstance(formulas, dict):
        normalised: dict[str, Expression] = {}
        for name, formula in formulas.items():
            if not isinstance(name, str):
                raise BiogemeError(
                    f'Formula names must be strings, not {name!r} of type '
                    f'{type(name).__name__}.'
                )
            normalised[name] = validate_and_convert(formula)
        return normalised

    raise BiogemeError(
        f'Invalid type for formulas: {type(formulas).__name__}. '
        'Expected an Expression or a dictionary of expressions.'
    )


def _beta_names(formulas: FormulaInput) -> set[str]:
    """Collect Beta names from all formula and catalog branches."""

    if isinstance(formulas, Expression):
        expressions = (formulas,)
    else:
        expressions = formulas.values()

    return {
        beta.name
        for expression in expressions
        for beta in list_of_all_betas_in_expression(expression)
    }


def _rewrite_container(
    value: Any,
    replacements: dict[str, Expression],
    visited: set[int],
    replaced: dict[int, Expression],
    container_memo: dict[int, Any],
) -> Any:
    """Rewrite expression references in a copied expression object graph.

    Expression classes keep both semantic attributes (for example ``left``
    and ``right``) and a ``children`` list.  The rewriter therefore walks all
    expression-bearing attributes, rather than changing only ``children``.
    Non-expression objects such as catalog controllers are deliberately not
    traversed; they were already copied by :func:`copy.deepcopy`, and walking
    them would follow their back references to catalogs indefinitely.
    """

    if isinstance(value, Beta):
        existing_replacement = replaced.get(id(value))
        if existing_replacement is not None:
            return existing_replacement

        replacement = replacements.get(value.name)
        if replacement is None:
            return value

        # The replacement is copied, but intentionally not passed back to this
        # function.  This makes every replacement opaque for this pass and
        # prevents recursive self-replacement.
        copied_replacement = copy.deepcopy(replacement)
        replaced[id(value)] = copied_replacement
        return copied_replacement

    if isinstance(value, Expression):
        object_id = id(value)
        if object_id in visited:
            return value
        visited.add(object_id)
        for attribute_name, attribute_value in vars(value).items():
            rewritten = _rewrite_container(
                attribute_value,
                replacements,
                visited,
                replaced,
                container_memo,
            )
            if rewritten is not attribute_value:
                setattr(value, attribute_name, rewritten)
        return value

    if isinstance(value, list):
        object_id = id(value)
        if object_id in container_memo:
            return container_memo[object_id]
        container_memo[object_id] = value
        for index, item in enumerate(value):
            value[index] = _rewrite_container(
                item, replacements, visited, replaced, container_memo
            )
        return value

    if isinstance(value, dict):
        object_id = id(value)
        if object_id in container_memo:
            return container_memo[object_id]
        container_memo[object_id] = value

        rewritten_items = [
            (
                _rewrite_container(
                    key, replacements, visited, replaced, container_memo
                ),
                _rewrite_container(
                    item, replacements, visited, replaced, container_memo
                ),
            )
            for key, item in value.items()
        ]
        value.clear()
        value.update(rewritten_items)
        return value

    if isinstance(value, tuple):
        object_id = id(value)
        if object_id in container_memo:
            return container_memo[object_id]

        rewritten_items = tuple(
            _rewrite_container(item, replacements, visited, replaced, container_memo)
            for item in value
        )
        if hasattr(value, '_fields'):
            rewritten_tuple = type(value)(*rewritten_items)
        else:
            rewritten_tuple = rewritten_items
        container_memo[object_id] = rewritten_tuple
        return rewritten_tuple

    return value


def apply_parameter_overrides(
    formulas: FormulaInput, overrides: ParameterOverrides
) -> FormulaInput:
    """Apply explicit Beta replacements to formulas without mutating them.

    The complete expression graph is copied before replacement.  Every Beta
    with a matching ``name`` is replaced, including Betas in non-selected
    catalog alternatives.  The function performs one simultaneous pass: an
    expression inserted as a replacement is never inspected for further
    replacements.

    :param formulas: One Biogeme expression or a dictionary of formulas.
    :param overrides: Explicit replacements keyed by Beta name.
    :return: A new expression or formula dictionary with the replacements
        applied.  With no overrides, the original object is returned.
    :raises BiogemeError: for invalid formulas, an invalid override object, or
        an override name absent from all formula branches.
    """

    if not isinstance(overrides, ParameterOverrides):
        raise BiogemeError(
            'The overrides argument must be a ParameterOverrides object, '
            f'not {type(overrides).__name__}.'
        )

    normalised_formulas = _normalise_formulas(formulas)
    if not overrides:
        return formulas

    formula_beta_names = _beta_names(normalised_formulas)
    unknown_names = [name for name in overrides if name not in formula_beta_names]
    if unknown_names:
        raise BiogemeError(
            'The following override names do not occur in the formulas: '
            f'{", ".join(unknown_names)}.'
        )

    copied_formulas = copy.deepcopy(normalised_formulas)
    replacements = dict(overrides.items())
    rewritten = _rewrite_container(
        copied_formulas,
        replacements=replacements,
        visited=set(),
        replaced={},
        container_memo={},
    )
    return rewritten
