"""Function collecting recursively information about expressions

Michel Bierlaire
Thu May 01 2025, 18:50:24
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .beta_parameters import Beta
from .draws import Draws
from .random_variable import RandomVariable
from .variable import Variable

if TYPE_CHECKING:
    from .base_expressions import Expression


class ExpressionCollector:
    """Walks the tree and collects handler return values."""

    def __init__(self):
        self._registry = {}

    def register(self, expr_type: type[Expression]):
        """
        Register a handler function for a specific expression type. The handler must
        return a list of results.

        :param expr_type: the type of Expression for which the handler should be used
        :return: decorator that registers the handler
        """

        def decorator(func):
            self._registry[expr_type] = func
            return func

        return decorator

    def walk(self, expr: Expression, context: Any = None) -> list[Any]:
        """
        Traverse the expression tree and apply handlers to matching types.

        :param expr: the root expression to walk
        :param context: optional context object passed to handlers
        :return: list of collected results from the handlers
        """
        return self._visit(expr, context)

    def _visit(self, expr: Expression, context: dict[str, Any]) -> list[Any]:
        """
        Recursively visit expressions and collect handler results.

        Each handler function must return a list. If multiple handlers are triggered
        during the traversal, their outputs are flattened into a single list.

        :param expr: current expression node
        :param context: context passed down to handler functions
        :return: list of results from handler invocations
        """
        if context is None:
            context = {}
        if 'ancestors' not in context:
            context['ancestors'] = []
        context['ancestors'].append(expr)

        results = []
        handler = self._registry.get(type(expr))
        if handler:
            result = handler(expr, context)
            if not isinstance(result, list):
                raise TypeError(
                    f"Handler for {type(expr).__name__} must return a list, got {type(result).__name__}"
                )
            results.extend(result)
        for child in expr.get_children():
            child_result = self._visit(child, context)
            results.extend(child_result)
        context['ancestors'].pop()
        return results


def collect_init_values(expression: Expression) -> dict[str, float]:
    collector = ExpressionCollector()

    @collector.register(Beta)
    def collect_beta(expr: Beta, context) -> list[tuple[str, float]]:
        if expr.is_free:
            return [(expr.name, expr.init_value)]
        return []

    collected = collector.walk(expression)
    return dict(collected)


def list_of_variables_in_expression(
    the_expression: Expression,
) -> list[Variable]:
    # Create walker
    walker = ExpressionCollector()

    @walker.register(Variable)
    def retrieve_controllers(
        expr: Variable, context: Any | None = None
    ) -> list[Variable]:
        return [expr]

    # Now use it
    return walker.walk(the_expression)


def list_of_all_betas_in_expression(
    the_expression: Expression,
) -> list[Beta]:
    # Create walker
    walker = ExpressionCollector()

    @walker.register(Beta)
    def retrieve_controllers(expr: Beta, context: Any | None = None) -> list[Beta]:
        return [expr]

    # Now use it
    return walker.walk(the_expression)


def list_of_free_betas_in_expression(
    the_expression: Expression,
) -> list[Beta]:
    # Create walker
    walker = ExpressionCollector()

    @walker.register(Beta)
    def retrieve_controllers(expr: Beta, context: Any | None = None) -> list[Beta]:
        return [expr] if expr.is_free else []

    # Now use it
    return walker.walk(the_expression)


def list_of_fixed_betas_in_expression(
    the_expression: Expression,
) -> list[Beta]:
    # Create walker
    walker = ExpressionCollector()

    @walker.register(Beta)
    def retrieve_controllers(expr: Beta, context: Any | None = None) -> list[Beta]:
        return [] if expr.is_free else [expr]

    # Now use it
    return walker.walk(the_expression)


def list_of_random_variables_in_expression(
    the_expression: Expression,
) -> list[RandomVariable]:
    # Create walker
    walker = ExpressionCollector()

    @walker.register(RandomVariable)
    def retrieve_controllers(
        expr: RandomVariable, context: Any | None = None
    ) -> list[RandomVariable]:
        return [expr]

    # Now use it
    return walker.walk(the_expression)


def list_of_draws_in_expression(
    the_expression: Expression,
) -> list[Draws]:
    # Create walker
    walker = ExpressionCollector()

    @walker.register(Draws)
    def retrieve_controllers(expr: Draws, context: Any | None = None) -> list[Draws]:
        return [expr]

    # Now use it
    return walker.walk(the_expression)
