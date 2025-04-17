"""Function to visit the tree of Expressions

Michel Bierlaire
15.04.2025 12:55
"""

from typing import Any

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

    def _visit(self, expr: Expression, context: Any) -> list[Any]:
        """
        Recursively visit expressions and collect handler results.

        Each handler function must return a list. If multiple handlers are triggered
        during the traversal, their outputs are flattened into a single list.

        :param expr: current expression node
        :param context: context passed down to handler functions
        :return: list of results from handler invocations
        """
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
        return results


class ExpressionVisitor:
    """Walks the tree and executes side-effect handlers with no return values."""

    def __init__(self):
        self._registry = {}

    def register(self, expr_type: type[Expression]):
        """
        Register a handler function for a specific expression type. The handler must
        return None and perform side effects only.

        :param expr_type: the type of Expression for which the handler should be used
        :return: decorator that registers the handler
        """

        def decorator(func):
            self._registry[expr_type] = func
            return func

        return decorator

    def visit(self, expr: Expression, context: Any = None) -> None:
        """
        Traverse the expression tree and apply handlers for side effects.

        :param expr: the root expression to visit
        :param context: optional context object passed to handlers
        """
        handler = self._registry.get(type(expr))
        if handler:
            result = handler(expr, context)
            if result is not None:
                raise TypeError(
                    f"Handler for {type(expr).__name__} must return None, got {type(result).__name__}"
                )
        for child in expr.get_children():
            self.visit(child, context)
