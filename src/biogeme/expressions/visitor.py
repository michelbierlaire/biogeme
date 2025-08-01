"""Function to visit the tree of Expressions

Michel Bierlaire
15.04.2025 12:55
"""

from typing import Any

from .base_expressions import Expression


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
        if context is None:
            context = {}
        if 'ancestors' not in context:
            context['ancestors'] = []
        context['ancestors'].append(expr)

        handler = self._registry.get(type(expr))
        if handler:
            result = handler(expr, context)
            if result is not None:
                raise TypeError(
                    f"Handler for {type(expr).__name__} must return None, got {type(result).__name__}"
                )
        for child in expr.get_children():
            self.visit(child, context)

        context['ancestors'].pop()
