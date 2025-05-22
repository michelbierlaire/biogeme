"""Tests of the objects for visiting arithmetic expressions

Michel Bierlaire
15.04.2025 16:56
"""

import unittest
from typing import Any

from biogeme.expressions import Expression
from biogeme.expressions.collectors import ExpressionCollector
from biogeme.expressions.visitor import ExpressionVisitor


class DummyExpression(Expression):
    """A dummy expression used for testing purposes."""

    def __init__(self, name: str, children=None):
        self.name = name
        self._children = children or []
        super().__init__()

    def get_children(self) -> list[Expression]:
        return self._children


class TestExpressionWalkerVisitor(unittest.TestCase):
    def setUp(self):
        self.leaf = DummyExpression("leaf")
        self.child = DummyExpression("child", [self.leaf])
        self.root = DummyExpression("root", [self.child])

    def test_expression_walker_collects_results(self):
        walker = ExpressionCollector()

        @walker.register(DummyExpression)
        def handler(expr: DummyExpression, context: Any) -> list[str]:
            return [f"handled: {expr.name}"]

        results = walker.walk(self.root)
        expected = ["handled: root", "handled: child", "handled: leaf"]
        self.assertEqual(results, expected)

    def test_expression_walker_invalid_handler_output(self):
        walker = ExpressionCollector()

        @walker.register(DummyExpression)
        def handler(expr: DummyExpression, context: Any):
            return "invalid"

        with self.assertRaises(TypeError):
            walker.walk(self.root)

    def test_expression_visitor_executes_handlers(self):
        visitor = ExpressionVisitor()
        visited = []

        @visitor.register(DummyExpression)
        def handler(expr: DummyExpression, context: Any) -> None:
            visited.append(expr.name)

        visitor.visit(self.root)
        self.assertEqual(visited, ["root", "child", "leaf"])

    def test_expression_visitor_invalid_handler_output(self):
        visitor = ExpressionVisitor()

        @visitor.register(DummyExpression)
        def handler(expr: DummyExpression, context: Any) -> str:
            return "not None"

        with self.assertRaises(TypeError):
            visitor.visit(self.root)


class TestExpressionVisitor(unittest.TestCase):
    def setUp(self):
        # Create a simple tree: root -> [child -> [leaf]]
        self.leaf = DummyExpression("leaf")
        self.child = DummyExpression("child", [self.leaf])
        self.root = DummyExpression("root", [self.child])

    def test_basic_traversal(self):
        visitor = ExpressionVisitor()
        visited = []

        @visitor.register(DummyExpression)
        def handle(expr: DummyExpression, context: Any) -> None:
            visited.append(expr.name)

        visitor.visit(self.root)
        self.assertEqual(visited, ["root", "child", "leaf"])

    def test_no_handler(self):
        # Test that traversal works fine with no registered handlers
        visitor = ExpressionVisitor()
        try:
            visitor.visit(self.root)
        except Exception as e:
            self.fail(f"Visitor raised an exception unexpectedly: {e}")

    def test_handler_must_return_none(self):
        visitor = ExpressionVisitor()

        @visitor.register(DummyExpression)
        def invalid_handler(expr: DummyExpression, context: Any) -> str:
            return "not none"

        with self.assertRaises(TypeError) as context:
            visitor.visit(self.root)
        self.assertIn("must return None", str(context.exception))

    def test_partial_handler_registration(self):
        visitor = ExpressionVisitor()
        visited = []

        class AnotherExpression(Expression):
            def get_children(self) -> list[Expression]:
                return []

        @visitor.register(DummyExpression)
        def dummy_handler(expr: DummyExpression, context: Any) -> None:
            visited.append(expr.name)

        mixed = DummyExpression("dummy", [AnotherExpression()])
        visitor.visit(mixed)
        self.assertEqual(visited, ["dummy"])  # Only DummyExpression triggers handler


if __name__ == "__main__":
    unittest.main()
