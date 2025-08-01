"""Tests for Expression

Michel Bierlaire
Tue Mar 25 17:43:42 2025
"""

import unittest

from biogeme.exceptions import BiogemeError
from biogeme.expressions import Expression, OldNewName, list_of_all_betas_in_expression
from biogeme.expressions.rename_variables import rename_all_variables


class DummyExpression(Expression):
    def __init__(self, label="dummy"):
        super().__init__()
        self.label = label

    def __str__(self):
        return self.label

    def recursive_construct_jax_function(self):
        raise NotImplementedError

    def get_signature(self):
        return [f"<Dummy>{{{id(self)}}}({len(self.children)})".encode()]


class TestExpression(unittest.TestCase):
    def setUp(self):
        self.expr1 = DummyExpression("A")
        self.expr2 = DummyExpression("B")

    def test_addition(self):
        result = self.expr1 + self.expr2
        self.assertEqual(result.get_class_name(), "Plus")

    def test_subtraction(self):
        result = self.expr1 - self.expr2
        self.assertEqual(result.get_class_name(), "Minus")

    def test_multiplication(self):
        result = self.expr1 * self.expr2
        self.assertEqual(result.get_class_name(), "Times")

    def test_division(self):
        result = self.expr1 / self.expr2
        self.assertEqual(result.get_class_name(), "Divide")

    def test_unary_minus(self):
        result = -self.expr1
        self.assertEqual(result.get_class_name(), "UnaryMinus")

    def test_power_expression(self):
        result = self.expr1**self.expr2
        self.assertEqual(result.get_class_name(), "Power")

    def test_power_constant(self):
        result = self.expr1**2
        self.assertEqual(result.get_class_name(), "PowerConstant")

    def test_logical_and(self):
        result = self.expr1 & self.expr2
        self.assertEqual(result.get_class_name(), "And")

    def test_logical_or(self):
        result = self.expr1 | self.expr2
        self.assertEqual(result.get_class_name(), "Or")

    def test_comparisons(self):
        self.assertEqual((self.expr1 == self.expr2).get_class_name(), "Equal")
        self.assertEqual((self.expr1 != self.expr2).get_class_name(), "NotEqual")
        self.assertEqual((self.expr1 < self.expr2).get_class_name(), "Less")
        self.assertEqual((self.expr1 <= self.expr2).get_class_name(), "LessOrEqual")
        self.assertEqual((self.expr1 > self.expr2).get_class_name(), "Greater")
        self.assertEqual((self.expr1 >= self.expr2).get_class_name(), "GreaterOrEqual")

    def test_set_and_dict_of_elementary_expression(self):
        expr = DummyExpression()
        the_betas = list_of_all_betas_in_expression(expr)
        self.assertFalse(the_betas)

    def test_get_elementary_expression_none(self):
        self.assertIsNone(self.expr1.get_elementary_expression("nonexistent"))

    def test_rename_elementary_does_not_fail(self):
        try:
            _ = rename_all_variables(
                expr=self.expr1,
                renaming_list=[OldNewName(old_name='x', new_name='pre_x_suf')],
            )
            _ = rename_all_variables(
                expr=self.expr1,
                renaming_list=[OldNewName(old_name='y', new_name='pre_y_suf')],
            )
        except Exception as e:
            self.fail(f"rename_elementary raised an exception unexpectedly: {e}")

    def test_boolean_context_raises(self):
        with self.assertRaises(BiogemeError):
            bool(self.expr1)


if __name__ == '__main__':
    unittest.main()
