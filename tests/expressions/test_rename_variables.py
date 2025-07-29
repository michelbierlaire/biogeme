"""Tests for renaming variables

Michel Bierlaire
Fri Jul 25 2025, 17:57:40
"""

import unittest

from biogeme.expressions import OldNewName, Variable, rename_all_variables


class TestRenameAllVariables(unittest.TestCase):

    def test_single_variable_rename(self):
        expr = Variable('x')
        count = rename_all_variables(expr, [OldNewName(old_name='x', new_name='y')])
        self.assertEqual(expr.name, 'y')
        self.assertEqual(count, 1)

    def test_no_rename_needed(self):
        expr = Variable('x')
        count = rename_all_variables(expr, [OldNewName(old_name='a', new_name='b')])
        self.assertEqual(expr.name, 'x')
        self.assertEqual(count, 0)

    def test_nested_expression(self):
        expr = Variable('x') + Variable('x')
        count = rename_all_variables(expr, [OldNewName(old_name='x', new_name='z')])
        self.assertIsInstance(expr.left, Variable)
        self.assertIsInstance(expr.right, Variable)
        self.assertEqual(expr.left.name, 'z')
        self.assertEqual(expr.right.name, 'z')
        self.assertEqual(count, 2)

    def test_complex_expression(self):
        expr = Variable('x') ** 2
        count = rename_all_variables(expr, [OldNewName(old_name='x', new_name='z')])
        self.assertIsInstance(expr.child, Variable)
        self.assertEqual(expr.child.name, 'z')
        self.assertEqual(count, 1)

    def test_multiple_variable_renames(self):
        expr = Variable('x') + Variable('y')
        renames = [
            OldNewName(old_name='x', new_name='u'),
            OldNewName(old_name='y', new_name='v'),
        ]
        count = rename_all_variables(expr, renames)
        self.assertEqual(expr.left.name, 'u')
        self.assertEqual(expr.right.name, 'v')
        self.assertEqual(count, 2)


if __name__ == '__main__':
    unittest.main()
