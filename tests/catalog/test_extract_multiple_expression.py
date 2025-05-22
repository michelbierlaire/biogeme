import unittest

from biogeme.expressions import Variable
from biogeme.catalog import (
    Catalog,
    Controller,
    extract_multiple_expressions_controllers,
)


class TestExtractMultipleExpressions(unittest.TestCase):
    def test_no_catalog(self):
        """Test with a simple variable expression (no Catalog)"""
        expr = Variable('x')
        result = extract_multiple_expressions_controllers(expr)
        self.assertEqual(result, [])

    def test_one_catalog(self):
        """Test with one catalog expression"""
        expr1 = Variable('x')
        expr2 = Variable('y')
        controller = Controller(controller_name='ctrl', specification_names=['x', 'y'])
        expressions = {
            'x': expr1,
            'y': expr2,
        }
        catalog = Catalog.from_dict(
            catalog_name='C1', dict_of_expressions=expressions, controlled_by=controller
        )

        result = extract_multiple_expressions_controllers(catalog)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], controller)

    def test_nested_catalog(self):
        """Test with a nested catalog inside another"""
        expr1 = Variable('x')
        expr2 = Variable('y')
        controller_1 = Controller(
            controller_name='ctrl_1', specification_names=['x', 'y']
        )
        expressions_1 = {
            'x': expr1,
            'y': expr2,
        }
        catalog_1 = Catalog.from_dict(
            catalog_name='C1',
            dict_of_expressions=expressions_1,
            controlled_by=controller_1,
        )

        expr3 = Variable('z')
        expressions_2 = {
            'z': expr3,
            'c1': catalog_1,
        }
        controller_2 = Controller(
            controller_name='ctrl_2', specification_names=['z', 'c1']
        )
        catalog_2 = Catalog.from_dict(
            catalog_name='C2',
            dict_of_expressions=expressions_2,
            controlled_by=controller_2,
        )

        result = extract_multiple_expressions_controllers(catalog_2)
        self.assertEqual(set(result), {controller_1, controller_2})

    def test_duplicate_catalog(self):
        """Test that duplicate catalogs are returned separately"""
        expr1 = Variable('x')
        expr2 = Variable('y')
        controller = Controller(controller_name='ctrl', specification_names=['x', 'y'])
        expressions = {
            'x': expr1,
            'y': expr2,
        }
        catalog = Catalog.from_dict(
            catalog_name='C1', dict_of_expressions=expressions, controlled_by=controller
        )
        composite_expr = catalog + catalog

        result = extract_multiple_expressions_controllers(composite_expr)
        self.assertEqual(result.count(controller), 2)


if __name__ == "__main__":
    unittest.main()
