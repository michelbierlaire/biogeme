"""Test the catalog object

Michel Bierlaire
Thu Apr 17 2025, 08:31:00
"""

import unittest

from biogeme.catalog import Catalog, Controller
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Expression, NamedExpression


class DummyExpression(Expression):
    """Minimal Expression subclass for testing purposes."""

    def __init__(self, label):
        self.label = label

    def get_all_controllers(self):
        return set()


class TestCatalog(unittest.TestCase):
    def test_valid_catalog_creation(self):
        expr1 = DummyExpression("e1")
        expr2 = DummyExpression("e2")
        named_exprs = [
            NamedExpression("A", expr1),
            NamedExpression("B", expr2),
        ]
        catalog = Catalog("myCatalog", named_exprs)
        self.assertIsNotNone(catalog.controlled_by)
        self.assertEqual(catalog.selected().name, "A")
        self.assertIs(catalog.selected().expression, expr1)

    def test_catalog_with_empty_list_raises(self):
        with self.assertRaisesRegex(
            BiogemeError, "cannot create a catalog from an empty list"
        ):
            Catalog("emptyCatalog", [])

    def test_catalog_with_wrong_controller_type_raises(self):
        expr = DummyExpression("e")
        named_exprs = [NamedExpression("A", expr)]
        with self.assertRaisesRegex(
            BiogemeError, "The controller must be of type Controller"
        ):
            Catalog("badController", named_exprs, controlled_by="not a controller")

    def test_controller_name_mismatch_raises(self):
        expr = DummyExpression("e")
        named_exprs = [NamedExpression("A", expr), NamedExpression("B", expr)]
        wrong_controller = Controller("myController", ["X", "Y"])
        with self.assertRaisesRegex(BiogemeError, "Incompatible IDs between catalog"):
            Catalog("mismatch", named_exprs, controlled_by=wrong_controller)

    def test_catalog_from_dict(self):
        expr1 = DummyExpression("e1")
        expr2 = DummyExpression("e2")
        expr_dict = {
            "A": expr1,
            "B": expr2,
        }
        catalog = Catalog.from_dict("dictCatalog", expr_dict)
        self.assertEqual(len(list(catalog.get_iterator())), 2)
        self.assertEqual(catalog.selected().name, "A")
        self.assertIsInstance(catalog.selected().expression, DummyExpression)

    def test_get_all_controllers_returns_controller(self):
        expr = DummyExpression("e")
        named_exprs = [NamedExpression("A", expr)]
        catalog = Catalog("controllerTest", named_exprs)
        controllers = catalog.get_all_controllers()
        self.assertIsInstance(controllers, set)
        self.assertIn(catalog.controlled_by, controllers)
