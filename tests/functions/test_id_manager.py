"""
Unit tests for the IdManager class

Michel Bierlaire
Thu Jul 4 13:39:58 2024
"""

import unittest
from unittest.mock import MagicMock

import pandas as pd

from biogeme.database import Database
from biogeme.exceptions import BiogemeError
from biogeme.expressions import (
    IdManager,
    Expression,
    Beta,
    bioDraws,
)
from biogeme.expressions.idmanager import ElementsTuple


class TestIdManager(unittest.TestCase):

    def setUp(self):
        self.database = MagicMock(spec=Database)
        self.database.data = pd.DataFrame({'var1': [1, 2, 3], 'var2': [4, 5, 6]})
        self.database.is_panel.return_value = False

        self.expr1 = MagicMock(spec=Expression)
        self.expr1.set_of_elementary_expression.return_value = set()
        self.expr1.embed_expression.side_effect = lambda x: x == 'MonteCarlo'

        self.expr2 = MagicMock(spec=Expression)
        self.expr2.set_of_elementary_expression.return_value = set()
        self.expr2.embed_expression.side_effect = lambda x: x == 'bioDraws'

        self.expressions = [self.expr1, self.expr2]

    def test_valid_initialization(self):
        manager = IdManager(
            expressions=self.expressions, database=self.database, number_of_draws=1000
        )
        self.assertIsInstance(manager, IdManager)

    def test_initialization_without_database(self):
        self.expr1.set_of_elementary_expression.return_value = {'var1'}
        with self.assertRaises(BiogemeError):
            IdManager(expressions=self.expressions, database=None, number_of_draws=1000)

    def test_prepare(self):
        manager = IdManager(
            expressions=self.expressions, database=self.database, number_of_draws=1000
        )
        self.assertEqual(len(manager.free_betas.names), 0)
        self.assertEqual(len(manager.fixed_betas.names), 0)
        self.assertEqual(len(manager.random_variables.names), 0)
        self.assertEqual(len(manager.draws.names), 0)
        self.assertEqual(len(manager.variables.names), 2)

    def test_draw_types(self):
        draw_expression = MagicMock(spec=bioDraws)
        draw_expression.drawType = 'type1'
        self.expressions.append(draw_expression)

        manager = IdManager(
            expressions=self.expressions, database=self.database, number_of_draws=1000
        )
        manager.draws = ElementsTuple(
            expressions={'draw1': draw_expression},
            indices={'draw1': 0},
            names=['draw1'],
        )

        draw_types = manager.draw_types()
        self.assertEqual(draw_types['draw1'], 'type1')

    def test_audit(self):
        self.database.is_panel.return_value = True
        self.expr1.check_panel_trajectory.return_value = {'var1': 'issue'}

        manager = IdManager(
            expressions=self.expressions, database=self.database, number_of_draws=1000
        )
        errors, warnings = manager.audit()

        self.assertEqual(len(errors), 2)
        self.assertEqual(len(warnings), 0)

    def test_eq(self):
        manager1 = IdManager(
            expressions=self.expressions, database=self.database, number_of_draws=1000
        )
        manager2 = IdManager(
            expressions=self.expressions, database=self.database, number_of_draws=1000
        )

        self.assertEqual(manager1, manager2)

    def test_repr(self):
        manager = IdManager(
            expressions=self.expressions, database=self.database, number_of_draws=1000
        )
        self.assertIsInstance(repr(manager), str)


if __name__ == '__main__':
    unittest.main()
