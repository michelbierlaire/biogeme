"""Tests for DataContainer

Michel Bierlaire
Tue Mar 25 17:43:42 2025
"""

import unittest

import pandas as pd


from biogeme.database import Database
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Variable
from biogeme.floating_point import PANDAS_FLOAT


class TestDataContainer(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(
            {
                'x': [1, 2, 3],
                'y': [4, 5, 6],
            },
            dtype=PANDAS_FLOAT,
        )
        self.container = Database('self', self.df.copy())

    def test_init_valid(self):
        self.assertEqual(self.container.num_rows(), 3)
        self.assertEqual(self.container.num_columns(), 2)

    def test_init_empty(self):
        with self.assertRaises(BiogemeError):
            Database('test', pd.DataFrame())

    def test_get_column_valid(self):
        col = self.container.get_column('x')
        pd.testing.assert_series_equal(
            col.reset_index(drop=True),
            pd.Series([1, 2, 3], name='x', dtype=PANDAS_FLOAT),
        )

    def test_get_column_invalid(self):
        with self.assertRaises(BiogemeError):
            self.container.get_column('z')

    def test_column_exists(self):
        self.assertTrue(self.container.column_exists('x'))
        self.assertFalse(self.container.column_exists('z'))

    def test_add_column_success(self):
        new_col = pd.Series([7, 8, 9])
        self.container.add_column('z', new_col)
        self.assertTrue(self.container.column_exists('z'))

    def test_add_column_duplicate(self):
        with self.assertRaises(ValueError):
            self.container.add_column('x', pd.Series([1, 2, 3]))

    def test_add_column_wrong_length(self):
        with self.assertRaises(ValueError):
            self.container.add_column('z', pd.Series([1, 2]))

    def test_scale_column_success(self):
        self.container.scale_column('x', 2)
        expected = pd.Series([2, 4, 6], name='x', dtype=PANDAS_FLOAT)
        pd.testing.assert_series_equal(self.container.get_column('x'), expected)

    def test_scale_column_missing(self):
        with self.assertRaises(BiogemeError):
            self.container.scale_column('z', 2)

    def test_remove_column(self):
        self.container.remove_column('x')
        self.assertFalse(self.container.column_exists('x'))

    def test_remove_rows(self):
        condition = self.df['x'] > 1
        self.container.remove_rows(condition)
        self.assertEqual(self.container.num_rows(), 1)
        pd.testing.assert_series_equal(
            self.container.get_column('x'), pd.Series([1], name='x', dtype=PANDAS_FLOAT)
        )

    def test_remove_with_expression(self):

        x = Variable('x')
        condition = x > 1
        self.container.remove(condition)
        self.assertEqual(self.container.num_rows(), 1)
        pd.testing.assert_series_equal(
            self.container.get_column('x'), pd.Series([1], name='x', dtype=PANDAS_FLOAT)
        )

    def test_dataframe_property(self):
        df = self.container.dataframe
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ['x', 'y'])

    def test_define_variable(self):
        x = Variable('x')
        expr = x * 10
        self.container.define_variable('x_scaled', expr)
        expected = pd.Series([10.0, 20.0, 30.0], name='x_scaled', dtype=PANDAS_FLOAT)
        pd.testing.assert_series_equal(self.container.get_column('x_scaled'), expected)

    def test_dataframe_dtype(self):
        # Check that all columns have the correct dtype
        for col in self.container.dataframe.columns:
            self.assertEqual(self.container.dataframe[col].dtype, PANDAS_FLOAT)

    def test_bootstrap_sample_dtype(self):
        # Check that the bootstrap sample also uses the correct float type
        bootstrap = self.container.bootstrap_sample()
        for col in bootstrap.dataframe.columns:
            self.assertEqual(bootstrap.dataframe[col].dtype, PANDAS_FLOAT)

    def test_define_variable_dtype(self):
        # Define a new variable and check its dtype
        x = Variable('x')
        self.container.define_variable('x_squared', x * x)
        self.assertEqual(self.container.get_column('x_squared').dtype, PANDAS_FLOAT)

    def test_scale_column_dtype(self):
        # Scale a column and verify dtype remains correct
        self.container.scale_column('x', 1.5)
        self.assertEqual(self.container.get_column('x').dtype, PANDAS_FLOAT)

    def test_add_column_dtype(self):
        # Add a column and check its dtype
        new_col = pd.Series([0.1, 0.2, 0.3], dtype=PANDAS_FLOAT)
        self.container.add_column('z', new_col)
        self.assertEqual(self.container.get_column('z').dtype, PANDAS_FLOAT)


if __name__ == '__main__':
    unittest.main()
