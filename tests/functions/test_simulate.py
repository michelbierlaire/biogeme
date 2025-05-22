import os
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from biogeme.database import Database
from biogeme.expressions import Beta, Variable, exp
from biogeme.results_processing import EstimationResults
from biogeme.tools import simulate


class DummyResults(EstimationResults):
    def __init__(self, beta_values):
        self.beta_values = beta_values

    def get_beta_values(self):
        return self.beta_values


class TestSimulateFunction(unittest.TestCase):

    def setUp(self):
        # Minimal dummy database
        self.data = pd.DataFrame({'x': [1, 2, 3]})
        self.db = Database('testdb', self.data)

        # Dummy expression
        self.beta = Beta('beta', 1.0, None, None, 0)
        self.expressions = {'exp_beta_x': exp(self.beta * Variable('x'))}

        # Dummy estimation result
        self.results = DummyResults(beta_values={self.beta.name: 2.0})

    def test_simulation_returns_dataframe(self):
        df = simulate(self.db, self.expressions, self.results)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self.data))
        self.assertIn('exp_beta_x', df.columns)

    def test_simulation_csv_output(self):
        filename = 'test_output.csv'
        try:
            df = simulate(
                self.db, self.expressions, self.results, csv_filename=filename
            )
            self.assertTrue(os.path.exists(filename))
            df_loaded = pd.read_csv(filename)
            assert_frame_equal(df, df_loaded, check_dtype=False)
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_simulation_with_invalid_expression(self):
        with self.assertRaises(Exception):
            simulate(self.db, {'bad': 'not an expression'}, self.results)


if __name__ == '__main__':
    unittest.main()
