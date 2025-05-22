import os
import tempfile
import unittest

import pandas as pd

from biogeme.database import Database
from biogeme.expressions import Variable
from biogeme.partition import Partition
from biogeme.sampling_of_alternatives import (
    ChoiceSetsGeneration,
    CrossVariableTuple,
    SamplingContext,
)


class TestChoiceSetsGeneration(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(test_dir, 'test_file.csv')

        self.individuals = pd.DataFrame({'choice': [1, 2, 1], 'age': [25, 30, 35]})
        alternatives = pd.DataFrame({'alt_id': [1, 2, 3, 4], 'cost': [10, 20, 30, 40]})
        the_partition = Partition([{1, 2}, {3, 4}])
        sample_sizes = [1, 1]
        mev_partition = Partition([{1, 3}, {2, 4}])
        mev_sample_sizes = [1, 1]
        context = SamplingContext(
            the_partition=the_partition,
            sample_sizes=sample_sizes,
            mev_partition=mev_partition,
            mev_sample_sizes=mev_sample_sizes,
            individuals=self.individuals,
            choice_column='choice',
            alternatives=alternatives,
            biogeme_file_name=self.test_file,
            id_column='alt_id',
            utility_function=Variable('choice'),
            combined_variables=[],
        )
        self.choice_set_generator = ChoiceSetsGeneration(context)

    def test_initialization(self):
        self.assertIsNotNone(self.choice_set_generator)

    def test_get_attributes_from_expression(self):
        expr = Variable('cost') + Variable('unknown_var')
        result = self.choice_set_generator.get_attributes_from_expression(expr)
        self.assertEqual(result, {'cost'})

    def test_process_row(self):
        row = self.individuals.iloc[0]
        processed_data = self.choice_set_generator.process_row(row)
        self.assertIn('choice', processed_data)
        self.assertIn('age', processed_data)

    def test_sample_and_merge(self):
        the_database = self.choice_set_generator.sample_and_merge(recycle=False)
        self.assertTrue(os.path.exists(self.test_file))

        df = pd.read_csv(self.test_file)
        expected_columns = [
            'choice',
            'age',
            'alt_id_0',
            'cost_0',
            '_log_proba_0',
            'alt_id_1',
            'cost_1',
            '_log_proba_1',
            '_MEV_alt_id_0',
            '_MEV_cost_0',
            '_MEV__mev_weight_0',
            '_MEV_alt_id_1',
            '_MEV_cost_1',
            '_MEV__mev_weight_1',
        ]
        self.assertListEqual(sorted(expected_columns), sorted(df.columns))
        os.remove(self.test_file)  # cleanup

    def test_define_new_variables(self):
        # Create a dummy database
        biogeme_data = pd.DataFrame({'var1': [1, 2, 3], 'var2': [4, 5, 6]})
        biogeme_database = Database('test_data', biogeme_data)

        # Set a sample size and dummy combined variables in the generator
        self.choice_set_generator.sample_size = 2
        self.choice_set_generator.combined_variables = [
            CrossVariableTuple(
                name='new_var', formula=Variable('var1') + Variable('var2')
            )
        ]

        self.choice_set_generator.define_new_variables(biogeme_database)

        # The new variable names should be new_var_0 and new_var_1 based on the sample size of 2
        defined_variables = biogeme_database.dataframe.columns
        self.assertIn('new_var_0', defined_variables)
        self.assertIn('new_var_1', defined_variables)


if __name__ == '__main__':
    unittest.main()
