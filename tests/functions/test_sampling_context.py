from dataclasses import dataclass
import unittest
import pandas as pd

from biogeme.sampling_of_alternatives import (
    StratumTuple,
    CrossVariableTuple,
    SamplingContext,
)
from biogeme.partition import Partition
from biogeme.expressions import Beta, Variable
from biogeme.exceptions import BiogemeError


class TestSamplingContext(unittest.TestCase):
    def setUp(self):
        self.valid_individuals = pd.DataFrame(
            {'choice': [1, 2, 3], 'age': [22, 33, 44]}
        )

        self.valid_alternatives = pd.DataFrame(
            {
                'id': [1, 2, 3, 5, 6],
                'travel_time': [12, 23, 34, 45, 56],
                'travel_cost': [13, 11, 9, 7, 5],
            }
        )

        self.valid_partition = Partition([{1, 2}, {3, 5, 6}])
        self.valid_sample_sizes = [1, 2]

        self.valid_utility_function = Beta('beta_cost', 0, None, None, 0) * Variable(
            'travel_cost'
        ) + Beta('beta_age_time', 0, None, None, 0) * Variable('age_time')

        self.valid_combined_variables = [
            CrossVariableTuple('age_time', Variable('age') * Variable('travel_time'))
        ]

    def test_check_expression(self):
        context = SamplingContext(
            the_partition=self.valid_partition,
            sample_sizes=self.valid_sample_sizes,
            individuals=self.valid_individuals,
            choice_column='choice',
            alternatives=self.valid_alternatives,
            id_column='id',
            biogeme_file_name='test.biogeme',
            utility_function=self.valid_utility_function,
            combined_variables=self.valid_combined_variables,
        )

        expression = Variable('any_variable')
        with self.assertRaisesRegex(BiogemeError, 'Invalid expression'):
            context.check_expression(expression)

    def test_check_partition(self):
        # Test valid partition
        context = SamplingContext(
            the_partition=self.valid_partition,
            sample_sizes=self.valid_sample_sizes,
            individuals=self.valid_individuals,
            choice_column='choice',
            alternatives=self.valid_alternatives,
            id_column='id',
            biogeme_file_name='test.biogeme',
            utility_function=self.valid_utility_function,
            combined_variables=self.valid_combined_variables,
        )

        context.check_partition()

        # Test invalid partition (empty stratum)
        invalid_partition = [(), ('b', 'c')]
        with self.assertRaisesRegex(Exception, 'A stratum is empty'):
            context = SamplingContext(
                the_partition=invalid_partition,
                sample_sizes=self.valid_sample_sizes,
                individuals=self.valid_individuals,
                choice_column='choice',
                alternatives=self.valid_alternatives,
                id_column='id',
                biogeme_file_name='test.biogeme',
                utility_function=self.valid_utility_function,
                combined_variables=self.valid_combined_variables,
            )

        # Test invalid partition (alternative not in DB)
        invalid_partition = [('x',), ('b', 'c')]
        with self.assertRaisesRegex(
            Exception, 'Alternative x does not appear in the database of alternatives'
        ):
            context = SamplingContext(
                the_partition=invalid_partition,
                sample_sizes=self.valid_sample_sizes,
                individuals=self.valid_individuals,
                choice_column='choice',
                alternatives=self.valid_alternatives,
                id_column='id',
                biogeme_file_name='test.biogeme',
                utility_function=self.valid_utility_function,
                combined_variables=self.valid_combined_variables,
            )

    def test_check_mev_partition(self):
        # Test valid partition
        context = SamplingContext(
            the_partition=self.valid_partition,
            sample_sizes=self.valid_sample_sizes,
            individuals=self.valid_individuals,
            choice_column='choice',
            alternatives=self.valid_alternatives,
            id_column='id',
            biogeme_file_name='test.biogeme',
            utility_function=self.valid_utility_function,
            combined_variables=self.valid_combined_variables,
            mev_partition=self.valid_partition,
            mev_sample_sizes=self.valid_sample_sizes
        )

        context.check_mev_partition()

        # Test invalid partition
        with self.assertRaisesRegex(Exception, 'If mev_partition'):
            context = SamplingContext(
                the_partition=self.valid_partition,
                sample_sizes=self.valid_sample_sizes,
                individuals=self.valid_individuals,
                choice_column='choice',
                alternatives=self.valid_alternatives,
                id_column='id',
                biogeme_file_name='test.biogeme',
                utility_function=self.valid_utility_function,
                combined_variables=self.valid_combined_variables,
                mev_partition=self.valid_partition,
                mev_sample_sizes=None
            )

        with self.assertRaisesRegex(Exception, 'If mev_sample_sizes'):
            context = SamplingContext(
                the_partition=self.valid_partition,
                sample_sizes=self.valid_sample_sizes,
                individuals=self.valid_individuals,
                choice_column='choice',
                alternatives=self.valid_alternatives,
                id_column='id',
                biogeme_file_name='test.biogeme',
                utility_function=self.valid_utility_function,
                combined_variables=self.valid_combined_variables,
                mev_partition=None,
                mev_sample_sizes=self.valid_sample_sizes
            )
            
        # Test invalid partition (alternative not in DB)
        invalid_partition = [('x',), ('b', 'c')]
        with self.assertRaisesRegex(Exception, 'Alternative x does not appear in the database of alternatives'):
            context = SamplingContext(
                the_partition=invalid_partition,
                sample_sizes=self.valid_sample_sizes,
                individuals=self.valid_individuals,
                choice_column='choice',
                alternatives=self.valid_alternatives,
                id_column='id',
                biogeme_file_name='test.biogeme',
                utility_function=self.valid_utility_function,
                combined_variables=self.valid_combined_variables,
                mev_partition=invalid_partition,
                mev_sample_sizes=self.valid_sample_sizes
            )


if __name__ == '__main__':
    unittest.main()
