import unittest
from unittest.mock import Mock

import pandas as pd

from biogeme.expressions import Beta, Expression, Variable
from biogeme.nests import (
    NestsForCrossNestedLogit,
    NestsForNestedLogit,
    OneNestForCrossNestedLogit,
    OneNestForNestedLogit,
)
from biogeme.sampling_of_alternatives import (
    CrossVariableTuple,
    GenerateModel,
    SamplingContext,
)


class TestGenerateModel(unittest.TestCase):
    def setUp(self):
        # Mocking the context and other required data
        self.mock_context = Mock(spec=SamplingContext)
        self.mock_context.utility_function = Beta('Beta', 0, None, None, 0) * Variable(
            'x'
        )
        self.mock_context.total_sample_size = 5
        self.mock_context.total_mev_sample_size = 4
        self.mock_context.attributes = ['att1', 'att2']
        self.mock_context.mev_prefix = 'mev'
        self.mock_context.mev_partition = None
        self.mock_context.id_column = 'id'

        # Nests for nested logit tests
        a_nest = OneNestForNestedLogit(
            nest_param=Beta('mu', 1, 1, None, 0),
            list_of_alternatives=[1, 5],
            name='any_nest',
        )
        self.nests = NestsForNestedLogit(
            choice_set={1, 2, 3, 4, 5}, tuple_of_nests=(a_nest,)
        )
        first_nest = OneNestForCrossNestedLogit(
            nest_param=Beta('mu_1', 1, 1, None, 0),
            dict_of_alpha={1: 1.0, 3: 0.5},
            name='first_nest',
        )
        second_nest = OneNestForCrossNestedLogit(
            nest_param=Beta('mu_2', 1, 1, None, 0),
            dict_of_alpha={3: 0.5, 5: 1},
            name='second_nest',
        )
        self.cnl_nests = NestsForCrossNestedLogit(
            choice_set={1, 2, 3, 4, 5}, tuple_of_nests=(first_nest, second_nest)
        )

        self.individuals = pd.DataFrame(
            {'choice': [1, 2, 3, 1, 5], 'age': [22, 33, 44, 55, 66]}
        )

        self.alternatives = pd.DataFrame(
            {
                'id': [1, 2, 3, 4, 5],
                'travel_time': [12, 23, 34, 45, 56],
                'travel_cost': [13, 11, 9, 7, 5],
            }
        )

        self.partition = [(1, 2), (3, 4, 5)]
        self.sample_sizes = [1, 2]

        self.utility_function = Beta('beta_cost', 0, None, None, 0) * Variable(
            'travel_cost'
        ) + Beta('beta_age_time', 0, None, None, 0) * Variable('age_time')

        self.combined_variables = [
            CrossVariableTuple('age_time', Variable('age') * Variable('travel_time'))
        ]

    def test_generate_utility(self):
        context = SamplingContext(
            the_partition=self.partition,
            sample_sizes=self.sample_sizes,
            individuals=self.individuals,
            choice_column='choice',
            alternatives=self.alternatives,
            id_column='id',
            biogeme_file_name='tmp_file.dat',
            utility_function=self.utility_function,
            combined_variables=self.combined_variables,
        )
        model = GenerateModel(context)
        utility = model.generate_utility(prefix='prefix_', suffix='_suffix')
        self.assertIsInstance(utility, Expression)
        utility_str = str(utility)
        expected = (
            "((Beta('beta_cost', 0, None, None, "
            "0) * prefix_travel_cost_suffix) + (Beta('beta_age_time', 0, "
            "None, None, 0) * "
            "prefix_age_time_suffix))"
        )
        self.assertEqual(utility_str, expected)

    def test_get_logit(self):
        context = SamplingContext(
            the_partition=self.partition,
            sample_sizes=self.sample_sizes,
            individuals=self.individuals,
            choice_column='choice',
            alternatives=self.alternatives,
            id_column='id',
            biogeme_file_name='tmp_file.dat',
            utility_function=self.utility_function,
            combined_variables=self.combined_variables,
        )
        model = GenerateModel(context)
        logit_expression = model.get_logit()
        self.assertIsInstance(logit_expression, Expression)
        expected_start = 'LogLogit[choice=`0.0`]'
        self.assertTrue(str(logit_expression).startswith(expected_start))

    def test_get_nested_logit(self):
        model = GenerateModel(self.mock_context)
        nested_logit_expression = model.get_nested_logit(self.nests)
        self.assertIsInstance(nested_logit_expression, Expression)
        expected = 'LogLogit[choice=`0.0`]'
        self.assertTrue(str(nested_logit_expression).startswith(expected))

    def test_get_cross_nested_logit(self):
        self.mock_context.cnl_nests = self.cnl_nests
        model = GenerateModel(self.mock_context)
        cross_nested_logit_expression = model.get_cross_nested_logit()
        self.assertIsInstance(cross_nested_logit_expression, Expression)
        expected = 'LogLogit[choice=`0.0`]'
        self.assertTrue(str(cross_nested_logit_expression).startswith(expected))


if __name__ == '__main__':
    unittest.main()
