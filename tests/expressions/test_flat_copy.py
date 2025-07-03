"""Tests for deep and flat copy

Michel Bierlaire
Fri Jun 27 2025, 17:30:22
"""

import unittest

import pandas as pd
from biogeme.catalog import Catalog, CentralController, segmentation_catalogs
from biogeme.database import Database
from biogeme.expressions import (
    BelongsTo,
    Beta,
    BinaryMax,
    BinaryMin,
    Numeric,
    Variable,
    log,
)
from biogeme.models import loglogit


class TestExpressions(unittest.TestCase):
    def test_belongs_to(self):
        the_set = {1, 2, 3}
        value = Numeric(1)
        expression = BelongsTo(child=value, the_set=the_set)
        copy_expression = expression.deep_flat_copy()
        self.assertEqual(expression.the_set, copy_expression.the_set)
        self.assertEqual(
            expression.child.get_value(), copy_expression.child.get_value()
        )

    def test_belongs_to_with_numeric(self):
        the_set = {1, 2, 3}
        expression = BelongsTo(child=1, the_set=the_set)
        copy_expression = expression.deep_flat_copy()
        self.assertEqual(expression.the_set, copy_expression.the_set)
        self.assertEqual(
            expression.child.get_value(), copy_expression.child.get_value()
        )

    def test_beta(self):
        beta = Beta(name='test', value=0.1, lowerbound=-1, upperbound=1, status=0)
        copy_beta = beta.deep_flat_copy()
        self.assertEqual(beta.name, copy_beta.name)
        self.assertEqual(beta.init_value, copy_beta.init_value)
        self.assertEqual(beta.lower_bound, copy_beta.lower_bound)
        self.assertEqual(beta.upper_bound, copy_beta.upper_bound)
        self.assertEqual(beta.status, copy_beta.status)

    def test_max(self):
        expression = BinaryMax(1, 2)
        copy_expression = expression.deep_flat_copy()
        self.assertEqual(expression.left.get_value(), copy_expression.left.get_value())
        self.assertEqual(
            expression.right.get_value(), copy_expression.right.get_value()
        )

    def test_min(self):
        expression = BinaryMin(1, 2)
        copy_expression = expression.deep_flat_copy()
        self.assertEqual(expression.left.get_value(), copy_expression.left.get_value())
        self.assertEqual(
            expression.right.get_value(), copy_expression.right.get_value()
        )


class TestMultipleExpressions(unittest.TestCase):
    def setUp(self):
        data = {
            'choice': [0, 1, 2, 1, 0],
            'male': [0, 1, 0, 1, 0],
            'train_tt': [15.593820, 11.937003, 8.916655, 1.999498, 9.184978],
            'train_cost': [6.674172, 2.857336, 13.017769, 1.128232, 14.439975],
            'sm_tt': [18.771054, 0.015575, 19.844231, 12.349630, 12.233063],
            'sm_cost': [0.141326, 0.461249, 10.495493, 7.997219, 0.933313],
            'car_tt': [19.475110, 4.655427, 1.812129, 12.367720, 7.649240],
            'car_cost': [19.664618, 9.335258, 17.198808, 13.606151, 9.009985],
        }

        df = pd.DataFrame(data)

        self.database = Database(name='test', dataframe=df)
        asc_car = Beta('asc_car', 0, None, None, 0)
        asc_train = Beta('asc_train', 0, None, None, 0)
        b_time = Beta('b_time', 0, None, None, 0)
        b_cost = Beta('b_cost', 0, None, None, 0)

        # %%
        segmentation_gender = self.database.generate_segmentation(
            variable=Variable('male'), mapping={0: 'female', 1: 'male'}
        )

        # %%
        # We define catalogs with two different specifications for the
        # ASC_CAR: non segmented, and segmented.
        asc_train_catalog, asc_car_catalog = segmentation_catalogs(
            generic_name='asc',
            beta_parameters=[asc_train, asc_car],
            potential_segmentations=(segmentation_gender,),
            maximum_number=1,
        )

        # %%
        # We now define a catalog  with the log travel time as well as the travel time.

        # %%
        # First for train
        train_tt_catalog = Catalog.from_dict(
            catalog_name='train_tt_catalog',
            dict_of_expressions={
                'linear': Variable('train_tt'),
                'log': log(Variable('train_tt')),
            },
        )
        # %%
        # Then for SM. But we require that the specification is the same as
        # train by defining the same controller.
        sm_tt_catalog = Catalog.from_dict(
            catalog_name='sm_tt_catalog',
            dict_of_expressions={
                'linear': Variable('sm_tt'),
                'log': Variable('sm_tt'),
            },
            controlled_by=train_tt_catalog.controlled_by,
        )

        # %%
        # Definition of the utility functions with linear cost.
        v_train = (
            asc_train_catalog
            + b_time * train_tt_catalog
            + b_cost * Variable('train_cost')
        )
        v_swissmetro = b_time * sm_tt_catalog + b_cost * Variable('sm_cost')
        v_car = (
            asc_car_catalog
            + b_time * Variable('car_tt')
            + b_cost * Variable('car_cost')
        )

        # %%
        # Associate utility functions with the numbering of alternatives.
        v = {1: v_train, 2: v_swissmetro, 3: v_car}

        # %%
        # Definition of the model. This is the contribution of each
        # observation to the log likelihood function.
        self.log_probability = loglogit(v, None, Variable('choice'))

        central_controller = CentralController(
            expression=self.log_probability, maximum_number_of_configurations=10
        )
        self.the_iterator = central_controller.expression_iterator()

    def test_multiple_expressions(self):
        self.assertTrue(self.log_probability.embed_expression(Catalog))
        for named_expression in self.the_iterator:
            flat_expression = named_expression.expression.deep_flat_copy()
            self.assertFalse(flat_expression.embed_expression(Catalog))
