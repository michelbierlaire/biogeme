import unittest

import pandas as pd
from biogeme.catalog import Catalog, segmentation_catalogs
from biogeme.database import Database
from biogeme.expressions import (
    Beta,
    MonteCarlo,
    MultipleExpression,
    Variable,
    log,
)
from biogeme.models import loglogit


class TestEmbedExpression(unittest.TestCase):
    def test_embed_expression(self):
        beta_1 = Beta('beta_1', 0, None, None, 0)
        var_1 = Variable('var_1')
        expression = beta_1 * var_1
        self.assertTrue(expression.embed_expression(Beta))
        self.assertTrue(expression.embed_expression(Variable))
        self.assertFalse(expression.embed_expression(MonteCarlo))


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
        self.asc_train_catalog, self.asc_car_catalog = segmentation_catalogs(
            generic_name='asc',
            beta_parameters=[asc_train, asc_car],
            potential_segmentations=(segmentation_gender,),
            maximum_number=1,
        )

        # %%
        # We now define a catalog  with the log travel time as well as the travel time.

        # %%
        # First for train
        self.train_tt_catalog = Catalog.from_dict(
            catalog_name='train_tt_catalog',
            dict_of_expressions={
                'linear': Variable('train_tt'),
                'log': log(Variable('train_tt')),
            },
        )
        # %%
        # Then for SM. But we require that the specification is the same as
        # train by defining the same controller.
        self.sm_tt_catalog = Catalog.from_dict(
            catalog_name='sm_tt_catalog',
            dict_of_expressions={
                'linear': Variable('sm_tt'),
                'log': Variable('sm_tt'),
            },
            controlled_by=self.train_tt_catalog.controlled_by,
        )

        # %%
        # Definition of the utility functions with linear cost.
        v_train = (
            self.asc_train_catalog
            + b_time * self.train_tt_catalog
            + b_cost * Variable('train_cost')
        )
        v_swissmetro = b_time * self.sm_tt_catalog + b_cost * Variable('sm_cost')
        v_car = (
            self.asc_car_catalog
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

    def test_embed_expression(self):
        self.assertTrue(self.asc_car_catalog.embed_expression(Catalog))
        self.assertTrue(self.log_probability.embed_expression(Variable))
        self.assertTrue(self.log_probability.embed_expression(Catalog))
        self.assertTrue(self.log_probability.embed_expression(MultipleExpression))


if __name__ == '__main__':
    unittest.main()
