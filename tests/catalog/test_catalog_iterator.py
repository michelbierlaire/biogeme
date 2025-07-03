import unittest

import pandas as pd

from biogeme.catalog import (
    Catalog,
    CentralController,
    Configuration,
    Controller,
    SelectedConfigurationsIterator,
    segmentation_catalogs,
)
from biogeme.database import Database
from biogeme.expressions import (
    Beta,
    NamedExpression,
    Numeric,
    Variable,
    list_of_free_betas_in_expression,
    log,
)
from biogeme.models import loglogit


class TestSelectedExpressionsIterator(unittest.TestCase):

    def setUp(self):
        # Mock configurations

        expr_1 = Numeric(1)
        expr_2 = Numeric(2)
        expr_3 = Numeric(3)
        named_expr_1 = NamedExpression(name='expr_1', expression=expr_1)
        named_expr_2 = NamedExpression(name='expr_2', expression=expr_2)
        named_expr_3 = NamedExpression(name='expr_3', expression=expr_3)
        controller = Controller(
            controller_name='test', specification_names=['expr_1', 'expr_2', 'expr_3']
        )
        catalog = Catalog(
            catalog_name='test',
            named_expressions=[named_expr_1, named_expr_2, named_expr_3],
            controlled_by=controller,
        )
        self.central_controller = CentralController(expression=catalog)
        self.configs = self.central_controller.all_configurations

    def test_iterator_initialization_with_all_configurations(self):
        iterator = SelectedConfigurationsIterator(self.central_controller)
        self.assertEqual(iterator.current_configuration in self.configs, True)

    def test_iterator_initialization_with_selected_configurations(self):
        for selected in self.configs:
            iterator = SelectedConfigurationsIterator(
                self.central_controller, {selected}
            )
            current_configuration = next(iterator)
            self.assertEqual(current_configuration, selected)

    def test_iteration_order(self):
        iterator = SelectedConfigurationsIterator(self.central_controller)
        seen = []
        for config in iterator:
            seen.append(config)
        self.assertEqual(set(seen), set(self.configs))
        self.assertEqual(len(seen), len(self.configs))

    def test_iteration_stops(self):
        iterator = SelectedConfigurationsIterator(self.central_controller)
        for _ in range(len(self.configs)):
            next(iterator)
        with self.assertRaises(StopIteration):
            next(iterator)

    def test_first_iteration_flag(self):
        iterator = SelectedConfigurationsIterator(
            self.central_controller, set(self.configs)
        )
        self.assertTrue(iterator.first)
        _ = next(iterator)
        self.assertFalse(iterator.first)

    def test_number_counter(self):
        iterator = SelectedConfigurationsIterator(
            self.central_controller, set(self.configs)
        )
        for i in range(len(self.configs)):
            next(iterator)
            self.assertEqual(iterator.number, i + 1)

    def test_value(self):
        iterator = SelectedConfigurationsIterator(
            self.central_controller, set(self.configs)
        )
        expected_values = {'test:expr_2': 2.0, 'test:expr_3': 3.0, 'test:expr_1': 1.0}

        for config in iterator:
            config_id = config.get_string_id()
            the_configuration = Configuration.from_string(config_id)
            self.central_controller.set_configuration(the_configuration)
            value = self.central_controller.expression.get_value()
            self.assertEqual(value, expected_values[config_id])


class TestIterator(unittest.TestCase):
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
            dict_of_expressions={'linear': Variable('sm_tt'), 'log': Variable('sm_tt')},
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

    def test_iterator(self):
        for named_expression in self.the_iterator:
            the_betas = list_of_free_betas_in_expression(
                the_expression=named_expression.expression
            )


if __name__ == '__main__':
    unittest.main()
