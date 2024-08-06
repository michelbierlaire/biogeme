"""
Test the nests module

:author: Michel Bierlaire
:date: Thu Oct  5 16:13:40 2023
"""

import unittest
from unittest.mock import MagicMock

from biogeme.nests import (
    OneNestForNestedLogit,
    NestsForNestedLogit,
    OneNestForCrossNestedLogit,
    NestsForCrossNestedLogit,
    get_nest,
    Nests,
)
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Beta, Expression, Numeric, get_dict_values


class TestOneNestForNestedLogit(unittest.TestCase):
    def test_default_nest_name(self):
        nest = OneNestForNestedLogit(Beta('mu', 1, 1, None, 0), [1, 2, 3])

        # Assuming nest_name should be None by default
        self.assertIsNone(nest.name)

    def test_from_tuple(self):
        """Test creation from tuple."""
        nest_param = Numeric(0.5)
        list_of_alternatives = [1, 2, 3]
        tuple_input = (nest_param, list_of_alternatives)
        nest = OneNestForNestedLogit.from_tuple(tuple_input)
        self.assertEqual(repr(nest.nest_param), repr(nest_param))
        self.assertEqual(nest.list_of_alternatives, list_of_alternatives)

    def test_intersection(self):
        """Test intersection method."""
        nest1 = OneNestForNestedLogit(Numeric(0.5), [1, 2, 3])
        nest2 = OneNestForNestedLogit(Numeric(0.5), [2, 3, 4])
        self.assertEqual(nest1.intersection(nest2), {2, 3})


class TestNestsForNestedLogitExtended(unittest.TestCase):
    def test_nest_name_setting(self):
        choice_set = [1, 2, 3, 4, 5, 6]
        nests_tuple = (
            OneNestForNestedLogit(
                nest_param=Beta('mu_1', 1, 1, None, 0), list_of_alternatives=[1, 2]
            ),
            OneNestForNestedLogit(
                nest_param=Beta('mu_2', 1, 1, None, 0),
                list_of_alternatives=[3, 4],
                name="Nest2",
            ),
            OneNestForNestedLogit(
                nest_param=Beta('mu_3', 1, 1, None, 0), list_of_alternatives=[5, 6]
            ),
        )

        nests_for_nested_logit = NestsForNestedLogit(
            choice_set=choice_set, tuple_of_nests=nests_tuple
        )

        # Assuming that nest names are set in the NestsForNestedLogit constructor
        self.assertEqual(nests_for_nested_logit.tuple_of_nests[0].name, "nest_1")
        self.assertEqual(nests_for_nested_logit.tuple_of_nests[1].name, "Nest2")
        self.assertEqual(nests_for_nested_logit.tuple_of_nests[2].name, "nest_3")

    def test_check_union_extra_values(self):
        choice_set = [1, 2, 3, 4]
        nests = (
            OneNestForNestedLogit(Beta('mu_1', 1, 1, None, 0), [1, 2], 'Nest1'),
            OneNestForNestedLogit(Beta('mu_2', 1, 1, None, 0), [3, 4, 5, 6], 'Nest2'),
            # Notice that 5 and 6 are extra here
        )

        with self.assertRaisesRegex(BiogemeError, '5, 6'):
            nests_for_nested_logit = NestsForNestedLogit(
                choice_set=choice_set, tuple_of_nests=nests
            )

    def test_check_intersection_with_overlapping_nests(self):
        choice_set = [1, 2, 3, 4, 5, 6]
        nests = (
            OneNestForNestedLogit(Beta('mu_1', 1, 1, None, 0), [1, 2, 3], "Nest1"),
            OneNestForNestedLogit(Beta('mu_2', 1, 1, None, 0), [3, 4, 5], "Nest2"),
            # Notice that 3 is common in both nests
        )

        nests_for_nested_logit = NestsForNestedLogit(
            choice_set=choice_set, tuple_of_nests=nests
        )

        # Test that check_intersection identifies the intersection
        is_valid, msg = nests_for_nested_logit.check_intersection()
        self.assertFalse(is_valid)
        self.assertIn('3', msg)

    def test_old_initialization(self):
        mu_1 = Beta('mu_1', 1, 1, 10, 0)
        alternatives_1 = [1, 3]
        nest_1 = mu_1, alternatives_1
        the_nest_1 = OneNestForNestedLogit.from_tuple(nest_1)
        self.assertIs(the_nest_1.nest_param, mu_1)
        self.assertListEqual(the_nest_1.list_of_alternatives, alternatives_1)
        mu_2 = Beta('mu_2', 1, 1, 10, 0)
        alternatives_2 = [2]
        nest_2 = mu_2, alternatives_2
        the_nest_2 = OneNestForNestedLogit.from_tuple(nest_2)
        nests = NestsForNestedLogit(
            choice_set=[1, 2, 3], tuple_of_nests=(nest_1, nest_2)
        )
        for nest in nests:
            self.assertIsInstance(nest, OneNestForNestedLogit)


class TestNestsForCrossNestedLogit(unittest.TestCase):

    def setUp(self):
        choice_set = [1, 2, 3, 4]
        dict_of_alpha = {1: Numeric(0.2), 2: Numeric(0.3)}
        nest_param = Numeric(0.5)
        self.nests = (OneNestForCrossNestedLogit(nest_param, dict_of_alpha, "nest1"),)
        self.model = NestsForCrossNestedLogit(choice_set, self.nests)

    def test_get_alpha_dict(self):
        """Test that alpha dict correctly maps alternative IDs to alpha expressions."""
        expected_alpha_dict = {'nest1': Numeric(0.2)}
        alpha_dict = self.model.get_alpha_dict(1)
        self.assertIsInstance(alpha_dict['nest1'], Numeric)
        self.assertEqual(expected_alpha_dict.keys(), alpha_dict.keys())
        self.assertEqual(
            expected_alpha_dict['nest1'].get_value(),
            alpha_dict['nest1'].get_value(),
        )

    def test_one_nest(self):
        alpha_1 = Beta('alpha_1', 0, 0, 1, 0)
        alpha_2 = Beta('alpha_2', 0, 0, 1, 0)
        nest_1 = OneNestForCrossNestedLogit(
            nest_param=Beta('mu', 1, 1, None, 0), dict_of_alpha={1: alpha_1, 2: alpha_2}
        )
        is_fixed = nest_1.all_alpha_fixed()
        self.assertFalse(is_fixed)

        nest_2 = OneNestForCrossNestedLogit(
            nest_param=Beta('mu', 1, 1, None, 0), dict_of_alpha={1: 0.5, 2: 0.5}
        )
        is_fixed = nest_2.all_alpha_fixed()
        self.assertTrue(is_fixed)

    def test_get_alpha_dict_2(self):
        mu_param_1 = Beta('mu_param_1', 1, 1, 10, 0)
        alpha_param_1 = Beta('alpha_param_1', 0.5, 0, 1, 0)
        alpha_dict_1 = {1: alpha_param_1, 3: 1.0}
        the_nest_1 = OneNestForCrossNestedLogit(
            nest_param=mu_param_1, dict_of_alpha=alpha_dict_1, name='nest_1'
        )
        mu_param_2 = Beta('mu_param_2', 1, 1, 10, 0)
        alpha_param_2 = Beta('alpha_param_2', 0.5, 0, 1, 0)
        alpha_dict_2 = {1: alpha_param_2, 2: 1.0}
        the_nest_2 = OneNestForCrossNestedLogit(
            nest_param=mu_param_2, dict_of_alpha=alpha_dict_2, name='nest_2'
        )

        the_nests = NestsForCrossNestedLogit(
            choice_set=[1, 2, 3], tuple_of_nests=(the_nest_1, the_nest_2)
        )
        alpha_dict = the_nests.get_alpha_dict(alternative_id=1)
        expected_dict = {'nest_1': alpha_param_1, 'nest_2': alpha_param_2}
        self.assertDictEqual(alpha_dict, expected_dict)
        alpha_dict = get_dict_values(
            the_dict=the_nests.get_alpha_dict(alternative_id=2)
        )
        expected_dict = {'nest_1': 0, 'nest_2': 1}
        self.assertDictEqual(alpha_dict, expected_dict)

    def test_old_initialization(self):
        mu_param_1 = Beta('mu_param_1', 1, 1, 10, 0)
        alpha_param_1 = Beta('alpha_param_1', 0.5, 0, 1, 0)
        alpha_dict_1 = {1: alpha_param_1, 2: Numeric(0.0), 3: Numeric(1.0)}
        nest_1 = mu_param_1, alpha_dict_1
        the_nest_1 = OneNestForCrossNestedLogit.from_tuple(nest_1)
        self.assertIs(the_nest_1.nest_param, mu_param_1)
        self.assertEqual(
            set(the_nest_1.dict_of_alpha.keys()),
            set(alpha_dict_1.keys()),
            "Dictionaries have different keys",
        )
        for key in alpha_dict_1:
            self.assertEqual(
                repr(the_nest_1.dict_of_alpha[key]), repr(alpha_dict_1[key])
            )
        mu_param_2 = Beta('mu_param_2', 1, 1, 10, 0)
        alpha_param_2 = Beta('alpha_param_2', 0.5, 0, 1, 0)
        alpha_dict_2 = {1: alpha_param_2, 2: 1.0, 3: 0.0}
        nest_2 = mu_param_2, alpha_dict_2
        the_nest_2 = OneNestForCrossNestedLogit.from_tuple(nest_2)

        the_nests = NestsForCrossNestedLogit(
            choice_set=[1, 2, 3], tuple_of_nests=(nest_1, nest_2)
        )
        for nest in the_nests:
            self.assertIsInstance(nest, OneNestForCrossNestedLogit)


class TestOneNestForCrossNestedLogit(unittest.TestCase):
    def setUp(self):
        self.nest_param = Numeric(0.5)
        self.dict_of_alpha = {1: Numeric(0.2), 2: Numeric(0.3)}
        self.nest = OneNestForCrossNestedLogit(self.nest_param, self.dict_of_alpha)

    def test_from_tuple_old_syntax(self):
        """Test creating an instance from the old tuple syntax."""
        nest_param = Numeric(0.5)
        dict_of_alpha = {1: Numeric(0.3), 2: Numeric(0.7)}
        tuple_input = (nest_param, dict_of_alpha)
        nest = OneNestForCrossNestedLogit.from_tuple(tuple_input)
        self.assertEqual(repr(nest.nest_param), repr(nest_param))
        self.assertTrue(
            all(isinstance(alpha, Numeric) for alpha in nest.dict_of_alpha.values())
        )

    def test_all_alpha_fixed_with_mixed_types(self):
        """Test checking if all alphas are fixed when they are mixed types."""
        nest_param = Numeric(0.5)
        dict_of_alpha = {1: 0.3, 2: Numeric(0.7)}
        nest = OneNestForCrossNestedLogit(nest_param, dict_of_alpha)
        self.assertTrue(nest.all_alpha_fixed())

    def test_all_alpha_fixed_with_expressions(self):
        """Test checking if all alphas are fixed when they are expressions."""
        nest_param = Numeric(0.5)
        dict_of_alpha = {1: Numeric(0.3), 2: Numeric(0.7)}
        nest = OneNestForCrossNestedLogit(nest_param, dict_of_alpha)
        self.assertTrue(nest.all_alpha_fixed())
        dict_of_alpha = {1: Numeric(0.3), 2: Beta('alpha', 0, None, None, 0)}
        nest = OneNestForCrossNestedLogit(nest_param, dict_of_alpha)
        self.assertFalse(nest.all_alpha_fixed())
        dict_of_alpha = {1: Numeric(0.3), 2: Numeric(1) + Numeric(2)}
        nest = OneNestForCrossNestedLogit(nest_param, dict_of_alpha)
        self.assertFalse(nest.all_alpha_fixed())

    def test_post_init_list_of_alternatives(self):
        """Test if list of alternatives is correctly initialized after post init."""
        nest_param = 0.5  # Using float for simplicity
        dict_of_alpha = {1: 0.3, 2: 0.7, 3: 0.5}  # Using floats for simplicity
        nest = OneNestForCrossNestedLogit(nest_param, dict_of_alpha)
        self.assertListEqual(nest.list_of_alternatives, list(dict_of_alpha.keys()))

    def test_from_tuple(self):
        """Test creation from the old tuple syntax."""
        tuple_input = (self.nest_param, self.dict_of_alpha)
        nest = OneNestForCrossNestedLogit.from_tuple(tuple_input)
        self.assertEqual(repr(nest.nest_param), repr(self.nest_param))
        self.assertEqual(nest.dict_of_alpha, self.dict_of_alpha)


class TestGetNest(unittest.TestCase):
    def setUp(self):
        self.expression = Numeric(0.5)
        self.list_of_alternatives = [1, 2, 3]

    def test_get_nest_with_OneNestForNestedLogit(self):
        """Test that the function returns the input when it's already an instance of OneNestForNestedLogit."""
        nest = OneNestForNestedLogit(self.expression, self.list_of_alternatives)
        result = get_nest(nest)
        self.assertIs(result, nest)

    def test_get_nest_with_old_tuple(self):
        """Test that the function converts old tuple-based nest definitions into OneNestForNestedLogit instances."""
        old_nest = (self.expression, self.list_of_alternatives)
        result = get_nest(old_nest)
        self.assertIsInstance(result, OneNestForNestedLogit)
        self.assertEqual(repr(result.nest_param), repr(self.expression))
        self.assertEqual(result.list_of_alternatives, self.list_of_alternatives)

    def test_get_nest_with_invalid_type(self):
        """Test that the function raises a TypeError when the input is neither OneNestForNestedLogit nor a tuple."""
        invalid_nest = 'not a valid nest definition'
        with self.assertRaises(TypeError) as context:
            get_nest(invalid_nest)
        self.assertIn('does not represent a nest', str(context.exception))


class TestNests(unittest.TestCase):
    def test_constructor_correct_initialization(self):
        choice_set = [1, 2, 3, 4]
        nest1 = MagicMock(name='Nest1', list_of_alternatives=[1, 2])
        nest2 = MagicMock(name='Nest2', list_of_alternatives=[3, 4])
        nests = Nests(choice_set, (nest1, nest2))
        self.assertEqual(len(nests.mev_alternatives), 4)
        self.assertTrue(nests.check_names())

    def test_constructor_with_unnamed_nests(self):
        choice_set = [1, 2]
        nest = MagicMock(list_of_alternatives=[1, 2])
        nest.name = None
        nests = Nests(choice_set, (nest,))
        self.assertEqual(nest.name, 'nest_1')

    def test_constructor_with_invalid_alternatives(self):
        choice_set = [1, 2]
        nest = MagicMock(name='Nest', list_of_alternatives=[3])
        with self.assertRaises(BiogemeError) as cm:
            Nests(choice_set, (nest,))
        self.assertIn('not in the choice set', str(cm.exception))

    def test_getitem_valid_index(self):
        nest = MagicMock(name='Nest', list_of_alternatives=[1, 2])
        nests = Nests([1, 2], (nest,))
        self.assertEqual(nests[0], nest)

    def test_getitem_index_out_of_bounds(self):
        nest = MagicMock(name='Nest', list_of_alternatives=[1, 2])
        nests = Nests([1, 2], (nest,))
        with self.assertRaises(IndexError):
            _ = nests[1]

    def test_iteration(self):
        nest1 = MagicMock(name='Nest1', list_of_alternatives=[1])
        nest2 = MagicMock(name='Nest2', list_of_alternatives=[2])
        nests = Nests([1, 2], (nest1, nest2))
        for i, nest in enumerate(nests):
            self.assertEqual(nest, [nest1, nest2][i])

    def test_check_union(self):
        choice_set = [1, 2, 3, 4, 5]
        nest1 = MagicMock(name='Nest1', list_of_alternatives=[1, 2])
        nest2 = MagicMock(name='Nest2', list_of_alternatives=[3])
        nests = Nests(choice_set, (nest1, nest2))
        nests.alone = set()
        result, message = nests.check_union()
        self.assertFalse(result)
        self.assertIn('not in any nest', message)


if __name__ == '__main__':
    unittest.main()
