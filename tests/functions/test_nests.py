"""
Test the nests module

:author: Michel Bierlaire
:date: Thu Oct  5 16:13:40 2023
"""
import unittest
from biogeme.nests import (
    OneNestForNestedLogit,
    NestsForNestedLogit,
    OneNestForCrossNestedLogit,
    NestsForCrossNestedLogit,
)
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Beta, Expression


class TestOneNestForNestedLogit(unittest.TestCase):
    def test_default_nest_name(self):
        nest = OneNestForNestedLogit(Beta('mu', 1, 1, None, 0), [1, 2, 3])

        # Assuming nest_name should be None by default
        self.assertIsNone(nest.name)


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
    def test_one_nest(self):
        alpha_1 = Beta('alpha_1', 0, 0, 1, 0)
        alpha_2 = Beta('alpha_2', 0, 0, 1, 0)
        nest_1 = OneNestForCrossNestedLogit(
            nest_param=Beta('mu', 1, 1, None, 0), dict_of_alpha={1: alpha_1, 2: alpha_2}
        )
        is_fixed = nest_1.is_alpha_fixed()
        self.assertFalse(is_fixed)

        nest_2 = OneNestForCrossNestedLogit(
            nest_param=Beta('mu', 1, 1, None, 0), dict_of_alpha={1: 0.5, 2: 0.5}
        )
        is_fixed = nest_2.is_alpha_fixed()
        self.assertTrue(is_fixed)

    def test_get_alpha_dict(self):
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
        alpha_dict = the_nests.get_alpha_values(alternative_id=2)
        expected_dict = {'nest_1': 0, 'nest_2': 1}
        self.assertDictEqual(alpha_dict, expected_dict)

    def test_old_initialization(self):
        mu_param_1 = Beta('mu_param_1', 1, 1, 10, 0)
        alpha_param_1 = Beta('alpha_param_1', 0.5, 0, 1, 0)
        alpha_dict_1 = {1: alpha_param_1, 2: 0.0, 3: 1.0}
        nest_1 = mu_param_1, alpha_dict_1
        the_nest_1 = OneNestForCrossNestedLogit.from_tuple(nest_1)
        self.assertIs(the_nest_1.nest_param, mu_param_1)
        self.assertDictEqual(the_nest_1.dict_of_alpha, alpha_dict_1)
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


if __name__ == '__main__':
    unittest.main()
