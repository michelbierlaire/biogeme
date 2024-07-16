"""
Test the "deprecated" decorators"

:author: Michel Bierlaire
:date: Mon May 13 09:35:05 2024

"""

import warnings

from biogeme.deprecated import deprecated, deprecated_parameters

import unittest
from unittest.mock import patch


class TestValidateParamsDecorator(unittest.TestCase):
    def test_valid_parameters(self):
        @deprecated_parameters(obsolete_params={})
        def func(a, b):
            return a + b

        self.assertEqual(func(a=1, b=2), 3)

    def test_valid_parameters_default(self):
        @deprecated_parameters(obsolete_params={})
        def func(a, b=2):
            return a + b

        self.assertEqual(func(a=1), 3)

    def test_deprecated_parameters(self):
        @deprecated_parameters(obsolete_params={'old_name': 'new_name'})
        def func(new_name):
            return new_name * 2

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = func(old_name=5)
            self.assertEqual(result, 10)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))

    def test_deprecated_parameters_mixed(self):
        @deprecated_parameters(
            obsolete_params={'old_name_a': 'new_name_a', 'old_name_b': 'new_name_b'}
        )
        def func(new_name_a, new_name_b):
            return new_name_a * new_name_b

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = func(old_name_a=5, new_name_b=2)
            self.assertEqual(result, 10)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = func(old_name_a=5, old_name_b=2)
            self.assertEqual(result, 10)
            self.assertEqual(len(w), 2)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))

    def test_deprecated_parameters_None(self):
        @deprecated_parameters(obsolete_params={'old_name': None})
        def func(x):
            return 2 * x

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = func(x=3, old_name=5)
            self.assertEqual(result, 6)

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))

        with self.assertRaises(TypeError):
            result = func(old_name=5)


if __name__ == '__main__':
    unittest.main()
