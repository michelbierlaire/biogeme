"""
Test the "deprecated" decorators"

:author: Michel Bierlaire
:date: Mon May 13 09:35:05 2024

"""

import logging
import unittest

from biogeme.deprecated import deprecated_parameters

logger = logging.getLogger("biogeme.deprecated")


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

        with self.assertLogs(logger, level="WARNING") as cm:
            result = func(old_name=5)

            self.assertEqual(result, 10)
            # exactly one WARNING; message contains both names
            self.assertEqual(len(cm.records), 1)
            self.assertEqual(cm.records[0].levelno, logging.WARNING)
            self.assertIn("old_name", cm.output[0])
            self.assertIn("new_name", cm.output[0])

    def test_deprecated_parameters_mixed(self):
        @deprecated_parameters(
            obsolete_params={'old_name_a': 'new_name_a', 'old_name_b': 'new_name_b'}
        )
        def func(new_name_a, new_name_b):
            return new_name_a * new_name_b

        with self.assertLogs(logger, level="WARNING") as cm:
            result = func(old_name_a=5, new_name_b=2)
            self.assertEqual(result, 10)
            # exactly one WARNING; message contains both names
            self.assertEqual(len(cm.records), 1)
            self.assertEqual(cm.records[0].levelno, logging.WARNING)
            self.assertIn("old_name_a", cm.output[0])

        with self.assertLogs(logger, level="WARNING") as cm:
            result = func(old_name_a=5, old_name_b=2)
            self.assertEqual(result, 10)
            # exactly one WARNING; message contains both names
            self.assertEqual(len(cm.records), 2)
            self.assertEqual(cm.records[0].levelno, logging.WARNING)
            self.assertIn("old_name_a", cm.output[0])
            self.assertIn("old_name_b", cm.output[1])

    def test_deprecated_parameters_None(self):
        @deprecated_parameters(obsolete_params={'old_name': None})
        def func(x):
            return 2 * x

        with self.assertLogs(logger, level="WARNING") as cm:
            result = func(x=3, old_name=5)

        self.assertEqual(result, 6)
        # exactly one WARNING; message contains both names
        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].levelno, logging.WARNING)
        self.assertIn("old_name", cm.output[0])

        with self.assertRaises(TypeError):
            result = func(old_name=5)


if __name__ == '__main__':
    unittest.main()
