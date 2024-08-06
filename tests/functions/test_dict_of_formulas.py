"""
Test the dict_of_formulas functions

:author: Michel Bierlaire
:date: Mon May 13 16:27:54 2024

"""

import unittest
from unittest.mock import patch, MagicMock

from biogeme.exceptions import BiogemeError
from biogeme.expressions import Expression
from biogeme.dict_of_formulas import (
    check_validity,
    get_expression,
)


class TestExpressions(unittest.TestCase):

    def setUp(self):
        self.expression_mock = MagicMock(spec=Expression)
        self.dict_of_valid_loglike = {'log_like': self.expression_mock}
        self.dict_of_valid_weight = {'weight': self.expression_mock}

    def test_get_expression_valid(self):
        result = get_expression(
            dict_of_formulas=self.dict_of_valid_loglike, valid_keywords=['log_like']
        )
        self.assertEqual(result, self.expression_mock)

    def test_check_validity_valid(self):
        try:
            check_validity(self.dict_of_valid_loglike)
        except BiogemeError:
            self.fail("check_validity() raised BiogemeError unexpectedly!")

    def test_check_validity_invalid_type(self):
        with self.assertRaises(BiogemeError):
            check_validity({'log_like': 'not_an_expression'})

    @patch('biogeme.dict_of_formulas.logger')
    def test_get_expression_warning_similar_name(self, mock_logger):
        with patch('biogeme.dict_of_formulas.is_similar_to', return_value=True):
            get_expression(
                dict_of_formulas={'loc_like': self.expression_mock},
                valid_keywords=['log_like', 'loglike'],
            )
            self.assertEqual(mock_logger.warning.call_count, 2)

    @patch('biogeme.dict_of_formulas.logger')
    def test_get_expression_no_valid(self, mock_logger):
        result = get_expression(dict_of_formulas={}, valid_keywords=['weight'])
        self.assertIsNone(result)
        mock_logger.warning.assert_not_called()

    def test_get_expression_multiple_valid_names(self):
        dict_of_formulas = {
            'log_like': self.expression_mock,
            'loglike': self.expression_mock,
        }
        with self.assertRaises(BiogemeError):
            get_expression(
                dict_of_formulas=dict_of_formulas,
                valid_keywords=['log_like', 'loglike'],
            )


if __name__ == '__main__':
    unittest.main()
