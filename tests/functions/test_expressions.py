"""
Test the expressions module

:author: Michel Bierlaire
:data: Wed Apr 29 17:47:53 2020

"""

import unittest

import numpy as np
import pandas as pd

import biogeme.exceptions as excep
import biogeme.expressions as ex
from biogeme import models
from biogeme.expressions import IdManager, TypeOfElementaryExpression, LinearTermTuple
from biogeme.function_output import (
    BiogemeDisaggregateFunctionOutput,
    BiogemeFunctionOutput,
)
from biogeme.expressions.numeric_tools import validate, MAX_VALUE
from test_data import getData


class test_expressions(unittest.TestCase):
    def setUp(self):
        self.myData = getData(2)
        self.Person = ex.Variable('Person')
        self.Variable1 = ex.Variable('Variable1')
        self.Variable2 = ex.Variable('Variable2')
        self.Choice = ex.Variable('Choice')
        self.Av1 = ex.Variable('Av1')
        self.Av2 = ex.Variable('Av2')
        self.Av3 = ex.Variable('Av3')
        self.beta1 = ex.Beta('beta1', 0.2, None, None, 0)
        self.beta2 = ex.Beta('beta2', 0.4, None, None, 0)
        self.beta3 = ex.Beta('beta3', 1, None, None, 1)
        self.beta4 = ex.Beta('beta4', 0, None, None, 1)
        self.omega1 = ex.RandomVariable('omega1')
        self.omega2 = ex.RandomVariable('omega2')
        self.xi1 = ex.bioDraws('xi1', 'NORMAL')
        self.xi2 = ex.bioDraws('xi2', 'UNIF')
        self.xi3 = ex.bioDraws('xi3', 'WRONGTYPE')
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataframeEqual)

    def test_errors(self):
        with self.assertRaises(excep.BiogemeError):
            _ = ex.Numeric(1) / 'ert'

    #        with self.assertRaises(exceptions.BiogemeError):
    #            _ = 'ert' / Numeric(1)

    def assertDataframeEqual(self, a, b, msg):
        try:
            pd.testing.assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def test_is_numeric(self):
        result = ex.is_numeric(1)
        self.assertTrue(result)
        result = ex.is_numeric(0.1)
        self.assertTrue(result)
        result = ex.is_numeric(True)
        self.assertTrue(result)
        result = ex.is_numeric(self)
        self.assertFalse(result)

    def test_add(self):
        result = self.Variable1 + self.Variable2
        self.assertEqual(str(result), '(Variable1 + Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 + 1
        self.assertEqual(str(result), '(Variable1 + `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 + self.Variable1
        self.assertEqual(str(result), '(`1.0` + Variable1)')
        self.assertTrue(result.children[1] is self.Variable1)

        with self.assertRaises(excep.BiogemeError):
            result = self + self.Variable1

        with self.assertRaises(excep.BiogemeError):
            result = self.Variable1 + self

    def test_sub(self):
        result = self.Variable1 - self.Variable2
        self.assertEqual(str(result), '(Variable1 - Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 - 1
        self.assertEqual(str(result), '(Variable1 - `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 - self.Variable1
        self.assertEqual(str(result), '(`1.0` - Variable1)')
        self.assertTrue(result.children[1] is self.Variable1)

        with self.assertRaises(excep.BiogemeError):
            result = self - self.Variable1

        with self.assertRaises(excep.BiogemeError):
            result = self.Variable1 - self

    def test_mul(self):
        result = self.Variable1 * self.Variable2
        self.assertEqual(str(result), '(Variable1 * Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 * 1
        self.assertEqual(str(result), '(Variable1 * `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 * self.Variable1
        self.assertEqual(str(result), '(`1.0` * Variable1)')
        self.assertTrue(result.children[1] is self.Variable1)

        with self.assertRaises(excep.BiogemeError):
            result = self * self.Variable1

        with self.assertRaises(excep.BiogemeError):
            result = self.Variable1 * self

    def test_div(self):
        result = self.Variable1 / self.Variable2
        self.assertEqual(str(result), '(Variable1 / Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 / 1
        self.assertEqual(str(result), '(Variable1 / `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 / self.Variable1
        self.assertEqual(str(result), '(`1.0` / Variable1)')
        self.assertTrue(result.children[1] is self.Variable1)

        with self.assertRaises(excep.BiogemeError):
            result = self / self.Variable1

        with self.assertRaises(excep.BiogemeError):
            result = self.Variable1 / self

    def test_neg(self):
        result = -self.Variable1
        self.assertEqual(str(result), '(-Variable1)')
        self.assertTrue(result.children[0] is self.Variable1)

    def test_pow(self):
        result = self.Variable1**self.Variable2
        self.assertEqual(str(result), '(Variable1 ** Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1**1
        self.assertEqual(str(result), '(Variable1 ** `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1**self.Variable1
        self.assertEqual(str(result), '(`1.0` ** Variable1)')
        self.assertTrue(result.children[1] is self.Variable1)

        with self.assertRaises(excep.BiogemeError):
            result = self**self.Variable1

        with self.assertRaises(excep.BiogemeError):
            result = self.Variable1**self

    def test_and(self):
        result = self.Variable1 & self.Variable2
        self.assertEqual(str(result), '(Variable1 and Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 & 1
        self.assertEqual(str(result), '(Variable1 and `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 & self.Variable1
        self.assertEqual(str(result), '(`1.0` and Variable1)')
        self.assertTrue(result.children[1] is self.Variable1)

        with self.assertRaises(excep.BiogemeError):
            result = self & self.Variable1

        with self.assertRaises(excep.BiogemeError):
            result = self.Variable1 & self

    def test_or(self):
        result = self.Variable1 | self.Variable2
        self.assertEqual(str(result), '(Variable1 or Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 | 1
        self.assertEqual(str(result), '(Variable1 or `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 | self.Variable1
        self.assertEqual(str(result), '(`1.0` or Variable1)')
        self.assertTrue(result.children[1] is self.Variable1)

        with self.assertRaises(excep.BiogemeError):
            result = self | self.Variable1

        with self.assertRaises(excep.BiogemeError):
            result = self.Variable1 | self

    def test_eq(self):
        result = self.Variable1 == self.Variable2
        self.assertEqual(str(result), '(Variable1 == Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 == 1
        self.assertEqual(str(result), '(Variable1 == `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 == self.Variable1
        self.assertEqual(str(result), '(Variable1 == `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        with self.assertRaises(excep.BiogemeError):
            result = self == self.Variable1

        with self.assertRaises(excep.BiogemeError):
            result = self.Variable1 == self

    def test_neq(self):
        result = self.Variable1 != self.Variable2
        self.assertEqual(str(result), '(Variable1 != Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 != 1
        self.assertEqual(str(result), '(Variable1 != `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 != self.Variable1
        self.assertEqual(str(result), '(Variable1 != `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        with self.assertRaises(excep.BiogemeError):
            result = self != self.Variable1

        with self.assertRaises(excep.BiogemeError):
            result = self.Variable1 != self

    def test_le(self):
        result = self.Variable1 <= self.Variable2
        self.assertEqual(str(result), '(Variable1 <= Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 <= 1
        self.assertEqual(str(result), '(Variable1 <= `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 <= self.Variable1
        self.assertEqual(str(result), '(Variable1 >= `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        with self.assertRaises(excep.BiogemeError):
            result = self <= self.Variable1

        with self.assertRaises(excep.BiogemeError):
            result = self.Variable1 <= self

    def test_ge(self):
        result = self.Variable1 >= self.Variable2
        self.assertEqual(str(result), '(Variable1 >= Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 >= 1
        self.assertEqual(str(result), '(Variable1 >= `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 >= self.Variable1
        self.assertEqual(str(result), '(Variable1 <= `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        with self.assertRaises(excep.BiogemeError):
            result = self >= self.Variable1

        with self.assertRaises(excep.BiogemeError):
            result = self.Variable1 >= self

    def test_lt(self):
        result = self.Variable1 < self.Variable2
        self.assertEqual(str(result), '(Variable1 < Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 < 1
        self.assertEqual(str(result), '(Variable1 < `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 < self.Variable1
        self.assertEqual(str(result), '(Variable1 > `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        with self.assertRaises(excep.BiogemeError):
            result = self < self.Variable1

        with self.assertRaises(excep.BiogemeError):
            result = self.Variable1 < self

    def test_gt(self):
        result = self.Variable1 > self.Variable2
        self.assertEqual(str(result), '(Variable1 > Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 > 1
        self.assertEqual(str(result), '(Variable1 > `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 > self.Variable1
        self.assertEqual(str(result), '(Variable1 < `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        with self.assertRaises(excep.BiogemeError):
            result = self > self.Variable1

        with self.assertRaises(excep.BiogemeError):
            result = self.Variable1 > self

    def test_get_value_c(self):
        result = self.Variable1.get_value_c(database=self.myData, prepare_ids=True)
        np.testing.assert_equal(result, [10, 20, 30, 40, 50])

    def test_DefineVariable(self):
        _ = self.myData.define_variable('newvar_b', self.Variable1 + self.Variable2)
        cols = self.myData.data.columns
        added = 'newvar_b' in cols
        self.assertTrue(added)
        self.myData.data['newvar_p'] = (
            self.myData.data['Variable1'] + self.myData.data['Variable2']
        )
        pd.testing.assert_series_equal(
            self.myData.data['newvar_b'],
            self.myData.data['newvar_p'],
            check_dtype=False,
            check_names=False,
        )

    def test_expr1(self):
        self.beta1.status = 1
        self.beta2.status = 1
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        self.assertAlmostEqual(expr1.get_value(), -1.275800115089098, 5)
        res = expr1.get_value_c(prepare_ids=True)
        self.assertAlmostEqual(res, -1.275800115089098, 5)

    def test_expr1_newvalues(self):
        self.beta1.status = 1
        self.beta2.status = 1
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        newvalues = {'beta1': 1, 'beta2': 2, 'beta3': 3, 'beta4': 2}
        expr1.change_init_values(newvalues)
        self.assertAlmostEqual(expr1.get_value(), 1.9323323583816936, 5)
        res = expr1.get_value_c(prepare_ids=True)
        self.assertAlmostEqual(res, 1.9323323583816936, 5)

    def test_expr1_derivatives(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        newvalues = {'beta1': 1, 'beta2': 2, 'beta3': 3, 'beta4': 2}
        expr1.change_init_values(newvalues)
        the_function_output: BiogemeFunctionOutput = expr1.get_value_and_derivatives(
            prepare_ids=True
        )
        self.assertAlmostEqual(the_function_output.function, 1.9323323583816936, 5)
        g_ok = [2.0, 0.10150146242745953]
        h0_ok = [0.0, 0.0]
        h1_ok = [0.0, -0.16916910404576588]
        bhhh0_ok = [4, 0.20300292]
        bhhh1_ok = [0.20300292, 0.01030255]
        for check_left, check_right in zip(the_function_output.gradient, g_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.hessian[0], h0_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.hessian[1], h1_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.bhhh[0], bhhh0_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.bhhh[1], bhhh1_ok):
            self.assertAlmostEqual(check_left, check_right, 5)

    def test_expr1_gradient(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        the_function_output: BiogemeFunctionOutput = expr1.get_value_and_derivatives(
            gradient=True, hessian=False, bhhh=False, prepare_ids=True
        )
        self.assertAlmostEqual(the_function_output.function, -1.275800115089098, 5)
        g_ok = [2.0, 5.865300402811844]
        for check_left, check_right in zip(the_function_output.gradient, g_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        self.assertIsNone(the_function_output.hessian)
        self.assertIsNone(the_function_output.bhhh)

    def test_expr1_function(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        the_function = expr1.create_function()
        the_function_output: BiogemeFunctionOutput = the_function([1, 2])
        self.assertAlmostEqual(the_function_output.function, 1.9323323583816936, 5)
        g_ok = [2.0, 0.10150146242745953]
        h0_ok = [0.0, 0.0]
        h1_ok = [0.0, -0.16916910404576588]
        for check_left, check_right in zip(the_function_output.gradient, g_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.hessian[0], h0_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.hessian[1], h1_ok):
            self.assertAlmostEqual(check_left, check_right, 5)

        the_function_output: BiogemeFunctionOutput = the_function([10, -2])
        self.assertAlmostEqual(the_function_output.function, 23.694528049465326, 5)
        g_ok = [2, -1.84726402]
        h0_ok = [0.0, 0.0]
        h1_ok = [0.0, 1.84726402]
        for check_left, check_right in zip(the_function_output.gradient, g_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.hessian[0], h0_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.hessian[1], h1_ok):
            self.assertAlmostEqual(check_left, check_right, 5)

    def test_expr1_database(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        the_function_output: BiogemeDisaggregateFunctionOutput = (
            expr1.get_value_and_derivatives(
                database=self.myData, aggregation=False, prepare_ids=True
            )
        )
        f_ok: float = -1.27580012
        g_ok = [2.0, 5.8653004]
        h0_ok = [0.0, 0.0]
        h1_ok = [0.0, -31.00230213]
        bhhh0_ok = [4.0, 11.73060081]
        bhhh1_ok = [11.73060081, 34.40174882]
        for i, f in enumerate(the_function_output.functions):
            self.assertAlmostEqual(f, f_ok, 5)
            for check_left, check_right in zip(the_function_output.gradients[i], g_ok):
                self.assertAlmostEqual(check_left, check_right, 5)
            for check_left, check_right in zip(
                the_function_output.hessians[i][0], h0_ok
            ):
                self.assertAlmostEqual(check_left, check_right, 5)
            for check_left, check_right in zip(
                the_function_output.hessians[i][1], h1_ok
            ):
                self.assertAlmostEqual(check_left, check_right, 5)
            for check_left, check_right in zip(
                the_function_output.bhhhs[i][0], bhhh0_ok
            ):
                self.assertAlmostEqual(check_left, check_right, 5)
            for check_left, check_right in zip(
                the_function_output.bhhhs[i][1], bhhh1_ok
            ):
                self.assertAlmostEqual(check_left, check_right, 5)

    def test_expr1_database_agg(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        the_function_output: BiogemeFunctionOutput = expr1.get_value_and_derivatives(
            database=self.myData, aggregation=True, prepare_ids=True
        )
        f_ok = -6.37900057544549
        g_ok = [10.0, 29.32650201405922]
        h0_ok = [0.0, 0.0]
        h1_ok = [0.0, -155.01151064574157]
        b0_ok = [20.0, 58.65300403]
        b1_ok = [58.65300403, 172.00874408]
        self.assertAlmostEqual(the_function_output.function, f_ok, 5)
        for check_left, check_right in zip(the_function_output.gradient, g_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.hessian[0], h0_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.hessian[1], h1_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.bhhh[0], b0_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(the_function_output.bhhh[1], b1_ok):
            self.assertAlmostEqual(check_left, check_right, 5)

    def test_setOfBetas(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta3 * (self.beta2 >= self.beta1)
        )
        s = expr1.set_of_elementary_expression(
            the_type=TypeOfElementaryExpression.FREE_BETA
        )
        self.assertSetEqual(s, {'beta1', 'beta2'})
        s = expr1.set_of_elementary_expression(
            the_type=TypeOfElementaryExpression.FIXED_BETA
        )
        self.assertSetEqual(s, {'beta3'})
        s = expr1.set_of_elementary_expression(the_type=TypeOfElementaryExpression.BETA)
        self.assertSetEqual(s, {'beta1', 'beta2', 'beta3'})

    def test_setOfVariables(self):
        expr1 = 2 * self.Variable1 - ex.exp(-self.Variable2) / (
            self.Variable1 * (self.Variable1 >= self.Variable2)
        )
        s = expr1.set_of_elementary_expression(
            the_type=TypeOfElementaryExpression.VARIABLE
        )
        self.assertSetEqual(s, {'Variable1', 'Variable2'})

    def test_getElementaryExpression(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta3 * (self.beta2 >= self.beta1)
        )
        ell = expr1.get_elementary_expression('beta2')
        self.assertEqual(ell.name, 'beta2')

    def test_expr2(self):
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(
            -self.beta2 * self.Variable2
        ) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        with self.assertRaises(excep.BiogemeError):
            expr2.get_value()
        with self.assertRaises(excep.BiogemeError):
            expr2.get_value_c()
        res = list(expr2.get_value_c(database=self.myData, prepare_ids=True))
        self.assertListEqual(res, [4.0, 8.0, 12.0, 16.0, 20.0])

    def test_dictOfBetas(self):
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(
            -self.beta2 * self.Variable2
        ) / (self.beta3 * (self.beta2 >= self.beta1))
        b = expr2.dict_of_elementary_expression(
            the_type=TypeOfElementaryExpression.BETA
        )
        # Note that the following checks only the labels. Its probably
        # good enough for our purpose.
        self.assertDictEqual(b, {'beta1': 0, 'beta2': self.beta2, 'beta3': self.beta3})

    def test_dictOfVariables(self):
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(
            -self.beta2 * self.Variable2
        ) / (self.beta3 * (self.beta2 >= self.beta1))
        b = expr2.dict_of_elementary_expression(
            the_type=TypeOfElementaryExpression.VARIABLE
        )
        # Note that the following checks only the labels. Its probably
        # good enough for our purpose.
        self.assertDictEqual(
            b, {'Variable1': self.Variable1, 'Variable2': self.Variable2}
        )

    def test_dictOfRandomVariables(self):
        expr = -(self.omega1 + self.omega2 + self.Variable1)
        b = expr.dict_of_elementary_expression(
            the_type=TypeOfElementaryExpression.RANDOM_VARIABLE
        )
        self.assertDictEqual(b, {'omega1': self.omega1, 'omega2': self.omega2})

    def test_dictOfDraws(self):
        expr = -(self.xi1 + self.xi2 - self.xi3 + self.Variable1)
        the_types = expr.dict_of_draw_types()
        self.assertDictEqual(
            the_types, {'xi1': 'NORMAL', 'xi2': 'UNIF', 'xi3': 'WRONGTYPE'}
        )

    def test_getClassName(self):
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(
            -self.beta2 * self.Variable2
        ) / (self.beta3 * (self.beta2 >= self.beta1))
        c = expr2.get_class_name()
        self.assertEqual(c, 'Minus')

    def test_getSignature(self):
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(
            -self.beta2 * self.Variable2
        ) / (self.beta3 * (self.beta2 >= self.beta1))
        expr2.set_id_manager(IdManager([expr2], self.myData, 0))
        s = expr2.get_signature()
        self.assertEqual(len(s), 17)

    def test_embedExpression(self):
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(
            -self.beta2 * self.Variable2
        ) / (self.beta3 * (self.beta2 >= self.beta1))
        self.assertTrue(expr2.embed_expression('Minus'))
        self.assertFalse(expr2.embed_expression('bioDraws'))

    def test_panel_variables(self):
        expr_ok = ex.PanelLikelihoodTrajectory(self.Variable1)
        check_ok = expr_ok.check_panel_trajectory()
        self.assertEqual(len(check_ok), 0)
        expr_not_ok = self.Variable2 + ex.PanelLikelihoodTrajectory(self.Variable1)
        check_not_ok = expr_not_ok.check_panel_trajectory()
        self.assertSetEqual(check_not_ok, {'Variable2'})

    def test_check_draws(self):
        d1 = ex.bioDraws('d1', 'UNIFORM')
        d2 = ex.bioDraws('d2', 'UNIFORM')
        expr_ok = ex.MonteCarlo(d1 * d2)
        check_ok = expr_ok.check_draws()
        self.assertEqual(len(check_ok), 0)
        expr_not_ok = d1 + ex.MonteCarlo(d1 * d2)
        check_not_ok = expr_not_ok.check_draws()
        self.assertSetEqual(check_not_ok, {'d1'})

    def test_check_rv(self):
        d1 = ex.RandomVariable('d1')
        d2 = ex.RandomVariable('d2')
        expr_ok = ex.Integrate(1, d1)
        check_ok = expr_ok.check_rv()
        self.assertEqual(len(check_ok), 0)
        expr_not_ok = d1 + ex.Integrate(1, d2)
        check_not_ok = expr_not_ok.check_rv()
        self.assertSetEqual(check_not_ok, {'d1'})

    def test_countPanelTrajectoryExpressions(self):
        expr1 = self.beta1
        c1 = expr1.count_panel_trajectory_expressions()
        self.assertEqual(c1, 0)
        expr2 = ex.PanelLikelihoodTrajectory(self.beta1)
        c2 = expr2.count_panel_trajectory_expressions()
        self.assertEqual(c2, 1)
        expr3 = ex.PanelLikelihoodTrajectory(ex.PanelLikelihoodTrajectory(self.beta1))
        c3 = expr3.count_panel_trajectory_expressions()
        expr4 = self.Variable1 + ex.PanelLikelihoodTrajectory(
            ex.PanelLikelihoodTrajectory(self.beta1)
        )
        c4 = expr4.count_panel_trajectory_expressions()
        self.assertEqual(c4, 2)

    def test_ids_multiple_formulas(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(
            -self.beta2 * self.Variable2
        ) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        collection_of_formulas = [expr1, expr2]
        formulas = IdManager(
            expressions=collection_of_formulas, database=self.myData, number_of_draws=0
        )
        elementary_expression_index_ok = {
            'beta1': 0,
            'beta2': 1,
            'beta3': 2,
            'beta4': 3,
            'Person': 4,
            'Exclude': 5,
            'Variable1': 6,
            'Variable2': 7,
            'Choice': 8,
            'Av1': 9,
            'Av2': 10,
            'Av3': 11,
        }
        self.assertDictEqual(
            formulas.elementary_expressions.indices,
            elementary_expression_index_ok,
        )
        self.assertListEqual(
            list(formulas.free_betas.expressions.keys()), ['beta1', 'beta2']
        )
        self.assertListEqual(formulas.free_betas.names, ['beta1', 'beta2'])
        self.assertListEqual(
            list(formulas.fixed_betas.expressions.keys()), ['beta3', 'beta4']
        )
        self.assertListEqual(formulas.fixed_betas.names, ['beta3', 'beta4'])
        self.assertFalse(formulas.random_variables.expressions)
        self.assertFalse(formulas.draws.expressions)

    def test_expr3(self):
        myDraws = ex.bioDraws('myDraws', 'UNIFORM')
        expr3 = ex.MonteCarlo(myDraws * myDraws)
        with self.assertRaises(excep.BiogemeError):
            res = expr3.get_value_c(number_of_draws=100000)
        res = expr3.get_value_c(
            database=self.myData, number_of_draws=100000, prepare_ids=True
        )
        for v in res:
            self.assertAlmostEqual(v, 1.0 / 3.0, 2)

    def test_expr4(self):
        omega = ex.RandomVariable('omega')
        a = 0
        b = 1
        x = a + (b - a) / (1 + ex.exp(-omega))
        dx = (b - a) * ex.exp(-omega) * (1 + ex.exp(-omega)) ** (-2)
        integrand = x * x
        expr4 = ex.Integrate(integrand * dx / (b - a), 'omega')
        res = expr4.get_value_c(database=self.myData, prepare_ids=True)
        for v in res:
            self.assertAlmostEqual(v, 1.0 / 3.0, 2)

    def test_expr5(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta3 * (self.beta2 >= self.beta1)
        )
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(
            -self.beta2 * self.Variable2
        ) / (self.beta3 * (self.beta2 >= self.beta1))
        expr5 = ex.Elem({1: expr1, 2: expr2}, self.Person) / 10
        res = list(expr5.get_value_c(database=self.myData, prepare_ids=True))
        res_ok = [
            -0.02703200460356393,
            -0.02703200460356393,
            -0.02703200460356393,
            1.6,
            2.0,
        ]
        for check_left, check_right in zip(res, res_ok):
            self.assertAlmostEqual(check_left, check_right, 5)

    def test_expr6(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta3 * (self.beta2 >= self.beta1)
        )
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(
            -self.beta2 * self.Variable2
        ) / (self.beta3 * (self.beta2 >= self.beta1))
        omega = ex.RandomVariable('omega')
        a = 0
        b = 1
        x = a + (b - a) / (1 + ex.exp(-omega))
        dx = (b - a) * ex.exp(-omega) * (1 + ex.exp(-omega)) ** (-2)
        integrand = x * x
        expr4 = ex.Integrate(integrand * dx / (b - a), 'omega')
        expr6 = ex.bioMultSum([expr1, expr2, expr4])
        res = list(expr6.get_value_c(database=self.myData, prepare_ids=True))
        res_ok = [
            4.063012266030643,
            8.063012266030643,
            12.063012266030643,
            16.063012266030643,
            20.063012266030643,
        ]
        for check_left, check_right in zip(res, res_ok):
            self.assertAlmostEqual(check_left, check_right, 5)

    def test_expr7(self):
        self.beta1.status = 1
        self.beta2.status = 1
        V = {0: -self.beta1, 1: -self.beta2, 2: -self.beta1}
        av = {0: 1, 1: 1, 2: 1}
        expr7 = ex.LogLogit(V, av, 1)
        r = expr7.get_value()
        self.assertAlmostEqual(r, -1.2362866960692134, 5)
        expr8 = models.loglogit(V, av, 1)
        res = expr8.get_value_c(database=self.myData, prepare_ids=True)
        for v in res:
            self.assertAlmostEqual(v, -1.2362866960692136, 5)

    def test_expr9(self):
        V = {0: -self.beta1, 1: -self.beta2, 2: -self.beta1}
        av = {0: 1, 1: 1, 2: 1}
        expr8 = models.loglogit(V, av, 1)
        expr9 = ex.Derive(expr8, 'beta2')
        res = expr9.get_value_c(database=self.myData, prepare_ids=True)
        for v in res:
            self.assertAlmostEqual(v, -0.7095392129298093, 5)

    def test_expr10(self):
        expr10 = ex.bioNormalCdf(self.Variable1 / 10 - 1)
        res = expr10.get_value_c(database=self.myData, prepare_ids=True)
        for i, j in zip(
            res,
            [
                0.5,
                0.8413447460685283,
                0.9772498680518218,
                0.99865010196837,
                0.9999683287581669,
            ],
        ):
            self.assertAlmostEqual(i, j, 5)

    def test_expr11(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta3 * (self.beta2 >= self.beta1)
        )
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(
            -self.beta2 * self.Variable2
        ) / (self.beta3 * (self.beta2 >= self.beta1))
        expr5 = ex.Elem({1: expr1, 2: expr2}, self.Person) / 10
        expr10 = ex.bioNormalCdf(self.Variable1 / 10 - 1)
        expr11 = ex.bioMin(expr5, expr10)
        res = expr11.get_value_c(database=self.myData, prepare_ids=True)
        res_ok = [-0.027032, -0.027032, -0.027032, 0.9986501, 0.99996833]
        for i, j in zip(
            res,
            res_ok,
        ):
            self.assertAlmostEqual(i, j, 5)

    def test_expr12(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta3 * (self.beta2 >= self.beta1)
        )
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(
            -self.beta2 * self.Variable2
        ) / (self.beta3 * (self.beta2 >= self.beta1))
        expr5 = ex.Elem({1: expr1, 2: expr2}, self.Person) / 10
        expr10 = ex.bioNormalCdf(self.Variable1 / 10 - 1)
        expr12 = ex.bioMax(expr5, expr10)
        res = expr12.get_value_c(database=self.myData, prepare_ids=True)
        for i, j in zip(res, [0.5, 0.8413447460685283, 0.9772498680518218, 1.6, 2.0]):
            self.assertAlmostEqual(i, j, 5)

    def test_expr13(self):
        newvar = self.myData.define_variable('newvar', self.Variable1 + self.Variable2)
        terms = [
            LinearTermTuple(beta=self.beta1, x=ex.Variable('Variable1')),
            LinearTermTuple(beta=self.beta2, x=ex.Variable('Variable2')),
            LinearTermTuple(beta=self.beta3, x=newvar),
        ]
        expr13 = ex.bioLinearUtility(terms)
        res = expr13.get_value_c(database=self.myData, prepare_ids=True)
        res_ok = [
            152,
            304,
            456,
            608,
            760,
        ]
        for i, j in zip(res, res_ok):
            self.assertAlmostEqual(i, j, 5)
        expr13bis = (
            self.beta1 * ex.Variable('Variable1')
            + self.beta2 * ex.Variable('Variable2')
            + self.beta3 * newvar
        )
        res = expr13bis.get_value_c(database=self.myData, prepare_ids=True)
        res_ok = [
            152,
            304,
            456,
            608,
            760,
        ]
        for i, j in zip(res, res_ok):
            self.assertAlmostEqual(i, j, 5)

    def test_panel(self):
        ss = self.myData.get_sample_size()
        self.assertEqual(ss, 5)
        self.myData.panel('Person')
        ss = self.myData.get_sample_size()
        self.assertEqual(ss, 2)

    def test_expr14(self):
        c1 = ex.bioDraws('draws1', 'NORMAL_HALTON2')
        c2 = ex.bioDraws('draws2', 'NORMAL_HALTON2')
        U1 = ex.Beta('beta1', 0, None, None, 0) * ex.Variable('Variable1') + 10 * c1
        U2 = ex.Beta('beta2', 0, None, None, 0) * ex.Variable('Variable2') + 10 * c2
        U3 = 0
        U = {1: U1, 2: U2, 3: U3}
        av = {1: self.Av1, 2: self.Av2, 3: self.Av3}
        expr14 = ex.log(
            ex.MonteCarlo(
                ex.PanelLikelihoodTrajectory(models.logit(U, av, self.Choice))
            )
        )
        self.myData.panel('Person')
        res = expr14.get_value_c(
            database=self.myData, number_of_draws=100000, prepare_ids=True
        )
        res_ok = [-3.93304323, -2.10368902]
        for i, j in zip(res, res_ok):
            self.assertAlmostEqual(i, j, 3)
        # f_list, g_list, h_list, b_list
        the_function_output: BiogemeDisaggregateFunctionOutput = (
            expr14.get_value_and_derivatives(
                database=self.myData,
                number_of_draws=10000000,
                gradient=True,
                hessian=True,
                bhhh=True,
                aggregation=False,
                prepare_ids=True,
            )
        )
        g_flat = the_function_output.gradients.flatten() / 100
        h_flat = the_function_output.hessians.flatten() / 1000
        b_flat = the_function_output.bhhhs.flatten() / 1000
        f_ok = [-3.93304323, -2.10368902]
        for i, j in zip(the_function_output.functions, f_ok):
            self.assertAlmostEqual(i, j, 3)
        g_ok = [-0.12583932, 0.74160679, -0.03182045, 0.68179551]
        for i, j in zip(g_flat, g_ok):
            self.assertAlmostEqual(i, j, 2)
        h_ok = [
            -0.16720814,
            1.59974094,
            1.59974094,
            -16.72081405,
            -0.98711321,
            9.80068839,
            9.80068839,
            -98.71132074,
        ]
        for i, j in zip(h_flat, h_ok):
            self.assertAlmostEqual(i, j, 2)
        b_ok = [
            0.15835535,
            -0.93323295,
            -0.93323295,
            5.49980634,
            0.01012541,
            -0.21695039,
            -0.21695039,
            4.64845114,
        ]
        for i, j in zip(b_flat, b_ok):
            self.assertAlmostEqual(i, j, 2)

    def test_belongs_to(self):
        yes = ex.BelongsTo(ex.Numeric(2), {1, 2, 3})
        self.assertEqual(yes.get_value_c(prepare_ids=True), 1.0)
        no = ex.BelongsTo(ex.Numeric(4), {1, 2, 3})
        self.assertEqual(no.get_value_c(prepare_ids=True), 0.0)

    def test_conditional_sum(self):
        the_terms = [
            ex.ConditionalTermTuple(condition=ex.Numeric(0), term=ex.Numeric(10)),
            ex.ConditionalTermTuple(condition=ex.Numeric(1), term=ex.Numeric(20)),
        ]
        the_sum = ex.ConditionalSum(the_terms)
        self.assertEqual(the_sum.get_value_c(prepare_ids=True), 20)

    def test_sin(self):
        beta = ex.Beta('Beta', 0.11, None, None, 0)
        the_sin = ex.sin(2 * beta)
        expression_function = the_sin.create_function()
        the_function_output: BiogemeFunctionOutput = expression_function([1.0])
        self.assertAlmostEqual(the_function_output.function, 0.909297, 3)
        optimization_function = the_sin.create_objective_function()
        f, g, h, gdiff, hdiff = optimization_function.check_derivatives(x=[1.0])
        norm_g = np.linalg.norm(gdiff)
        self.assertAlmostEqual(norm_g, 0.0, delta=1.0e-4)
        norm_h = np.linalg.norm(gdiff)
        self.assertAlmostEqual(norm_h, 0.0, delta=1.0e-4)

    def test_cos(self):
        beta = ex.Beta('Beta', 0.11, None, None, 0)
        the_cos = ex.cos(2 * beta)
        expression_function = the_cos.create_function()
        the_function_output: BiogemeFunctionOutput = expression_function([1.0])
        self.assertAlmostEqual(the_function_output.function, -0.4161468365471424, 3)

        optimization_function = the_cos.create_objective_function()
        f, g, h, gdiff, hdiff = optimization_function.check_derivatives(x=[1.0])
        norm_g = np.linalg.norm(gdiff)
        self.assertAlmostEqual(norm_g, 0.0, delta=1.0e-4)
        norm_h = np.linalg.norm(gdiff)
        self.assertAlmostEqual(norm_h, 0.0, delta=1.0e-4)


class TestValidateFunction(unittest.TestCase):
    def setUp(self):
        self.outside_up = np.nextafter(MAX_VALUE, np.inf)
        self.inside_up = np.nextafter(MAX_VALUE, -np.inf)
        self.outside_low = np.nextafter(-MAX_VALUE, -np.inf)
        self.inside_low = np.nextafter(-MAX_VALUE, np.inf)

    def test_value_within_range(self):
        self.assertEqual(validate(0), 0)
        self.assertEqual(validate(self.inside_up), self.inside_up)
        self.assertEqual(validate(self.inside_low), self.inside_low)

    def test_value_exceeds_max_with_modify(self):
        self.assertEqual(validate(self.outside_up, modify=True), MAX_VALUE)

    def test_value_exceeds_max_without_modify(self):
        with self.assertRaises(excep.BiogemeError):
            validate(self.outside_up, modify=False)

    def test_value_less_than_min_with_modify(self):
        self.assertEqual(validate(self.outside_low, modify=True), -MAX_VALUE)

    def test_value_less_than_min_without_modify(self):
        with self.assertRaises(excep.BiogemeError):
            validate(self.outside_low, modify=False)

    def test_edge_case_max_value(self):
        self.assertEqual(validate(MAX_VALUE), MAX_VALUE)

    def test_edge_case_min_value(self):
        self.assertEqual(validate(-MAX_VALUE), -MAX_VALUE)


if __name__ == '__main__':
    unittest.main()
