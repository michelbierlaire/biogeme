"""
Test the expressions module

:author: Michel Bierlaire
:data: Wed Apr 29 17:47:53 2020

"""
# Bug in pylint
# pylint: disable=no-member
#
# Too constraining
# pylint: disable=invalid-name, too-many-instance-attributes
#
# Not needed in test
# pylint: disable=missing-function-docstring, missing-class-docstring

import unittest
import numpy as np
import pandas as pd
import biogeme.exceptions as excep
import biogeme.expressions as ex
from biogeme import models
from biogeme.idmanager import IdManager
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

    def assertDataframeEqual(self, a, b, msg):
        try:
            pd.testing.assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def test_isNumeric(self):
        result = ex.isNumeric(1)
        self.assertTrue(result)
        result = ex.isNumeric(0.1)
        self.assertTrue(result)
        result = ex.isNumeric(True)
        self.assertTrue(result)
        result = ex.isNumeric(self)
        self.assertFalse(result)

    def test_add(self):
        result = self.Variable1 + self.Variable2
        self.assertEqual(result.__str__(), '(Variable1 + Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 + 1
        self.assertEqual(result.__str__(), '(Variable1 + `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 + self.Variable1
        self.assertEqual(result.__str__(), '(`1.0` + Variable1)')
        self.assertTrue(result.children[1] is self.Variable1)

        with self.assertRaises(excep.biogemeError):
            result = self + self.Variable1

        with self.assertRaises(excep.biogemeError):
            result = self.Variable1 + self

    def test_sub(self):
        result = self.Variable1 - self.Variable2
        self.assertEqual(result.__str__(), '(Variable1 - Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 - 1
        self.assertEqual(result.__str__(), '(Variable1 - `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 - self.Variable1
        self.assertEqual(result.__str__(), '(`1.0` - Variable1)')
        self.assertTrue(result.children[1] is self.Variable1)

        with self.assertRaises(excep.biogemeError):
            result = self - self.Variable1

        with self.assertRaises(excep.biogemeError):
            result = self.Variable1 - self

    def test_mul(self):
        result = self.Variable1 * self.Variable2
        self.assertEqual(result.__str__(), '(Variable1 * Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 * 1
        self.assertEqual(result.__str__(), '(Variable1 * `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 * self.Variable1
        self.assertEqual(result.__str__(), '(`1.0` * Variable1)')
        self.assertTrue(result.children[1] is self.Variable1)

        with self.assertRaises(excep.biogemeError):
            result = self * self.Variable1

        with self.assertRaises(excep.biogemeError):
            result = self.Variable1 * self

    def test_div(self):
        result = self.Variable1 / self.Variable2
        self.assertEqual(result.__str__(), '(Variable1 / Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 / 1
        self.assertEqual(result.__str__(), '(Variable1 / `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 / self.Variable1
        self.assertEqual(result.__str__(), '(`1.0` / Variable1)')
        self.assertTrue(result.children[1] is self.Variable1)

        with self.assertRaises(excep.biogemeError):
            result = self / self.Variable1

        with self.assertRaises(excep.biogemeError):
            result = self.Variable1 / self

    def test_neg(self):
        result = -self.Variable1
        self.assertEqual(result.__str__(), '(-Variable1)')
        self.assertTrue(result.children[0] is self.Variable1)

    def test_pow(self):
        result = self.Variable1**self.Variable2
        self.assertEqual(result.__str__(), '(Variable1 ** Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1**1
        self.assertEqual(result.__str__(), '(Variable1 ** `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1**self.Variable1
        self.assertEqual(result.__str__(), '(`1.0` ** Variable1)')
        self.assertTrue(result.children[1] is self.Variable1)

        with self.assertRaises(excep.biogemeError):
            result = self**self.Variable1

        with self.assertRaises(excep.biogemeError):
            result = self.Variable1**self

    def test_and(self):
        result = self.Variable1 & self.Variable2
        self.assertEqual(result.__str__(), '(Variable1 and Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 & 1
        self.assertEqual(result.__str__(), '(Variable1 and `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 & self.Variable1
        self.assertEqual(result.__str__(), '(`1.0` and Variable1)')
        self.assertTrue(result.children[1] is self.Variable1)

        with self.assertRaises(excep.biogemeError):
            result = self & self.Variable1

        with self.assertRaises(excep.biogemeError):
            result = self.Variable1 & self

    def test_or(self):
        result = self.Variable1 | self.Variable2
        self.assertEqual(result.__str__(), '(Variable1 or Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 | 1
        self.assertEqual(result.__str__(), '(Variable1 or `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 | self.Variable1
        self.assertEqual(result.__str__(), '(`1.0` or Variable1)')
        self.assertTrue(result.children[1] is self.Variable1)

        with self.assertRaises(excep.biogemeError):
            result = self | self.Variable1

        with self.assertRaises(excep.biogemeError):
            result = self.Variable1 | self

    def test_eq(self):
        result = self.Variable1 == self.Variable2
        self.assertEqual(result.__str__(), '(Variable1 == Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 == 1
        self.assertEqual(result.__str__(), '(Variable1 == `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 == self.Variable1
        self.assertEqual(result.__str__(), '(Variable1 == `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        with self.assertRaises(excep.biogemeError):
            result = self == self.Variable1

        with self.assertRaises(excep.biogemeError):
            result = self.Variable1 == self

    def test_neq(self):
        result = self.Variable1 != self.Variable2
        self.assertEqual(result.__str__(), '(Variable1 != Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 != 1
        self.assertEqual(result.__str__(), '(Variable1 != `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 != self.Variable1
        self.assertEqual(result.__str__(), '(Variable1 != `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        with self.assertRaises(excep.biogemeError):
            result = self != self.Variable1

        with self.assertRaises(excep.biogemeError):
            result = self.Variable1 != self

    def test_le(self):
        result = self.Variable1 <= self.Variable2
        self.assertEqual(result.__str__(), '(Variable1 <= Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 <= 1
        self.assertEqual(result.__str__(), '(Variable1 <= `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 <= self.Variable1
        self.assertEqual(result.__str__(), '(Variable1 >= `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        with self.assertRaises(excep.biogemeError):
            result = self <= self.Variable1

        with self.assertRaises(excep.biogemeError):
            result = self.Variable1 <= self

    def test_ge(self):
        result = self.Variable1 >= self.Variable2
        self.assertEqual(result.__str__(), '(Variable1 >= Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 >= 1
        self.assertEqual(result.__str__(), '(Variable1 >= `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 >= self.Variable1
        self.assertEqual(result.__str__(), '(Variable1 <= `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        with self.assertRaises(excep.biogemeError):
            result = self >= self.Variable1

        with self.assertRaises(excep.biogemeError):
            result = self.Variable1 >= self

    def test_lt(self):
        result = self.Variable1 < self.Variable2
        self.assertEqual(result.__str__(), '(Variable1 < Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 < 1
        self.assertEqual(result.__str__(), '(Variable1 < `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 < self.Variable1
        self.assertEqual(result.__str__(), '(Variable1 > `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        with self.assertRaises(excep.biogemeError):
            result = self < self.Variable1

        with self.assertRaises(excep.biogemeError):
            result = self.Variable1 < self

    def test_gt(self):
        result = self.Variable1 > self.Variable2
        self.assertEqual(result.__str__(), '(Variable1 > Variable2)')
        self.assertTrue(result.children[0] is self.Variable1)
        self.assertTrue(result.children[1] is self.Variable2)

        result = self.Variable1 > 1
        self.assertEqual(result.__str__(), '(Variable1 > `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        result = 1 > self.Variable1
        self.assertEqual(result.__str__(), '(Variable1 < `1.0`)')
        self.assertTrue(result.children[0] is self.Variable1)

        with self.assertRaises(excep.biogemeError):
            result = self > self.Variable1

        with self.assertRaises(excep.biogemeError):
            result = self.Variable1 > self

    def test_getValue_c(self):
        result = self.Variable1.getValue_c(
            database=self.myData, prepareIds=True
        )
        np.testing.assert_equal(result, [10, 20, 30, 40, 50])

    def test_DefineVariable(self):
        _ = self.myData.DefineVariable(
            'newvar_b', self.Variable1 + self.Variable2
        )
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
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        self.assertAlmostEqual(expr1.getValue(), -1.275800115089098, 5)
        res = expr1.getValue_c(prepareIds=True)
        self.assertAlmostEqual(res, -1.275800115089098, 5)

    def test_expr1_newvalues(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        newvalues = {'beta1': 1, 'beta2': 2, 'beta3': 3, 'beta4': 2}
        expr1.changeInitValues(newvalues)
        self.assertAlmostEqual(expr1.getValue(), 1.9323323583816936, 5)
        res = expr1.getValue_c(prepareIds=True)
        self.assertAlmostEqual(res, 1.9323323583816936, 5)

    def test_expr1_derivatives(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        newvalues = {'beta1': 1, 'beta2': 2, 'beta3': 3, 'beta4': 2}
        expr1.changeInitValues(newvalues)
        f, g, h, b = expr1.getValueAndDerivatives(prepareIds=True)
        self.assertAlmostEqual(f, 1.9323323583816936, 5)
        g_ok = [2.0, 0.10150146242745953]
        h0_ok = [0.0, 0.0]
        h1_ok = [0.0, -0.16916910404576588]
        bhhh0_ok = [4, 0.20300292]
        bhhh1_ok = [0.20300292, 0.01030255]
        for check_left, check_right in zip(g, g_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(h[0], h0_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(h[1], h1_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(b[0], bhhh0_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(b[1], bhhh1_ok):
            self.assertAlmostEqual(check_left, check_right, 5)

    def test_expr1_gradient(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        f, g, h, b = expr1.getValueAndDerivatives(
            gradient=True, hessian=False, bhhh=False, prepareIds=True
        )
        self.assertAlmostEqual(f, -1.275800115089098, 5)
        g_ok = [2.0, 5.865300402811844]
        for check_left, check_right in zip(g, g_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        self.assertIsNone(h)
        self.assertIsNone(b)

    def test_expr1_function(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        the_function = expr1.createFunction()
        f, g, h = the_function([1, 2])
        self.assertAlmostEqual(f, 1.9323323583816936, 5)
        g_ok = [2.0, 0.10150146242745953]
        h0_ok = [0.0, 0.0]
        h1_ok = [0.0, -0.16916910404576588]
        for check_left, check_right in zip(g, g_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(h[0], h0_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(h[1], h1_ok):
            self.assertAlmostEqual(check_left, check_right, 5)

        f, g, h = the_function([10, -2])
        self.assertAlmostEqual(f, 23.694528049465326, 5)
        g_ok = [2, -1.84726402]
        h0_ok = [0.0, 0.0]
        h1_ok = [0.0, 1.84726402]
        for check_left, check_right in zip(g, g_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(h[0], h0_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(h[1], h1_ok):
            self.assertAlmostEqual(check_left, check_right, 5)

    def test_expr1_database(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        f_list, g_list, h_list, bhhh_list = expr1.getValueAndDerivatives(
            database=self.myData, aggregation=False, prepareIds=True
        )
        f_ok = -1.27580012
        g_ok = [2.0, 5.8653004]
        h0_ok = [0.0, 0.0]
        h1_ok = [0.0, -31.00230213]
        bhhh0_ok = [4.0, 11.73060081]
        bhhh1_ok = [11.73060081, 34.40174882]
        for i, f in enumerate(f_list):
            self.assertAlmostEqual(f, f_ok, 5)
            for check_left, check_right in zip(g_list[i], g_ok):
                self.assertAlmostEqual(check_left, check_right, 5)
            for check_left, check_right in zip(h_list[i][0], h0_ok):
                self.assertAlmostEqual(check_left, check_right, 5)
            for check_left, check_right in zip(h_list[i][1], h1_ok):
                self.assertAlmostEqual(check_left, check_right, 5)
            for check_left, check_right in zip(bhhh_list[i][0], bhhh0_ok):
                self.assertAlmostEqual(check_left, check_right, 5)
            for check_left, check_right in zip(bhhh_list[i][1], bhhh1_ok):
                self.assertAlmostEqual(check_left, check_right, 5)

    def test_expr1_database_agg(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        f, g, h, b = expr1.getValueAndDerivatives(
            database=self.myData, aggregation=True, prepareIds=True
        )
        f_ok = -6.37900057544549
        g_ok = [10.0, 29.32650201405922]
        h0_ok = [0.0, 0.0]
        h1_ok = [0.0, -155.01151064574157]
        b0_ok = [20.0, 58.65300403]
        b1_ok = [58.65300403, 172.00874408]
        self.assertAlmostEqual(f, f_ok, 5)
        for check_left, check_right in zip(g, g_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(h[0], h0_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(h[1], h1_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(b[0], b0_ok):
            self.assertAlmostEqual(check_left, check_right, 5)
        for check_left, check_right in zip(b[1], b1_ok):
            self.assertAlmostEqual(check_left, check_right, 5)

    def test_setOfBetas(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta3 * (self.beta2 >= self.beta1)
        )
        s = expr1.setOfBetas()
        self.assertSetEqual(s, {'beta1', 'beta2'})
        s = expr1.setOfBetas(free=False, fixed=True)
        self.assertSetEqual(s, {'beta3'})
        s = expr1.setOfBetas(free=True, fixed=True)
        self.assertSetEqual(s, {'beta1', 'beta2', 'beta3'})

    def test_setOfVariables(self):
        expr1 = 2 * self.Variable1 - ex.exp(-self.Variable2) / (
            self.Variable1 * (self.Variable1 >= self.Variable2)
        )
        s = expr1.setOfVariables()
        self.assertSetEqual(s, {'Variable1', 'Variable2'})

    def test_getElementaryExpression(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (
            self.beta3 * (self.beta2 >= self.beta1)
        )
        ell = expr1.getElementaryExpression('beta2')
        self.assertEqual(ell.name, 'beta2')

    def test_expr2(self):
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(
            -self.beta2 * self.Variable2
        ) / (
            self.beta2 * (self.beta3 >= self.beta4)
            + self.beta1 * (self.beta3 < self.beta4)
        )
        with self.assertRaises(excep.biogemeError):
            expr2.getValue()
        with self.assertRaises(excep.biogemeError):
            expr2.getValue_c()
        res = list(expr2.getValue_c(database=self.myData, prepareIds=True))
        self.assertListEqual(res, [4.0, 8.0, 12.0, 16.0, 20.0])

    def test_dictOfBetas(self):
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(
            -self.beta2 * self.Variable2
        ) / (self.beta3 * (self.beta2 >= self.beta1))
        b = expr2.dictOfBetas(free=True, fixed=True)
        # Note that the following checks only the labels. Its probably
        # good enough for our purpose.
        self.assertDictEqual(
            b, {'beta1': 0, 'beta2': self.beta2, 'beta3': self.beta3}
        )

    def test_dictOfVariables(self):
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(
            -self.beta2 * self.Variable2
        ) / (self.beta3 * (self.beta2 >= self.beta1))
        b = expr2.dictOfVariables()
        # Note that the following checks only the labels. Its probably
        # good enough for our purpose.
        self.assertDictEqual(
            b, {'Variable1': self.Variable1, 'Variable2': self.Variable2}
        )

    def test_dictOfRandomVariables(self):
        expr = -(self.omega1 + self.omega2 + self.Variable1)
        b = expr.dictOfRandomVariables()
        self.assertDictEqual(b, {'omega1': self.omega1, 'omega2': self.omega2})

    def test_dictOfDraws(self):
        expr = -(self.xi1 + self.xi2 - self.xi3 + self.Variable1)
        b = expr.dictOfDraws()
        self.assertDictEqual(
            b, {'xi1': 'NORMAL', 'xi2': 'UNIF', 'xi3': 'WRONGTYPE'}
        )

    def test_getClassName(self):
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(
            -self.beta2 * self.Variable2
        ) / (self.beta3 * (self.beta2 >= self.beta1))
        c = expr2.getClassName()
        self.assertEqual(c, 'Minus')

    def test_getSignature(self):
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(
            -self.beta2 * self.Variable2
        ) / (self.beta3 * (self.beta2 >= self.beta1))
        expr2.setIdManager(IdManager([expr2], self.myData, 0))
        s = expr2.getSignature()
        self.assertEqual(len(s), 17)

    def test_embedExpression(self):
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(
            -self.beta2 * self.Variable2
        ) / (self.beta3 * (self.beta2 >= self.beta1))
        self.assertTrue(expr2.embedExpression('Minus'))
        self.assertFalse(expr2.embedExpression('bioDraws'))

    def test_panel_variables(self):
        expr_ok = ex.PanelLikelihoodTrajectory(self.Variable1)
        check_ok = expr_ok.check_panel_trajectory()
        self.assertEqual(len(check_ok), 0)
        expr_not_ok = self.Variable2 + ex.PanelLikelihoodTrajectory(
            self.Variable1
        )
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
        c1 = expr1.countPanelTrajectoryExpressions()
        self.assertEqual(c1, 0)
        expr2 = ex.PanelLikelihoodTrajectory(self.beta1)
        c2 = expr2.countPanelTrajectoryExpressions()
        self.assertEqual(c2, 1)
        expr3 = ex.PanelLikelihoodTrajectory(
            ex.PanelLikelihoodTrajectory(self.beta1)
        )
        c3 = expr3.countPanelTrajectoryExpressions()
        expr4 = self.Variable1 + ex.PanelLikelihoodTrajectory(
            ex.PanelLikelihoodTrajectory(self.beta1)
        )
        c4 = expr4.countPanelTrajectoryExpressions()
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
        collectionOfFormulas = [expr1, expr2]
        formulas = IdManager(collectionOfFormulas, self.myData, None)
        elementaryExpressionIndex_ok = {
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
            elementaryExpressionIndex_ok,
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
        with self.assertRaises(excep.biogemeError):
            res = expr3.getValue_c(numberOfDraws=100000)
        res = expr3.getValue_c(
            database=self.myData, numberOfDraws=100000, prepareIds=True
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
        res = expr4.getValue_c(database=self.myData, prepareIds=True)
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
        res = list(expr5.getValue_c(database=self.myData, prepareIds=True))
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
        res = list(expr6.getValue_c(database=self.myData, prepareIds=True))
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
        V = {0: -self.beta1, 1: -self.beta2, 2: -self.beta1}
        av = {0: 1, 1: 1, 2: 1}
        expr7 = ex.LogLogit(V, av, 1)
        r = expr7.getValue()
        self.assertAlmostEqual(r, -1.2362866960692134, 5)
        expr8 = models.loglogit(V, av, 1)
        res = expr8.getValue_c(database=self.myData, prepareIds=True)
        for v in res:
            self.assertAlmostEqual(v, -1.2362866960692136, 5)

    def test_expr9(self):
        V = {0: -self.beta1, 1: -self.beta2, 2: -self.beta1}
        av = {0: 1, 1: 1, 2: 1}
        expr8 = models.loglogit(V, av, 1)
        expr9 = ex.Derive(expr8, 'beta2')
        res = expr9.getValue_c(database=self.myData, prepareIds=True)
        for v in res:
            self.assertAlmostEqual(v, -0.7095392129298093, 5)

    def test_expr10(self):
        expr10 = ex.bioNormalCdf(self.Variable1 / 10 - 1)
        res = expr10.getValue_c(database=self.myData, prepareIds=True)
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
        res = expr11.getValue_c(database=self.myData, prepareIds=True)
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
        res = expr12.getValue_c(database=self.myData, prepareIds=True)
        for i, j in zip(
            res, [0.5, 0.8413447460685283, 0.9772498680518218, 1.6, 2.0]
        ):
            self.assertAlmostEqual(i, j, 5)

    def test_expr13(self):
        newvar = self.myData.DefineVariable(
            'newvar', self.Variable1 + self.Variable2
        )
        terms = [
            (self.beta1, ex.Variable('Variable1')),
            (self.beta2, ex.Variable('Variable2')),
            (self.beta3, newvar),
        ]
        expr13 = ex.bioLinearUtility(terms)
        res = expr13.getValue_c(database=self.myData, prepareIds=True)
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
        res = expr13bis.getValue_c(database=self.myData, prepareIds=True)
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
        ss = self.myData.getSampleSize()
        self.assertEqual(ss, 5)
        self.myData.panel('Person')
        ss = self.myData.getSampleSize()
        self.assertEqual(ss, 2)

    def test_expr14(self):
        c1 = ex.bioDraws('draws1', 'NORMAL_HALTON2')
        c2 = ex.bioDraws('draws2', 'NORMAL_HALTON2')
        U1 = (
            ex.Beta('beta1', 0, None, None, 0) * ex.Variable('Variable1')
            + 10 * c1
        )
        U2 = (
            ex.Beta('beta2', 0, None, None, 0) * ex.Variable('Variable2')
            + 10 * c2
        )
        U3 = 0
        U = {1: U1, 2: U2, 3: U3}
        av = {1: self.Av1, 2: self.Av2, 3: self.Av3}
        expr14 = ex.log(
            ex.MonteCarlo(
                ex.PanelLikelihoodTrajectory(models.logit(U, av, self.Choice))
            )
        )
        self.myData.panel('Person')
        res = expr14.getValue_c(
            database=self.myData, numberOfDraws=100000, prepareIds=True
        )
        res_ok = [-3.93304323, -2.10368902]
        for i, j in zip(res, res_ok):
            self.assertAlmostEqual(i, j, 3)

        f_list, g_list, h_list, b_list = expr14.getValueAndDerivatives(
            database=self.myData,
            numberOfDraws=10000000,
            gradient=True,
            hessian=True,
            bhhh=True,
            aggregation=False,
            prepareIds=True,
        )
        g_flat = g_list.flatten() / 100
        h_flat = h_list.flatten() / 1000
        b_flat = b_list.flatten() / 1000
        f_ok = [-3.93304323, -2.10368902]
        for i, j in zip(f_list, f_ok):
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


if __name__ == '__main__':
    unittest.main()
