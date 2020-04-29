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
import biogeme.expressions as ex
import biogeme.models as models
from testData import myData2

class testExpressions(unittest.TestCase):
    def setUp(self):
        self.myData = myData2
        self.Person = ex.Variable('Person')
        self.Variable1 = ex.Variable('Variable1')
        self.Variable2 = ex.Variable('Variable2')
        self.Choice = ex.Variable('Choice')
        self.Av1 = ex.Variable('Av1')
        self.Av2 = ex.Variable('Av2')
        self.Av3 = ex.Variable('Av3')
        self.beta1 = ex.Beta('beta1', 1, None, None, 0)
        self.beta2 = ex.Beta('beta2', 2, None, None, 0)
        self.beta3 = ex.Beta('beta3', 3, None, None, 1)
        self.beta4 = ex.Beta('beta4', 2, None, None, 1)

    def test_DefineVariable(self):
        _ = ex.DefineVariable('newvar', self.Variable1+self.Variable2, self.myData)
        cols = self.myData.data.columns
        added = 'newvar' in cols
        self.assertTrue(added)

    def test_expr1(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (self.beta3 * (self.beta2 >= self.beta1))
        self.assertAlmostEqual(expr1.getValue(), 1.954888238921129, 5)
        res = expr1.getValue_c(self.myData)
        for v in res:
            self.assertAlmostEqual(v, 1.954888238921129, 5)

    def test_setOfBetas(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (self.beta3 * (self.beta2 >= self.beta1))
        s = expr1.setOfBetas()
        self.assertSetEqual(s, {'beta1', 'beta2'})
        s = expr1.setOfBetas(free=False, fixed=True)
        self.assertSetEqual(s, {'beta3'})
        s = expr1.setOfBetas(free=True, fixed=True)
        self.assertSetEqual(s, {'beta1', 'beta2', 'beta3'})

    def test_getElementaryExpression(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (self.beta3 * (self.beta2 >= self.beta1))
        ell = expr1.getElementaryExpression('beta2')
        self.assertEqual(ell.name, 'beta2')

    def test_expr2(self):
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(-self.beta2 * self.Variable2) / \
            (self.beta3 * (self.beta2 >= self.beta1))
        res = expr2.getValue_c(self.myData)
        self.assertListEqual(res, [20.0, 40.0, 60.0, 80.0, 100.0])


    def test_dictOfBetas(self):
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(-self.beta2 * self.Variable2) / \
            (self.beta3 * (self.beta2 >= self.beta1))
        b = expr2.dictOfBetas(free=True, fixed=True)
        # Note that the following checks only the labels. Its probably
        # good enough for our purpose.
        self.assertDictEqual(b, {'beta1': 0, 'beta2': self.beta2, 'beta3': self.beta3})

    def test_getClassName(self):
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(-self.beta2 * self.Variable2) / \
            (self.beta3 * (self.beta2 >= self.beta1))
        c = expr2.getClassName()
        self.assertEqual(c, 'Minus')

    def test_signature(self):
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(-self.beta2 * self.Variable2) / \
            (self.beta3 * (self.beta2 >= self.beta1))
        expr2._prepareFormulaForEvaluation(self.myData)
        s = expr2.getSignature()
        self.assertEqual(len(s), 17)

    def test_expr3(self):
        myDraws = ex.bioDraws('myDraws', 'UNIFORM')
        expr3 = ex.MonteCarlo(myDraws * myDraws)
        res = expr3.getValue_c(self.myData, numberOfDraws=100000)
        for v in res:
            self.assertAlmostEqual(v, 1.0 / 3.0, 2)


    def test_expr4(self):
        omega = ex.RandomVariable('omega')
        a = 0
        b = 1
        x = a + (b - a) / (1 + ex.exp(-omega))
        dx = (b - a) * ex.exp(-omega) * (1 + ex.exp(-omega))**(-2)
        integrand = x * x
        expr4 = ex.Integrate(integrand * dx /(b - a), 'omega')
        res = expr4.getValue_c(self.myData)
        for v in res:
            self.assertAlmostEqual(v, 1.0 / 3.0, 2)

    def test_expr5(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (self.beta3 * (self.beta2 >= self.beta1))
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(-self.beta2 * self.Variable2) / \
            (self.beta3 * (self.beta2 >= self.beta1))
        expr5 = ex.Elem({1: expr1, 2: expr2}, self.Person) / 10
        res = expr5.getValue_c(self.myData)
        self.assertListEqual(res, [0.19548882389211292,
                                   0.19548882389211292,
                                   0.19548882389211292,
                                   8.0,
                                   10.0])

    def test_expr6(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (self.beta3 * (self.beta2 >= self.beta1))
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(-self.beta2 * self.Variable2) / \
            (self.beta3 * (self.beta2 >= self.beta1))
        omega = ex.RandomVariable('omega')
        a = 0
        b = 1
        x = a + (b - a) / (1 + ex.exp(-omega))
        dx = (b - a) * ex.exp(-omega) * (1+ex.exp(-omega))**(-2)
        integrand = x * x
        expr4 = ex.Integrate(integrand * dx /(b - a), 'omega')
        expr6 = ex.bioMultSum([expr1, expr2, expr4])
        res = expr6.getValue_c(self.myData)
        self.assertListEqual(res, [22.28822055098741,
                                   42.28822055098741,
                                   62.28822055098741,
                                   82.2882205509874,
                                   102.2882205509874])

    def test_expr7(self):
        V = {0: -self.beta1, 1: -self.beta2, 2: -self.beta1}
        av = {0: 1, 1: 1, 2: 1}
        expr7 = ex.LogLogit(V, av, 1)
        r = expr7.getValue()
        self.assertAlmostEqual(r, -1.861994804058251, 5)
        expr8 = models.loglogit(V, av, 1)
        res = expr8.getValue_c(self.myData)
        for v in res:
            self.assertAlmostEqual(v, -1.861994804058251, 5)

    def test_expr9(self):
        V = {0: -self.beta1, 1: -self.beta2, 2: -self.beta1}
        av = {0: 1, 1: 1, 2: 1}
        expr8 = models.loglogit(V, av, 1)
        expr9 = ex.Derive(expr8, 'beta2')
        res = expr9.getValue_c(self.myData)
        for v in res:
            self.assertAlmostEqual(v, -0.8446375965030364, 5)

    def test_expr10(self):
        expr10 = ex.bioNormalCdf(self.Variable1 / 10 - 1)
        res = expr10.getValue_c(self.myData)
        for i, j in zip(res, [0.5,
                              0.8413447460685283,
                              0.9772498680518218,
                              0.99865010196837,
                              0.9999683287581669]):
            self.assertAlmostEqual(i, j, 5)

    def test_expr11(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (self.beta3 * (self.beta2 >= self.beta1))
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(-self.beta2 * self.Variable2) / \
            (self.beta3 * (self.beta2 >= self.beta1))
        expr5 = ex.Elem({1: expr1, 2: expr2}, self.Person) / 10
        expr10 = ex.bioNormalCdf(self.Variable1 / 10 - 1)
        expr11 = ex.bioMin(expr5, expr10)
        res = expr11.getValue_c(self.myData)
        for i, j in zip(res, [0.19548882389211292,
                              0.19548882389211292,
                              0.19548882389211292,
                              0.99865010196837,
                              0.9999683287581669]):
            self.assertAlmostEqual(i, j, 5)

    def test_expr12(self):
        expr1 = 2 * self.beta1 - ex.exp(-self.beta2) / (self.beta3 * (self.beta2 >= self.beta1))
        expr2 = 2 * self.beta1 * self.Variable1 - ex.exp(-self.beta2 * self.Variable2) / \
            (self.beta3 * (self.beta2 >= self.beta1))
        expr5 = ex.Elem({1: expr1, 2: expr2}, self.Person) / 10
        expr10 = ex.bioNormalCdf(self.Variable1 / 10 - 1)
        expr12 = ex.bioMax(expr5, expr10)
        res = expr12.getValue_c(self.myData)
        for i, j in zip(res, [0.5,
                              0.8413447460685283,
                              0.9772498680518218,
                              8.0,
                              10.0]):
            self.assertAlmostEqual(i, j, 5)


if __name__ == '__main__':
    unittest.main()
