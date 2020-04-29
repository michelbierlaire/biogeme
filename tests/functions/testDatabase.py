"""
Test the database module

:author: Michel Bierlaire
:data: Wed Apr 29 18:02:59 2020

"""
# Bug in pylint
# pylint: disable=no-member
#
# Too constraining
# pylint: disable=invalid-name, too-many-instance-attributes
#
# Not needed in test
# pylint: disable=missing-function-docstring, missing-class-docstring

import os
import unittest

from copy import deepcopy
from pathlib import Path
import numpy as np
from biogeme.expressions import Variable, bioDraws
from testData import myData1

class testDatabase(unittest.TestCase):
    def setUp(self):
        np.random.seed(90267)
        self.myPanelData = deepcopy(myData1)
        self.Variable1 = Variable('Variable1')
        self.Variable2 = Variable('Variable2')
        self.Av1 = Variable('Av1')
        self.Av2 = Variable('Av2')
        self.Av3 = Variable('Av3')
        self.Choice = Variable('Choice')

    def test_valuesFromDatabase(self):
        expr = self.Variable1 + self.Variable2
        result = myData1.valuesFromDatabase(expr).tolist()
        self.assertListEqual(result, [11, 22, 33, 44, 55])

    def test_checkAvailabilityOfChosenAlt(self):
        avail = {1: self.Av1, 2: self.Av2, 3: self.Av3}
        result = myData1.checkAvailabilityOfChosenAlt(avail, self.Choice).tolist()
        self.assertListEqual(result, [False, True, True, True, True])

    def test_sumFromDatabase(self):
        expression = self.Variable2 / self.Variable1
        result = myData1.sumFromDatabase(expression)
        self.assertEqual(result, 50)

    def test_addColumn(self):
        Variable1 = Variable('Variable1')
        Variable2 = Variable('Variable2')
        expression = Variable2 * Variable1
        result = myData1.addColumn(expression, 'NewVariable')
        theList = result.tolist()
        self.assertListEqual(theList, [10, 40, 90, 160, 250])

    def test_count(self):
        c = myData1.count('Person', 1)
        self.assertEqual(c, 3)

    def test_remove(self):
        Exclude = Variable('Exclude')
        d = deepcopy(myData1)
        d.remove(Exclude)
        rows, _ = d.data.shape
        self.assertEqual(rows, 3)

    def test_dumpOnFile(self):
        f = myData1.dumpOnFile()
        exists = Path(f).is_file()
        os.remove(f)
        self.assertTrue(exists)

    def test_generateDraws(self):
        randomDraws1 = bioDraws('randomDraws1', 'NORMAL')
        randomDraws2 = bioDraws('randomDraws2', 'UNIFORMSYM')
        # We build an expression that involves the two random variables
        x = randomDraws1 + randomDraws2
        types = x.dictOfDraws()
        theDrawsTable = myData1.generateDraws(types, ['randomDraws1', 'randomDraws2'], 10)
        dim = theDrawsTable.shape
        self.assertTupleEqual(dim, (5, 10, 2))

    def test_setRandomGenerators(self):
        def logNormalDraws(sampleSize, numberOfDraws):
            return np.exp(np.random.randn(sampleSize, numberOfDraws))

        def exponentialDraws(sampleSize, numberOfDraws):
            return -1.0 * np.log(np.random.rand(sampleSize, numberOfDraws))

        # We associate these functions with a name
        theDict = {'LOGNORMAL': (logNormalDraws, 'Draws from lognormal distribution'),
                   'EXP': (exponentialDraws, 'Draws from exponential distributions')}
        myData1.setRandomNumberGenerators(theDict)

        # We can now generate draws from these distributions
        randomDraws1 = bioDraws('randomDraws1', 'LOGNORMAL')
        randomDraws2 = bioDraws('randomDraws2', 'EXP')
        x = randomDraws1 + randomDraws2
        types = x.dictOfDraws()
        theDrawsTable = myData1.generateDraws(types, ['randomDraws1', 'randomDraws2'], 10)
        dim = theDrawsTable.shape
        self.assertTupleEqual(dim, (5, 10, 2))

    def test_sampleWithReplacement(self):
        res1 = myData1.sampleWithReplacement()
        res2 = myData1.sampleWithReplacement(12)
        dim1 = res1.shape
        dim2 = res2.shape
        self.assertTupleEqual(dim1, (5, 9))
        self.assertTupleEqual(dim2, (12, 9))

    def test_panel(self):
        # Data is not considered panel yet
        shouldBeFalse = self.myPanelData.isPanel()
        self.myPanelData.panel('Person')
        shouldBeTrue = self.myPanelData.isPanel()
        self.assertTrue(shouldBeTrue)
        self.assertFalse(shouldBeFalse)

    def test_panelDraws(self):
        randomDraws1 = bioDraws('randomDraws1', 'NORMAL')
        randomDraws2 = bioDraws('randomDraws2', 'UNIFORMSYM')
        # We build an expression that involves the two random variables
        x = randomDraws1 + randomDraws2
        types = x.dictOfDraws()
        self.myPanelData.panel('Person')
        theDrawsTable = self.myPanelData.generateDraws(types, ['randomDraws1', 'randomDraws2'], 10)
        dim = theDrawsTable.shape
        self.assertTupleEqual(dim, (2, 10, 2))

    def test_getNumberOfObservations(self):
        self.myPanelData.panel('Person')
        self.assertEqual(myData1.getNumberOfObservations(), 5)
        self.assertEqual(self.myPanelData.getNumberOfObservations(), 5)

    def test_samplesSize(self):
        self.myPanelData.panel('Person')
        self.assertEqual(myData1.getSampleSize(), 5)
        self.assertEqual(self.myPanelData.getSampleSize(), 2)

    def test_sampleIndividualMapWithReplacement(self):
        self.myPanelData.panel('Person')
        res = self.myPanelData.sampleIndividualMapWithReplacement(10)
        dim = res.shape
        self.assertTupleEqual(dim, (10, 2))

if __name__ == '__main__':
    unittest.main()
