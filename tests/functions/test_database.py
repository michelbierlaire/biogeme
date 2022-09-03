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

from pathlib import Path
import pandas as pd
import numpy as np
import biogeme.database as db
import biogeme.exceptions as excep
from biogeme.expressions import Variable, bioDraws
from test_data import (
    getData,
    output_flatten_database_1,
    output_flatten_database_2,
    output_flatten_database_3,
)


class testDatabase(unittest.TestCase):
    def setUp(self):
        np.random.seed(90267)
        self.myData1 = getData(1)
        self.myPanelData = getData(3)
        self.Variable1 = Variable('Variable1')
        self.Variable2 = Variable('Variable2')
        self.Av1 = Variable('Av1')
        self.Av2 = Variable('Av2')
        self.Av3 = Variable('Av3')
        self.Choice = Variable('Choice')

    def test_valuesFromDatabase(self):
        expr = self.Variable1 + self.Variable2
        result = self.myData1.valuesFromDatabase(expr).tolist()
        self.assertListEqual(result, [11, 22, 33, 44, 55])

    def test_checkAvailabilityOfChosenAlt(self):
        avail = {1: self.Av1, 2: self.Av2, 3: self.Av3}
        result = self.myData1.checkAvailabilityOfChosenAlt(
            avail, self.Choice
        ).tolist()
        self.assertListEqual(result, [False, True, True, True, True])

    def test_sumFromDatabase(self):
        expression = self.Variable2 / self.Variable1
        result = expression.getValue_c(
            database=self.myData1, aggregation=True, prepareIds=True
        )
        self.assertEqual(result, 50)

    def test_addColumn(self):
        Variable1 = Variable('Variable1')
        Variable2 = Variable('Variable2')
        expression = Variable2 * Variable1
        result = self.myData1.addColumn(expression, 'NewVariable')
        theList = result.tolist()
        self.assertListEqual(theList, [10, 40, 90, 160, 250])

    def test_count(self):
        c = self.myData1.count('Person', 1)
        self.assertEqual(c, 3)

    def test_remove(self):
        Exclude = Variable('Exclude')
        d = getData(1)
        d.remove(Exclude)
        rows, _ = d.data.shape
        self.assertEqual(rows, 3)

    def test_dumpOnFile(self):
        f = self.myData1.dumpOnFile()
        exists = Path(f).is_file()
        os.remove(f)
        self.assertTrue(exists)

    def test_generateDraws(self):
        randomDraws1 = bioDraws('randomDraws1', 'NORMAL')
        randomDraws2 = bioDraws('randomDraws2', 'UNIFORMSYM')
        # We build an expression that involves the two random variables
        x = randomDraws1 + randomDraws2
        types = x.dictOfDraws()
        theDrawsTable = self.myData1.generateDraws(
            types, ['randomDraws1', 'randomDraws2'], 10
        )
        dim = theDrawsTable.shape
        self.assertTupleEqual(dim, (5, 10, 2))

    def test_setRandomGenerators(self):
        def logNormalDraws(sampleSize, numberOfDraws):
            return np.exp(np.random.randn(sampleSize, numberOfDraws))

        def exponentialDraws(sampleSize, numberOfDraws):
            return -1.0 * np.log(np.random.rand(sampleSize, numberOfDraws))

        # We associate these functions with a name
        theDict = {
            'LOGNORMAL': (logNormalDraws, 'Draws from lognormal distribution'),
            'EXP': (exponentialDraws, 'Draws from exponential distributions'),
        }
        self.myData1.setRandomNumberGenerators(theDict)

        # We can now generate draws from these distributions
        randomDraws1 = bioDraws('randomDraws1', 'LOGNORMAL')
        randomDraws2 = bioDraws('randomDraws2', 'EXP')
        x = randomDraws1 + randomDraws2
        types = x.dictOfDraws()
        theDrawsTable = self.myData1.generateDraws(
            types, ['randomDraws1', 'randomDraws2'], 10
        )
        dim = theDrawsTable.shape
        self.assertTupleEqual(dim, (5, 10, 2))

    def test_sampleWithReplacement(self):
        res1 = self.myData1.sampleWithReplacement()
        res2 = self.myData1.sampleWithReplacement(12)
        dim1 = res1.shape
        dim2 = res2.shape
        self.assertTupleEqual(dim1, (5, 8))
        self.assertTupleEqual(dim2, (12, 8))

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
        theDrawsTable = self.myPanelData.generateDraws(
            types, ['randomDraws1', 'randomDraws2'], 10
        )
        dim = theDrawsTable.shape
        self.assertTupleEqual(dim, (2, 10, 2))

    def test_generateFlatPanelDataframe(self):
        # If the data has not been declared panel, an exception is raised.
        with self.assertRaises(excep.biogemeError):
            result = self.myPanelData.generateFlatPanelDataframe()
        self.myPanelData.panel('Person')
        # Test with default parameters
        result1 = self.myPanelData.generateFlatPanelDataframe()
        result1 = result1.reindex(sorted(result1.columns), axis='columns')
        compared1 = output_flatten_database_1.reindex(
            sorted(output_flatten_database_1.columns), axis='columns'
        )
        compared1.index.name = 'Person'
        pd.testing.assert_frame_equal(result1, compared1)
        # Test with automatic detection of identical columns
        result2 = self.myPanelData.generateFlatPanelDataframe(
            identical_columns=None
        )
        result2 = result2.reindex(sorted(result2.columns), axis='columns')
        compared2 = output_flatten_database_2.reindex(
            sorted(output_flatten_database_2.columns), axis='columns'
        )
        compared2.index.name = 'Person'
        pd.testing.assert_frame_equal(result2, compared2)
        # Test with explicit list of identical columns
        result3 = self.myPanelData.generateFlatPanelDataframe(
            identical_columns=['Age']
        )
        result3 = result3.reindex(sorted(result3.columns), axis='columns')
        compared3 = output_flatten_database_3.reindex(
            sorted(output_flatten_database_3.columns), axis='columns'
        )
        compared3.index.name = 'Person'
        pd.testing.assert_frame_equal(result3, compared3)

    def test_getNumberOfObservations(self):
        self.myPanelData.panel('Person')
        self.assertEqual(self.myData1.getNumberOfObservations(), 5)
        self.assertEqual(self.myPanelData.getNumberOfObservations(), 5)

    def test_samplesSize(self):
        self.myPanelData.panel('Person')
        self.assertEqual(self.myData1.getSampleSize(), 5)
        self.assertEqual(self.myPanelData.getSampleSize(), 2)

    def test_sampleIndividualMapWithReplacement(self):
        self.myPanelData.panel('Person')
        res = self.myPanelData.sampleIndividualMapWithReplacement(10)
        dim = res.shape
        self.assertTupleEqual(dim, (10, 2))

    def test_split(self):
        data = {
            'userID': [
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
                4,
                5,
                5,
                5,
                5,
                5,
            ],
            'obsID': list(range(25)),
        }
        df = pd.DataFrame(data)
        database = db.Database('test', df)
        sp = database.split(5)
        estimation, validation = sp[0]
        # Verify that the intersection is empty
        intersection = set(estimation['obsID']) & set(validation['obsID'])
        self.assertSetEqual(intersection, set())
        # Verify that the union is the whole range
        union = set(estimation['obsID']) | set(validation['obsID'])
        self.assertSetEqual(union, set(range(25)))

        sp = database.split(5, groups='userID')
        estimation, validation = sp[0]
        # Verify that the intersection is empty
        intersection = set(estimation['userID']) & set(validation['userID'])
        self.assertSetEqual(intersection, set())
        # Verify that the union is the whole range
        union = set(estimation['userID']) | set(validation['userID'])
        self.assertSetEqual(union, set(range(1, 6)))

        with self.assertRaises(excep.biogemeError):
            sp = database.split(1)


if __name__ == '__main__':
    unittest.main()
