"""
Simple estimation test from the Ben-Akiva and Lerman book

:author: Michel Bierlaire
:date: Wed Apr 29 18:43:34 2020

"""

import shutil
import tempfile
import unittest
from os import path

import pandas as pd

import biogeme.biogeme as bio
import biogeme.database as db
from biogeme import models
from biogeme.expressions import Beta, Variable

GROUP = Variable('GROUP')
SURVEY = Variable('SURVEY')
SP = Variable('SP')
ID = Variable('ID')
PURPOSE = Variable('PURPOSE')
FIRST = Variable('FIRST')
TICKET = Variable('TICKET')
WHO = Variable('WHO')
LUGGAGE = Variable('LUGGAGE')
AGE = Variable('AGE')
MALE = Variable('MALE')
INCOME = Variable('INCOME')
GA = Variable('GA')
ORIGIN = Variable('ORIGIN')
DEST = Variable('DEST')
TRAIN_AV = Variable('TRAIN_AV')
CAR_AV = Variable('CAR_AV')
SM_AV = Variable('SM_AV')
TRAIN_TT = Variable('TRAIN_TT')
TRAIN_CO = Variable('TRAIN_CO')
TRAIN_HE = Variable('TRAIN_HE')
SM_TT = Variable('SM_TT')
SM_CO = Variable('SM_CO')
SM_HE = Variable('SM_HE')
SM_SEATS = Variable('SM_SEATS')
CAR_TT = Variable('CAR_TT')
CAR_CO = Variable('CAR_CO')
CHOICE = Variable('CHOICE')


class test_biogeme(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.toml_file = path.join(self.test_dir, 'biogeme.toml')

        data = {
            'ID': pd.Series([i + 1 for i in range(21)]),
            'auto_time': pd.Series(
                [
                    52.9,
                    4.1,
                    4.1,
                    56.2,
                    51.8,
                    0.2,
                    27.6,
                    89.9,
                    41.5,
                    95.0,
                    99.1,
                    18.5,
                    82.0,
                    8.6,
                    22.5,
                    51.4,
                    81.0,
                    51.0,
                    62.2,
                    95.1,
                    41.6,
                ]
            ),
            'transit_time': pd.Series(
                [
                    4.4,
                    28.5,
                    86.9,
                    31.6,
                    20.2,
                    91.2,
                    79.7,
                    2.2,
                    24.5,
                    43.5,
                    8.4,
                    84.0,
                    38.0,
                    1.6,
                    74.1,
                    83.8,
                    19.2,
                    85.0,
                    90.1,
                    22.2,
                    91.5,
                ]
            ),
            'Choice': pd.Series(
                [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
            ),
        }
        dataframe = pd.DataFrame(data)

        self.database = db.Database('akiva', dataframe)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def testEstimation(self):
        AutoTime = Variable('auto_time')
        TransitTime = Variable('transit_time')
        Choice = Variable('Choice')

        ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
        B_TIME = Beta('B_TIME', 0, None, None, 0)

        V = {0: ASC_CAR + B_TIME * AutoTime, 1: B_TIME * TransitTime}

        av = {0: 1, 1: 1}

        logprob = models.loglogit(V, av, Choice)

        biogeme = bio.BIOGEME(
            self.database,
            logprob,
            parameters=self.toml_file,
            generate_html=False,
            generate_yaml=False,
            save_iterations=False,
            optimization_algorithm='scipy',
        )
        biogeme.model_name = 'test'
        results = biogeme.estimate()
        self.assertAlmostEqual(results.final_log_likelihood, -6.166042, 2)


if __name__ == '__main__':
    unittest.main()
