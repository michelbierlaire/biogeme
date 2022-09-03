import os
import unittest
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models

from spec_optima import V, av, Choice

myPath = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(f'{myPath}/optima.dat', sep='\t')
df.loc[df['OccupStat'] > 2, 'OccupStat'] = 3
df.loc[df['OccupStat'] == -1, 'OccupStat'] = 3
df.loc[df['Education'] <= 3, 'Education'] = 3
df.loc[df['Education'] <= 3, 'Education'] = 3
df.loc[df['Education'] == 5, 'Education'] = 4
df.loc[df['Education'] == 8, 'Education'] = 7
df.loc[df['TripPurpose'] != 1, 'TripPurpose'] = 2
df.loc[df['CarAvail'] != 3, 'CarAvail'] = 1

exclude = (
    (df.Choice == -1)
    | (df.CostCarCHF < 0)
    | (df.CarAvail == 3) & (df.Choice == 1)
)
df.drop(df[exclude].index, inplace=True)
df = df.reset_index()

database = db.Database('optima', df)



class test_01(unittest.TestCase):
    def testQuickEstimate(self):
        logprob = models.loglogit(V, av, Choice)
        biogeme = bio.BIOGEME(database, logprob)
        biogeme.modelName = 'test_01'
        biogeme.generateHtml = False
        biogeme.generatePickle = False
        results = biogeme.quickEstimate()
        self.assertAlmostEqual(results.data.logLike, -1068.758, 2)

if __name__ == '__main__':
    unittest.main()
