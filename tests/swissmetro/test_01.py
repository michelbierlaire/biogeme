import os
import shutil
import tempfile
import unittest

import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta

from biogeme.data.swissmetro import (
    read_data,
    PURPOSE,
    CHOICE,
    GA,
    TRAIN_CO,
    SM_CO,
    SM_AV,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
    SM_TT_SCALED,
    SM_COST_SCALED,
    CAR_TT_SCALED,
    CAR_CO_SCALED,
    TRAIN_AV_SP,
    CAR_AV_SP,
)

database = read_data()
# Keep only trip purposes 1 (commuter) and 3 (business)
exclude = ((PURPOSE != 1) * (PURPOSE != 3)) > 0
database.remove(exclude)

ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)


V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}


# Associate the availability conditions with the alternatives

av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}


class test_01(unittest.TestCase):
    def setUp(self) -> None:
        """Create the configuration .py"""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.scipy_file = os.path.join(self.test_dir, 'scipy.toml')
        with open(self.scipy_file, 'w', encoding='utf-8') as f:
            print('[Estimation]', file=f)
            print('optimization_algorithm = "scipy"', file=f)
            print('[MonteCarlo]', file=f)
            print('seed = 10', file=f)
        self.ls_file = os.path.join(self.test_dir, 'line_search.toml')
        with open(self.ls_file, 'w', encoding='utf-8') as f:
            print('[Estimation]', file=f)
            print('optimization_algorithm = "LS-newton"', file=f)
            print('[MonteCarlo]', file=f)
            print('seed = 10', file=f)
        self.tr_file = os.path.join(self.test_dir, 'trust_region.toml')
        with open(self.tr_file, 'w', encoding='utf-8') as f:
            print('[Estimation]', file=f)
            print('optimization_algorithm = "TR-newton"', file=f)
            print('[MonteCarlo]', file=f)
            print('seed = 10', file=f)
        self.simple_bounds_file = os.path.join(self.test_dir, 'simple_bounds.toml')
        with open(self.simple_bounds_file, 'w', encoding='utf-8') as f:
            print('[Estimation]', file=f)
            print('optimization_algorithm = "simple_bounds"', file=f)
            print('[MonteCarlo]', file=f)
            print('seed = 10', file=f)

    def testEstimationScipy(self):
        logprob = models.loglogit(V, av, CHOICE)
        biogeme = bio.BIOGEME(database, logprob, parameter_file=self.scipy_file)
        biogeme.modelName = 'test_01'
        biogeme.generate_html = False
        biogeme.generate_pickle = False
        biogeme.saveIterations = False
        biogeme.bootstrap_samples = 10
        results = biogeme.estimate(run_bootstrap=True)
        self.assertAlmostEqual(results.data.logLike, -5331.252, 2)

    def testEstimationLineSearch(self):
        logprob = models.loglogit(V, av, CHOICE)
        biogeme = bio.BIOGEME(database, logprob, parameter_file=self.ls_file)
        biogeme.modelName = 'test_01'
        biogeme.generate_html = False
        biogeme.generate_pickle = False
        biogeme.saveIterations = False
        biogeme.bootstrap_samples = 10
        results = biogeme.estimate(run_bootstrap=True)
        self.assertAlmostEqual(results.data.logLike, -5331.252, 2)

    def testEstimationTrustRegion(self):
        logprob = models.loglogit(V, av, CHOICE)
        biogeme = bio.BIOGEME(database, logprob, parameter_file=self.tr_file)
        biogeme.modelName = 'test_01'
        biogeme.generate_html = False
        biogeme.generate_pickle = False
        biogeme.saveIterations = False
        biogeme.bootstrap_samples = 10
        results = biogeme.estimate(run_bootstrap=True)
        self.assertAlmostEqual(results.data.logLike, -5331.252, 2)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)


if __name__ == '__main__':
    unittest.main()
