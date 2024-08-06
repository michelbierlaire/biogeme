import unittest

import biogeme.biogeme as bio
from biogeme import models
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
from biogeme.expressions import Beta
from biogeme.parameters import Parameters

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
        self.scipy_configuration = Parameters()
        self.scipy_configuration.set_value(
            name='optimization_algorithm', value='scipy', section='Estimation'
        )
        self.scipy_configuration.set_value(name='seed', value=10, section='MonteCarlo')

        self.ls_configuration = Parameters()
        self.ls_configuration.set_value(
            name='optimization_algorithm', value='LS-newton', section='Estimation'
        )
        self.ls_configuration.set_value(name='seed', value=10, section='MonteCarlo')

        self.tr_configuration = Parameters()
        self.tr_configuration.set_value(
            name='optimization_algorithm', value='TR-newton', section='Estimation'
        )
        self.tr_configuration.set_value(name='seed', value=10, section='MonteCarlo')

        self.simple_bounds_configuration = Parameters()
        self.simple_bounds_configuration.set_value(
            name='optimization_algorithm', value='simple_bounds', section='Estimation'
        )
        self.simple_bounds_configuration.set_value(
            name='seed', value=10, section='MonteCarlo'
        )

    def testEstimationScipy(self):
        logprob = models.loglogit(V, av, CHOICE)
        biogeme = bio.BIOGEME(database, logprob, parameters=self.scipy_configuration)
        biogeme.modelName = 'test_01'
        biogeme.generate_html = False
        biogeme.generate_pickle = False
        biogeme.saveIterations = False
        biogeme.bootstrap_samples = 10
        results = biogeme.estimate(run_bootstrap=True)
        self.assertAlmostEqual(results.data.logLike, -5331.252, 2)

    def testEstimationLineSearch(self):
        logprob = models.loglogit(V, av, CHOICE)
        biogeme = bio.BIOGEME(database, logprob, parameters=self.ls_configuration)
        biogeme.modelName = 'test_01'
        biogeme.generate_html = False
        biogeme.generate_pickle = False
        biogeme.saveIterations = False
        biogeme.bootstrap_samples = 10
        results = biogeme.estimate(run_bootstrap=True)
        self.assertAlmostEqual(results.data.logLike, -5331.252, 2)

    def testEstimationTrustRegion(self):
        logprob = models.loglogit(V, av, CHOICE)
        biogeme = bio.BIOGEME(database, logprob, parameters=self.tr_configuration)
        biogeme.modelName = 'test_01'
        biogeme.generate_html = False
        biogeme.generate_pickle = False
        biogeme.saveIterations = False
        biogeme.bootstrap_samples = 10
        results = biogeme.estimate(run_bootstrap=True)
        self.assertAlmostEqual(results.data.logLike, -5331.252, 2)


if __name__ == '__main__':
    unittest.main()
