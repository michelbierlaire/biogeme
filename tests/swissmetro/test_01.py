import unittest

import biogeme.biogeme as bio
from biogeme import models
from biogeme.data.swissmetro import (
    CAR_AV_SP,
    CAR_CO_SCALED,
    CAR_TT_SCALED,
    CHOICE,
    GA,
    PURPOSE,
    SM_AV,
    SM_CO,
    SM_COST_SCALED,
    SM_TT_SCALED,
    TRAIN_AV_SP,
    TRAIN_CO,
    TRAIN_COST_SCALED,
    TRAIN_TT_SCALED,
    read_data,
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
        self.assertFalse(logprob.is_complex())
        biogeme = bio.BIOGEME(
            database,
            logprob,
            parameters=self.scipy_configuration,
            save_iterations=False,
            generate_html=False,
            generate_yaml=False,
            bootstrap_samples=10,
        )
        biogeme.model_name = 'test_01'
        results = biogeme.estimate(run_bootstrap=True)
        self.assertAlmostEqual(results.final_log_likelihood, -5331.252, 2)

    def _testEstimationLineSearch(self):
        logprob = models.loglogit(V, av, CHOICE)
        biogeme = bio.BIOGEME(
            database,
            logprob,
            parameters=self.ls_configuration,
            save_iterations=False,
            generate_html=False,
            generate_yaml=False,
            bootstrap_samples=10,
        )
        biogeme.model_name = 'test_01'
        results = biogeme.estimate(run_bootstrap=True)
        self.assertAlmostEqual(results.final_log_likelihood, -5331.252, 2)

    def testEstimationTrustRegion(self):
        logprob = models.loglogit(V, av, CHOICE)
        biogeme = bio.BIOGEME(
            database,
            logprob,
            parameters=self.tr_configuration,
            save_iterations=False,
            generate_html=False,
            generate_yaml=False,
            bootstrap_samples=10,
        )
        biogeme.model_name = 'test_01'
        results = biogeme.estimate(run_bootstrap=True)
        self.assertAlmostEqual(results.final_log_likelihood, -5331.252, 2)


if __name__ == '__main__':
    unittest.main()
