import unittest

import pandas as pd

from biogeme.database import Database
from biogeme.expressions import Beta, Numeric, Variable
from biogeme.mdcev import GammaProfile, Translated, Generalized, NonMonotonic


class TestMdcev(unittest.TestCase):

    def setUp(self):
        first_row = {
            'PersonID': 1.0,
            'weight': 0.2244021,
            'hhsize': 1.0,
            'childnum': 0.0,
            'faminc': 6.0,
            'faminc25K': 1.0,
            'income': 17.5,
            'employed': 0.0,
            'fulltime': 0.0,
            'spousepr': 0.0,
            'spousemp': 0.0,
            'male': 0.0,
            'married': 0.0,
            'age': 85.0,
            'age2': 7225.0,
            'age15_40': 0.0,
            'age41_60': 0.0,
            'age61_85': 1.0,
            'bachigher': 0.0,
            'white': 0.0,
            'metro': 1.0,
            'diaryday': 1.0,
            'Sunday': 1.0,
            'holiday': 0.0,
            'weekearn': -1.0,
            'weekwordur': -1.0,
            'hhchild': 0.0,
            'ohhchild': 0.0,
            't1': 0.0,
            't2': 240.0,
            't3': 0.0,
            't4': 30.0,
            'number_chosen': 2.0,
        }
        self.one_row = Database('one_row', pd.DataFrame([first_row]))
        weight = Variable('orig_weight') * 1.7718243289995812

        # %
        # Parameters to be estimated
        cte_shopping = Beta('cte_shopping', -0.736, None, None, 0)
        cte_socializing = Beta('cte_socializing', -0.736, None, None, 0)
        cte_recreation = Beta('cte_recreation', -0.837, None, None, 0)

        number_members_socializing = Beta(
            'number_members_socializing', 0.0166, None, None, 0
        )
        number_members_recreation = Beta(
            'number_members_recreation', 0.0198, None, None, 0
        )

        metropolitan_shopping = Beta('metropolitan_shopping', 0.062, None, None, 0)

        male_shopping = Beta('male_shopping', 0.0979, None, None, 0)
        male_socializing = Beta('male_socializing', 0.127, None, None, 0)
        male_recreation = Beta('male_recreation', 0.219, None, None, 0)

        age_15_40_shopping = Beta('age_15_40_shopping', 0.0755, None, None, 0)
        age_15_40_recreation = Beta('age_15_40_recreation', 0.121, None, None, 0)

        age_41_60_socializing = Beta('age_41_60_socializing', -0.0646, None, None, 0)
        age_41_60_personal = Beta('age_41_60_personal', -0.0551, None, None, 0)

        bachelor_socializing = Beta('bachelor_socializing', -0.051, None, None, 0)
        bachelor_personal = Beta('bachelor_personal', -0.0455, None, None, 0)

        white_personal = Beta('white_personal', -0.0884, None, None, 0)

        spouse_shopping = Beta('spouse_shopping', 0.0628, None, None, 0)
        spouse_recreation = Beta('spouse_recreation', -0.0656, None, None, 0)

        employed_shopping = Beta('employed_shopping', 0.0462, None, None, 0)

        sunday_socializing = Beta('sunday_socializing', 0.0963, None, None, 0)
        sunday_personal = Beta('sunday_personal', 0.0907, None, None, 0)

        # %
        # Definition of the utility functions
        shopping = (
            cte_shopping
            + metropolitan_shopping * Variable('metro')
            + male_shopping * Variable('male')
            + age_15_40_shopping * Variable('age15_40')
            + spouse_shopping * Variable('spousepr')
            + employed_shopping * Variable('employed')
        )

        socializing = (
            cte_socializing
            + number_members_socializing * Variable('hhsize')
            + male_socializing * Variable('male')
            + age_41_60_socializing * Variable('age41_60')
            + bachelor_socializing * Variable('bachigher')
            + sunday_socializing * Variable('Sunday')
        )

        recreation = (
            cte_recreation
            + number_members_recreation * Variable('hhsize')
            + male_recreation * Variable('male')
            + age_15_40_recreation * Variable('age15_40')
            + spouse_recreation * Variable('spousepr')
        )

        personal = (
            age_41_60_personal * Variable('age41_60')
            + bachelor_personal * Variable('bachigher')
            + white_personal * Variable('white')
            + sunday_personal * Variable('Sunday')
        )

        baseline_utilities = {
            1: shopping,
            2: socializing,
            3: recreation,
            4: personal,
        }

        # Observed consumed quantities
        consumed_quantities = {
            1: Variable('t1') / 60.0,
            2: Variable('t2') / 60.0,
            3: Variable('t3') / 60.0,
            4: Variable('t4') / 60.0,
        }
        gamma_shopping = Beta('gamma_shopping', 3.49, 0.001, None, 0)
        gamma_socializing = Beta('gamma_socializing', 11.1, 0.001, None, 0)
        gamma_recreation = Beta('gamma_recreation', 15.4, 0.001, None, 0)
        gamma_personal = Beta('gamma_personal', 1.56, 0.001, None, 0)

        scale_parameter = Beta('scale', 3.83, 0.0001, None, 0)

        gamma_parameters = {
            1: gamma_shopping,
            2: gamma_socializing,
            3: gamma_recreation,
            4: gamma_personal,
        }

        alpha_parameter = {
            1: Beta('alpha_1', 0.5, 0, 1, 0),
            2: Beta('alpha_1', 0.5, 0, 1, 0),
            3: Beta('alpha_1', 0.5, 0, 1, 0),
            4: Beta('alpha_1', 0.5, 0, 1, 0),
        }

        prices = {1: Numeric(1), 2: Numeric(1), 3: Numeric(1), 4: Numeric(1)}

        self.the_gamma_profile_with_scale = GammaProfile(
            model_name='gamma_profile_with_scale',
            baseline_utilities=baseline_utilities,
            prices=prices,
            gamma_parameters=gamma_parameters,
            scale_parameter=scale_parameter,
            weights=weight,
        )

        self.the_translated = Translated(
            model_name='translated',
            baseline_utilities=baseline_utilities,
            gamma_parameters=gamma_parameters,
            alpha_parameters=alpha_parameter,
            scale_parameter=scale_parameter,
            weights=weight,
        )

        self.generalized = Generalized(
            model_name='generalized',
            baseline_utilities=baseline_utilities,
            gamma_parameters=gamma_parameters,
            alpha_parameters=alpha_parameter,
            scale_parameter=scale_parameter,
            prices=prices,
            weights=weight,
        )

        self.non_monotonic = NonMonotonic(
            model_name='non_monotonic',
            baseline_utilities=baseline_utilities,
            mu_utilities=baseline_utilities,
            gamma_parameters=gamma_parameters,
            alpha_parameters=alpha_parameter,
            scale_parameter=scale_parameter,
            weights=weight,
        )

    def test_gamma_profile(self):
        error_msg = self.the_gamma_profile_with_scale.validation(one_row=self.one_row)
        self.assertEqual(error_msg, [], '\n'.join(error_msg))

    def test_translated(self):
        error_msg = self.the_translated.validation(one_row=self.one_row)
        self.assertEqual(error_msg, [], '\n'.join(error_msg))

    def test_generalized(self):
        error_msg = self.generalized.validation(one_row=self.one_row)
        self.assertEqual(error_msg, [], '\n'.join(error_msg))

    def test_non_monotonic(self):
        error_msg = self.non_monotonic.validation(one_row=self.one_row)
        self.assertEqual(error_msg, [], '\n'.join(error_msg))
